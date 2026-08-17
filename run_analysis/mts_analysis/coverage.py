"""
커버리지 감사 · 실패 귀속.

핵심 규칙: M1(모델 오류)은 다른 모든 코드에 해당하지 않을 때만 부여되는 **잔차**다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FAIL_CODES, FAIL_COST, INTENT_TO_STAGE

# answerable → 실패코드 1차 매핑
_ANSWERABLE_TO_CODE = {
    "no_source": "D1",   # 중단 여부는 outage_date 로 D2 재분류
    "no_tool": "T1",
    "no_slot": "S1",
    "no_auth": "A1",
}


def tool_evidence(q: pd.DataFrame, min_calls: int = 3) -> dict:
    """
    로그에서 '이 의도를 처리하는 기능이 실제로 존재하는가'를 도출한다.

    tool_expected 가 어노테이션되지 않았으므로, 해당 의도에서 툴이 성공적으로
    호출된 이력이 있으면 기능이 있는 것으로 본다.

      기능 있음 + 이번엔 미호출 → 라우팅 실패(T2)     ← 분류 개선으로 해결
      기능 없음                → 미구현(T1)          ← 개발 과제
      기능 있음 + 호출됐으나 결과 없음 → 커버리지 결손(D3)

    이 구분이 없으면 '벡터DB에 일부 있는' 기능까지 미구현으로 잡힌다.
    """
    if "tool_called" not in q.columns:
        return {"has_tool": set(), "call_rate": {}}
    d = q[q["tool_called"].notna()]
    cnt = d.groupby("l2_intent").size()
    has = set(cnt[cnt >= min_calls].index)
    rate = (q.assign(_c=q["tool_called"].notna())
              .groupby("l2_intent")["_c"].mean().to_dict())
    return {"has_tool": has, "call_rate": rate}


def derive_fail_codes(q: pd.DataFrame,
                      outage_date: pd.Timestamp | None = None,
                      outage_sources: tuple[str, ...] = ("fnguide",),
                      overblock_flag: str | None = None,
                      evidence: dict | None = None,
                      compliance_intents: tuple[str, ...] = (
                          "EVAL.verdict", "DISC.recommend_open",
                          "MON.rebalance", "MON.loss_reaction")) -> pd.DataFrame:
    """
    실패 귀속 코드를 파생한다. 성공 건은 code=None.

    overblock_flag: 사후 인적 검수 결과 컬럼명(bool). 있으면 C2 판정에 사용.
    """
    q = q.copy()
    n = len(q)
    # NOTE: pandas StringDtype 의 비교 결과는 NA 를 전파하고, Series.mask 는 NA 조건을
    # True 로 취급한다. 조건을 모두 numpy bool 로 확정해 그 함정을 피한다.
    code = np.full(n, "", dtype=object)

    is_fail = q["outcome"].eq("fail").fillna(False).to_numpy(dtype=bool)
    is_blocked = q["outcome"].eq("blocked").fillna(False).to_numpy(dtype=bool)

    # 1) answerable 기반 1차 귀속
    mapped = q["answerable"].map(_ANSWERABLE_TO_CODE).fillna("").to_numpy(dtype=object)
    sel = is_fail & (mapped != "")
    code[sel] = mapped[sel]

    # 2) D1 → D2 재분류 (소스가 있었으나 중단된 경우)
    if outage_date is not None:
        has_src = q["source_expected"].map(
            lambda s: bool(set(s) & set(outage_sources))
            if isinstance(s, (list, tuple, set)) else False).to_numpy(dtype=bool)
        if not has_src.any():
            # source_expected 가 어노테이션되지 않은 경우: 알려진 의존 의도로 대체 판정
            from .schema import FNGUIDE_INTENTS
            has_src = q["l2_intent"].isin(FNGUIDE_INTENTS).to_numpy(dtype=bool)
        after = q["ts"].ge(pd.Timestamp(outage_date)).to_numpy(dtype=bool)
        code[(code == "D1") & has_src & after] = "D2"

    # 3) 컨텍스트 상속 실패 X1 (S1 중 followup 턴)
    is_fu = q["f6_turn"].eq("followup").fillna(False).to_numpy(dtype=bool)
    code[(code == "S1") & is_fu] = "X1"

    # 3-b) 기능 존재 증거로 T1(미구현) ↔ T2(라우팅 실패) 분리
    #      로그상 같은 의도에서 툴이 정상 호출된 적이 있으면 미구현이 아니다.
    ev = evidence or tool_evidence(q)
    if ev.get("has_tool"):
        has = q["l2_intent"].isin(ev["has_tool"]).to_numpy(dtype=bool)
        code[(code == "T1") & has] = "T2"
        # 데이터 계열도 마찬가지: 기능이 있는데 결과가 없으면 커버리지 결손
        code[(code == "D1") & has] = "D3"

    # 3-c) 컴플라이언스 성격 의도의 실패는 정책 문제이지 기술 문제가 아니다.
    #      매매 판단·추천·손실 대응 요구는 애초에 답하면 안 되는 질의이므로
    #      '기능 없음'이나 '인증 미비'로 귀속하면 개선 우선순위가 왜곡된다.
    is_p3 = q["f4_compliance"].eq("P3").fillna(False).to_numpy(dtype=bool)
    is_comp_intent = q["l2_intent"].isin(compliance_intents).to_numpy(dtype=bool)
    policy = (is_p3 | is_comp_intent) & is_fail
    code[policy & np.isin(code, ["T1", "T2", "A1", "D1", "D3", "S1", "M1"])] = "C3"

    # 4) 차단 건: C1 기본, 과차단 검수 결과 있으면 C2
    code[is_blocked] = "C1"
    if overblock_flag and overblock_flag in q.columns:
        ob = q[overblock_flag].fillna(False).to_numpy(dtype=bool)
        code[is_blocked & ob] = "C2"

    # 5) 잔차 → M1  (다른 어느 코드에도 해당하지 않는 실패만)
    code[is_fail & (code == "")] = "M1"

    q["fail_code"] = pd.Series(code, index=q.index).replace("", pd.NA).astype("string")
    return q


# 정보 제공으로 회수해서는 안 되는 의도 — 안전·상담 경로가 정답이므로 C3 대상 아님
NO_DATA_RECOVERY = ("MON.loss_reaction", "RISK.distress")


def flag_c3(q: pd.DataFrame, window_min: int = 10,
            exclude_intents: tuple[str, ...] = NO_DATA_RECOVERY) -> pd.DataFrame:
    """
    C3 = 차단 후 대체 제공 실패.
    P3 차단이 발생한 뒤 같은 세션의 window_min 내에
    P1/P2 성공 응답이 하나도 없으면 회수 실패로 본다.

    exclude_intents 는 '판단 재료 제공'이 정답이 아닌 의도(감정적 손실 호소,
    자기위해 암시)로, 안전 경로가 정답이므로 C3 로 세지 않는다.
    """
    q = q.sort_values(["session_id", "ts"]).copy()
    blocked = q[q["outcome"].eq("blocked") & q["f4_compliance"].eq("P3")
                & ~q["l2_intent"].isin(exclude_intents)]
    if blocked.empty:
        return q

    ok = q[q["outcome"].eq("success") & q["f4_compliance"].isin(["P1", "P2"])]
    ok_by_sess = {s: g["ts"].to_numpy() for s, g in ok.groupby("session_id")}

    recovered = []
    for _, r in blocked.iterrows():
        arr = ok_by_sess.get(r["session_id"])
        if arr is None:
            recovered.append(False)
            continue
        hi = r["ts"] + pd.Timedelta(minutes=window_min)
        recovered.append(bool(((arr > r["ts"].to_datetime64()) & (arr <= hi.to_datetime64())).any()))

    q.loc[blocked.index, "fail_code"] = np.where(recovered, "C1", "C3")
    return q


def coverage_matrix(q: pd.DataFrame) -> pd.DataFrame:
    """의도 × 실패코드 히트맵 (층화 역가중 적용)."""
    d = q[q["fail_code"].notna()]
    mat = (d.pivot_table(index="l2_intent", columns="fail_code",
                         values="sample_weight", aggfunc="sum", fill_value=0.0))
    vol = q.groupby("l2_intent")["sample_weight"].sum().rename("질의량")
    out = mat.join(vol, how="right").fillna(0.0)
    out["실패량"] = out[[c for c in out.columns if c in FAIL_CODES and c != "C1"]].sum(axis=1)
    out["실패율"] = (out["실패량"] / out["질의량"]).replace([np.inf, -np.inf], np.nan)
    return out.sort_values("질의량", ascending=False)


def priority(q: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """
    개선 우선순위 = 질의량 × 실패율 ÷ 해결비용.
    C1(정당 차단)은 제외한다 — 실패가 아니므로.
    """
    d = q[q["fail_code"].notna() & q["fail_code"].ne("C1")]
    g = (d.groupby(["l2_intent", "fail_code"])["sample_weight"].sum()
           .rename("실패량").reset_index())
    vol = q.groupby("l2_intent")["sample_weight"].sum().rename("질의량")
    g = g.join(vol, on="l2_intent")
    g["실패율"] = g["실패량"] / g["질의량"]
    g["비용"] = g["fail_code"].map(FAIL_COST)
    g["우선순위"] = g["실패량"] * g["실패율"] / g["비용"].clip(lower=.5)
    g["담당"] = g["fail_code"].map(lambda c: FAIL_CODES[c][1])
    g["처방"] = g["fail_code"].map(lambda c: FAIL_CODES[c][2])
    g["단계"] = g["l2_intent"].map(INTENT_TO_STAGE)
    return g.sort_values("우선순위", ascending=False).head(top).reset_index(drop=True)


def stage_funnel(q: pd.DataFrame) -> pd.DataFrame:
    """L1 여정 단계별 질의량·실패율·차단율. 퍼널 절단 지점 확인용."""
    d = q.copy()
    d["is_fail"] = d["fail_code"].notna() & ~d["fail_code"].isin(["C1"])
    d["is_block"] = d["outcome"].eq("blocked")
    g = d.groupby("l1_stage").apply(
        lambda x: pd.Series({
            "질의량": x["sample_weight"].sum(),
            "실패율": np.average(x["is_fail"], weights=x["sample_weight"]),
            "차단율": np.average(x["is_block"], weights=x["sample_weight"]),
        }), include_groups=False)
    order = ["LEARN", "DISCOVER", "EVALUATE", "EXECUTE",
             "MONITOR", "SETTLE", "SERVICE", "RECOVER"]
    return g.reindex([s for s in order if s in g.index])
