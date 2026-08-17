"""
회수 잠재량 · 3대 요인 영향(프로텍터·의도분류·환각) · 의도별 출력 품질.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FAIL_CODES, FAIL_COST


# ------------------------------------------------------- 1. 방향별 회수율

def recovery_potential(q: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """
    실패코드(=개선 방향)별 회수 잠재량.

      회수 잠재 = 실패 질의량 × (성공 시 재방문율 − 실패 시 재방문율)

    ★ 이것은 인과 추정이 아니라 관측 기반 상한 근사입니다.
      성공/실패는 무작위 배정이 아니고, 성공한 질의는 애초에 답하기 쉬운
      질의였을 수 있습니다. 우선순위 비교용으로만 쓰고 절대 수치로 보고하지 마십시오.
    """
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    days = d.drop_duplicates(["user_id", "date"])[["user_id", "date"]].sort_values(
        ["user_id", "date"])
    days["gap"] = days.groupby("user_id")["date"].diff(-1).dt.days.abs()
    days["returned"] = (days["gap"] <= horizon_days).fillna(False).astype(int)
    d = d.merge(days[["user_id", "date", "returned"]], on=["user_id", "date"])

    umean = d.groupby("user_id")["returned"].transform("mean")
    d["ret_dm"] = d["returned"] - umean

    ok = d[d["outcome"].eq("success")]
    base_ok = float(ok["ret_dm"].mean())

    rows = []
    for code, grp in d[d["fail_code"].notna()].groupby("fail_code"):
        if code == "C1":
            continue
        vol = float(grp["sample_weight"].sum())
        lost = base_ok - float(grp["ret_dm"].mean())
        rows.append({
            "실패코드": code, "설명": FAIL_CODES[code][0], "담당": FAIL_CODES[code][1],
            "처방": FAIL_CODES[code][2], "실패질의량": vol,
            "재방문손실(pp)": lost, "회수잠재": vol * max(lost, 0),
            "해결비용": FAIL_COST[code],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["효율(잠재/비용)"] = out["회수잠재"] / out["해결비용"].clip(lower=.5)
    tot = out["회수잠재"].sum()
    out["회수비중"] = out["회수잠재"] / tot if tot else np.nan
    return out.sort_values("효율(잠재/비용)", ascending=False).round(4)


# ------------------------------------------------- 2. 3대 요인 영향

def protector_impact(q: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """
    차단의 영향을 '회수 성공 여부'로 갈라서 본다.
    차단 자체가 문제인지, 차단 후 대체 제공 실패(C3)가 문제인지 구분한다.
    """
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    days = d.drop_duplicates(["user_id", "date"])[["user_id", "date"]].sort_values(
        ["user_id", "date"])
    days["gap"] = days.groupby("user_id")["date"].diff(-1).dt.days.abs()
    days["returned"] = (days["gap"] <= horizon_days).fillna(False).astype(int)
    d = d.merge(days[["user_id", "date", "returned"]], on=["user_id", "date"])
    umean = d.groupby("user_id")["returned"].transform("mean")
    d["ret_dm"] = d["returned"] - umean

    fc = d["fail_code"].fillna("").to_numpy(dtype=object)
    d["구분"] = np.select(
        [fc == "C1", fc == "C3",
         d["outcome"].eq("success").fillna(False).to_numpy(dtype=bool)],
        ["차단·회수성공", "차단·회수실패", "성공"], default="기타실패")
    out = d.groupby("구분").agg(n=("ret_dm", "size"), 재방문편차=("ret_dm", "mean"))
    ref = out.loc["성공", "재방문편차"] if "성공" in out.index else 0.0
    out["성공대비"] = out["재방문편차"] - ref
    return out.reindex([i for i in
                        ["성공", "차단·회수성공", "차단·회수실패", "기타실패"]
                        if i in out.index]).round(4)


def intent_misclassification(q: pd.DataFrame, top: int = 15) -> dict:
    """
    운영 분류(intent_pred) vs 재어노테이션 정답(l2_intent).

    두 라벨이 같은 체계일 때만 '정확도'가 의미를 가진다.
    운영 분류가 기존 카테고리 체계(예: PLATFORM_USE)면 라벨 공간이 다르므로
    자동으로 크로스워크 모드(legacy_bridge)로 넘긴다.
    """
    from .schema import ALL_INTENTS
    if "intent_pred" not in q.columns or q["intent_pred"].isna().all():
        return {"안내": "intent_pred 없음 — 운영 분류 결과를 붙이면 활성화됩니다"}

    vals = set(q["intent_pred"].dropna().astype(str).unique())
    overlap = len(vals & set(ALL_INTENTS)) / max(len(vals), 1)
    if overlap < 0.3:
        return {"라벨공간불일치": True,
                "겹침비율": round(overlap, 3),
                "안내": ("intent_pred 가 신규 택소노미와 다른 라벨 체계입니다 "
                         f"(겹침 {overlap:.0%}). 정확도 대신 크로스워크로 대조합니다."),
                "_bridge": legacy_bridge(q, top=top)}
    d = q[q["intent_pred"].notna() & q["l2_intent"].notna()].copy()
    n_drop = len(q) - len(d)
    if len(d) == 0:
        return {"안내": "intent_pred / l2_intent 가 모두 있는 행이 없습니다"}
    # StringDtype 비교는 NA 를 전파하므로 문자열로 확정한 뒤 비교한다
    d["오분류"] = (d["intent_pred"].astype(str).str.strip()
                   != d["l2_intent"].astype(str).str.strip()).astype(int)

    acc = 1 - float(d["오분류"].mean())
    by_out = d.groupby("오분류").agg(
        n=("outcome", "size"),
        실패율=("outcome", lambda s: float((~s.eq("success")).mean())))

    conf = (d[d["오분류"].eq(1)].groupby(["l2_intent", "intent_pred"]).size()
              .rename("건수").reset_index()
              .sort_values("건수", ascending=False).head(top))

    by_intent = d.groupby("l2_intent").agg(
        n=("오분류", "size"), 오분류율=("오분류", "mean")).query("n >= 30")

    return {"전체정확도": round(acc, 4),
            "제외행": int(n_drop),
            "오분류여부별_실패율": by_out.round(4),
            "주요혼동쌍": conf.reset_index(drop=True),
            "의도별오분류율": by_intent.sort_values(
                "오분류율", ascending=False).head(top).round(4),
            "해석": ("오분류율 상위 의도는 어노테이션 가이드의 혼동 쌍 규칙을 "
                     "먼저 손봐야 할 후보입니다.")}


# 외부 근거가 필요한 단계. LEARN·RECOVER 는 내재 지식 응답이 정상이므로 제외한다.
GROUNDING_REQUIRED = ("DISCOVER", "EVALUATE", "MONITOR", "SETTLE", "EXECUTE")


def legacy_bridge(q: pd.DataFrame, top: int = 15) -> dict:
    """
    기존 운영 분류 ↔ 신규 택소노미 크로스워크 대조.

    세 가지를 본다.
      1) 정합률 — 기존 분류가 가리키는 신규 의도 집합 안에 정답이 들어가는가
      2) 분해도 — 기존 카테고리 하나가 신규 의도 몇 종으로 쪼개지는가
                  (재설계가 필요했다는 증거이자, 운영 라우팅의 해상도 부족 지점)
      3) 미포괄 — 기존 체계에 대응이 없던 신규 의도가 실제로 얼마나 발생하는가
                  (기존 분류로는 아예 잡히지 않던 수요)
    """
    from .schema import LEGACY_CROSSWALK, LEGACY_UNCOVERED, legacy_key
    d = q[q["intent_pred"].notna() & q["l2_intent"].notna()].copy()
    if d.empty:
        return {"안내": "대조 가능한 행이 없습니다"}
    # 그룹(INTENT_CATEGORY1) + 서브코드(INTENT_CATEGORY2) 2단 키.
    # BASIC 처럼 그룹에 따라 의미가 정반대인 코드가 있어 서브코드 단독 대조는 무의미하다.
    grp = (d["intent_pred_group"] if "intent_pred_group" in d.columns
           else pd.Series(None, index=d.index))
    d["_legacy"] = [legacy_key(g, s_) for g, s_ in zip(grp, d["intent_pred"])]
    d["_gold"] = d["l2_intent"].astype(str).str.strip()

    known = set(LEGACY_CROSSWALK)
    unmapped = (d.loc[~d["_legacy"].isin(known), "_legacy"]
                  .value_counts().head(20))

    m = d[d["_legacy"].isin(known)].copy()
    m["정합"] = [g in LEGACY_CROSSWALK[l] for l, g in zip(m["_legacy"], m["_gold"])]

    by_legacy = m.groupby("_legacy").agg(
        n=("정합", "size"), 정합률=("정합", "mean"),
        신규의도수=("_gold", "nunique"))
    # 분해도: 기존 카테고리가 신규 의도로 흩어진 정도 (정규화 엔트로피)
    ent = {}
    for lg, g in m.groupby("_legacy"):
        p_ = g["_gold"].value_counts(normalize=True)
        h = float(-(p_ * np.log(p_)).sum())
        ent[lg] = h / np.log(len(p_)) if len(p_) > 1 else 0.0
    by_legacy["분해도"] = pd.Series(ent)

    # 기존 분류가 놓치던 신규 의도
    unc = d[d["_gold"].isin(LEGACY_UNCOVERED)]
    unc_tab = (unc.groupby("_gold").agg(n=("_gold", "size"))
                 .assign(비중=lambda x: x["n"] / len(d))
                 .sort_values("n", ascending=False).head(top))

    mism = (m[~m["정합"]].groupby(["_legacy", "_gold"]).size()
              .rename("건수").reset_index()
              .sort_values("건수", ascending=False).head(top))

    return {"전체정합률": round(float(m["정합"].mean()), 4) if len(m) else np.nan,
            "매핑불가건수": int(len(d) - len(m)),
            "미매핑코드": unmapped,
            "기존분류별": by_legacy.sort_values("분해도", ascending=False).round(3),
            "정합실패쌍": mism.reset_index(drop=True),
            "기존체계_미포괄_의도": unc_tab.round(4),
            "2단키 사용": bool("intent_pred_group" in d.columns
                               and d["intent_pred_group"].notna().any()),
            "해석": ("분해도가 높은 기존 카테고리는 운영 라우팅의 해상도가 부족했던 "
                     "지점입니다. 미포괄 의도의 비중 합이 곧 '기존 분류로는 아예 "
                     "보이지 않던 수요'의 크기입니다.")}


def hallucination_risk(q: pd.DataFrame,
                       stages: tuple[str, ...] = GROUNDING_REQUIRED) -> dict:
    """
    환각은 프로덕션에서 직접 측정이 불가능하므로 두 층으로 접근한다.

      (1) 관측 프록시 — 툴이 필요했는데 호출 없이 응답한 건(= 파라메트릭 지식 응답)
          + 근거 미인용 비율
      (2) 검수 실측 — halluc_audit 컬럼이 있으면 샘플링 검수 결과로 프록시를 보정
    """
    d = q[q["l1_stage"].isin(stages)].copy()
    has_tool = "tool_called" in d.columns and d["tool_called"].notna().any()
    has_cite = "cited" in d.columns and d["cited"].notna().any()
    if not has_tool and not has_cite:
        return {"안내": "tool_called / cited 없음 — 환각 프록시 계산 불가"}

    d["needs_tool"] = d["tool_expected"].notna()
    d["no_tool_answer"] = (d["needs_tool"] & d["tool_called"].isna()
                           & d["outcome"].eq("success")) if has_tool else False
    d["uncited"] = (~d["cited"].fillna(False)) & d["outcome"].eq("success") if has_cite else False
    d["risk"] = (d["no_tool_answer"] | d["uncited"]).astype(int)

    by_intent = d.groupby("l2_intent").agg(
        n=("risk", "size"), 위험군비율=("risk", "mean"),
        무툴응답=("no_tool_answer", "mean") if has_tool else ("risk", "mean"),
    ).query("n >= 30").sort_values("위험군비율", ascending=False)

    res = {"전체_위험군비율": round(float(d["risk"].mean()), 4),
           "의도별": by_intent.head(15).round(4),
           "대상단계": list(stages),
           "해석": ("'툴 없이 성공 응답'은 모델이 내재 지식으로 답했다는 뜻이며 "
                    "시세·재무처럼 사실성이 필요한 의도에서 특히 위험합니다. "
                    "LEARN·RECOVER 는 내재 지식 응답이 정상이라 집계에서 제외했습니다.")}

    if "halluc_audit" in d.columns and d["halluc_audit"].notna().any():
        a = d[d["halluc_audit"].notna()]
        tab = pd.crosstab(a["risk"], a["halluc_audit"], normalize="index")
        res["검수대조"] = tab.round(3)
        res["프록시_정밀도"] = round(float(
            a[a["risk"].eq(1)]["halluc_audit"].mean()), 4)
    return res


# ------------------------------------------------- 3. 의도별 출력 품질

def response_quality(q: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """
    의도별 출력의 양과 질.

    분량은 많을수록 좋은 것이 아니다. 같은 의도 안에서
    '길이는 긴데 재질문율도 높다'면 정보가 아니라 장황함이다.
    """
    d = q[q["outcome"].eq("success")].copy()
    agg = {"n": ("outcome", "size")}
    if "response_len" in d.columns and d["response_len"].notna().any():
        agg |= {"길이_중앙": ("response_len", "median"),
                "길이_p90": ("response_len", lambda s: s.quantile(.9))}
    if "cited" in d.columns and d["cited"].notna().any():
        agg["근거인용률"] = ("cited", "mean")
    if "csat" in d.columns and d["csat"].notna().any():
        agg["CSAT"] = ("csat", "mean")
    agg["응답시간_p95"] = ("latency_ms", lambda s: s.quantile(.95))

    out = d.groupby("l2_intent").agg(**agg).query(f"n >= {min_n}")

    if "turn_kind" in q.columns:
        rq = q[q["is_followup"].fillna(False)].groupby("l2_intent")["turn_kind"].apply(
            lambda s: float(s.isin(["REPEAT", "FORMAT"]).mean()))
        out = out.join(pd.to_numeric(rq, errors="coerce").rename("복구성멀티턴율"))

    for c in ("길이_중앙", "길이_p90", "복구성멀티턴율"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if ("길이_중앙" in out.columns and "복구성멀티턴율" in out.columns
            and out["복구성멀티턴율"].notna().any()):
        # 시세 조회와 종목 분석은 적정 길이가 다르다. 같은 여정 단계 안에서만 비교한다.
        from .schema import INTENT_TO_STAGE
        out["_stage"] = out.index.map(INTENT_TO_STAGE)
        med = out.groupby("_stage")["길이_중앙"].transform("median")
        rq_med = out.groupby("_stage")["복구성멀티턴율"].transform("median")
        long_, bad = out["길이_중앙"] >= med, out["복구성멀티턴율"] >= rq_med
        out["진단"] = np.select(
            [long_ & bad, ~long_ & bad, long_],
            ["장황(길지만 미해결)", "부족(짧고 미해결)", "충실"], default="간결·충분")
        out = out.drop(columns="_stage")
    return out.round(3)
