"""
멀티턴 원인 분해 · 형식 재요청 분리 · 불만 분석.

핵심 설계: 멀티턴을 하나의 지표로 세면 안 된다.
'더 깊이 파고드는 멀티턴'과 '답을 못 받아 다시 묻는 멀티턴'은
방향이 정반대인데 합치면 서로를 상쇄한다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .schema import FORMAT_REQUEST_PATTERNS

# 후속 턴 유형
TURN_KINDS = {
    "REPEAT": "재질문 — 같은 의도·같은 대상 (실패 신호)",
    "FORMAT": "형식 재요청 — 내용 아닌 표현 방식 (부분 실패)",
    "REFINE": "세부화 — 같은 의도, 대상·조건 변경 (정상)",
    "PIVOT": "여정 진행 — 다른 의도로 이동 (정상·긍정)",
}
BAD_KINDS = ("REPEAT", "FORMAT")


def _slot_key(v) -> str:
    if isinstance(v, (list, tuple, set)):
        return "|".join(sorted(str(x) for x in v))
    return "" if pd.isna(v) else str(v)


def _char_bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", str(t))
    return {t[i:i + 2] for i in range(max(len(t) - 1, 0))}


def _text_sim(a, b) -> float:
    """문자 bigram 자카드. 형태소 분석기 없이 재질문 여부를 근사한다."""
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return 0.0
    x, y = _char_bigrams(a), _char_bigrams(b)
    return len(x & y) / len(x | y) if (x | y) else 0.0


def _looks_format(text) -> bool:
    if not isinstance(text, str):
        return False
    return any(p in text for p in FORMAT_REQUEST_PATTERNS)


def classify_followups(q: pd.DataFrame, gap_min: int = 10) -> pd.DataFrame:
    """
    세션 내 각 후속 턴을 4유형으로 분류하고, 직전 턴의 결과를 붙인다.

    query_text 가 있으면 형식 재요청을 원문으로 판정하고,
    없으면 '같은 의도·같은 슬롯 반복'만으로 REPEAT 를 잡는다(FORMAT 미분리).
    """
    d = q.sort_values(["session_id", "ts"]).copy()
    g = d.groupby("session_id")

    d["_slot"] = d["slot_target"].map(_slot_key)
    d["prev_intent"] = g["l2_intent"].shift(1)
    d["prev_slot"] = g["_slot"].shift(1)
    d["prev_outcome"] = g["outcome"].shift(1)
    d["prev_fail"] = g["fail_code"].shift(1)
    d["prev_ts"] = g["ts"].shift(1)
    d["gap_sec"] = (d["ts"] - d["prev_ts"]).dt.total_seconds()

    is_fu = d["prev_intent"].notna() & (d["gap_sec"] <= gap_min * 60)

    same_intent = d["l2_intent"].eq(d["prev_intent"])

    # slot_target 이 비어 있으면 same_slot 이 항상 True 가 되어
    # REPEAT 이 '같은 의도 반복'으로 퇴화하고 REFINE 은 0% 가 된다.
    # 이 경우 query_text 문자 bigram 유사도로 대체 판정한다.
    slot_filled = float((d["_slot"].fillna("") != "").mean())
    d.attrs["slot_filled"] = slot_filled
    if slot_filled >= 0.05:
        same_slot = d["_slot"].eq(d["prev_slot"])
        d.attrs["repeat_basis"] = "slot"
    elif "query_text" in d.columns and d["query_text"].notna().any():
        prev_q = g["query_text"].shift(1)
        same_slot = pd.Series(
            [_text_sim(a, b) >= 0.55 for a, b in zip(d["query_text"], prev_q)],
            index=d.index)
        d.attrs["repeat_basis"] = "query_text 유사도(0.55)"
    else:
        # 대체 근거가 없으면 REPEAT 을 만들지 않는다.
        # 억지로 같은 의도 반복을 REPEAT 으로 세면 지표가 통째로 부풀려진다.
        same_slot = pd.Series(False, index=d.index)
        d.attrs["repeat_basis"] = "판정 불가 — REPEAT 미생성"
    fmt = d["query_text"].map(_looks_format) if "query_text" in d.columns else pd.Series(
        False, index=d.index)

    kind = pd.Series(pd.NA, index=d.index, dtype="string")
    kind[is_fu & fmt.fillna(False)] = "FORMAT"
    kind[is_fu & kind.isna() & same_intent & same_slot] = "REPEAT"
    kind[is_fu & kind.isna() & same_intent] = "REFINE"
    kind[is_fu & kind.isna()] = "PIVOT"

    d["turn_kind"] = kind
    d["is_followup"] = is_fu
    return d.drop(columns=["_slot"])


def followup_cause_table(fu: pd.DataFrame) -> pd.DataFrame:
    """직전 턴 결과 → 후속 턴 유형의 조건부 분포. 멀티턴의 '원인' 표."""
    d = fu[fu["is_followup"]].copy()
    # StringDtype 비교는 NA 를 전파하므로 numpy bool 로 확정한 뒤 분기한다
    def _b(sr, how, arg):
        return getattr(sr, how)(arg).fillna(False).to_numpy(dtype=bool)

    d["직전결과"] = np.select(
        [_b(d["prev_outcome"], "eq", "blocked"),
         _b(d["prev_fail"], "isin", ["D1", "D2", "D3"]),
         _b(d["prev_fail"], "isin", ["T1", "T2"]),
         _b(d["prev_fail"], "isin", ["S1", "X1"]),
         _b(d["prev_fail"], "eq", "M1"),
         _b(d["prev_outcome"], "eq", "success")],
        ["차단(P3)", "데이터 부재", "툴 문제", "슬롯·맥락", "모델 오류", "성공"],
        default="기타")
    if d.empty:
        return pd.DataFrame({"안내": ["후속 턴 없음"]})
    ct = pd.crosstab(d["직전결과"], d["turn_kind"], normalize="index")
    ct["복구성 멀티턴"] = ct.reindex(columns=list(BAD_KINDS), fill_value=0).sum(axis=1)
    ct["n"] = d.groupby("직전결과").size()
    order = ["성공", "차단(P3)", "데이터 부재", "툴 문제", "슬롯·맥락", "모델 오류", "기타"]
    return ct.reindex([o for o in order if o in ct.index]).round(3)


def single_turn_failure_link(fu: pd.DataFrame) -> dict:
    """
    "멀티턴은 싱글턴이 실패해서 생기는가?" 를 두 수준에서 본다.

    - 의도 수준 상관: 해석은 참고용. 의도 간 비교라 생태학적 오류 위험이 있다.
    - 턴 수준 로지스틱(세션 FE 대용으로 사용자 더미 대신 의도 고정): 직전 턴 실패가
      '복구성 멀티턴' 확률을 얼마나 올리는지. 이쪽이 실제 답이다.
    """
    d = fu[fu["is_followup"]].copy()
    d["bad"] = d["turn_kind"].isin(BAD_KINDS).astype(int)
    d["prev_failed"] = (~d["prev_outcome"].eq("success")).astype(int)

    # (1) 의도 수준 상관
    by_intent = q_intent = None
    base = fu.groupby("l2_intent").apply(
        lambda x: pd.Series({
            "싱글턴실패율": float((~x["outcome"].eq("success")).mean()),
            "복구성멀티턴율": float(x["turn_kind"].isin(BAD_KINDS).mean()),
            "n": len(x)}), include_groups=False)
    base = base[base["n"] >= 30]
    r = float(base["싱글턴실패율"].corr(base["복구성멀티턴율"])) if len(base) > 3 else np.nan

    # (2) 턴 수준 — 의도 고정효과로 의도 간 이질성 제거
    #
    # 47개 의도를 그대로 더미로 넣으면 관측이 적거나 bad 가 전부 0/1 인 의도에서
    # 완전분리(perfect separation)가 나 추정치가 nan 이 된다.
    #  → 희소 의도를 _RARE 로 묶고, 분산 없는 의도는 제외한 뒤 적합한다.
    #  → 그래도 실패하면 l1_stage 로 고정효과 해상도를 낮춰 재시도한다.
    def _fit(frame, fe_col, min_n=50):
        f = frame.dropna(subset=["bad", "prev_failed", fe_col]).copy()
        vc = f[fe_col].value_counts()
        f["_fe"] = np.where(f[fe_col].isin(vc[vc >= min_n].index),
                            f[fe_col], "_RARE")
        var = f.groupby("_fe")["bad"].nunique()
        f = f[f["_fe"].isin(var[var > 1].index)]
        if len(f) < 200 or f["prev_failed"].nunique() < 2:
            return None
        try:
            return smf.logit("bad ~ prev_failed + C(_fe)", data=f).fit(disp=0)
        except Exception:
            return None

    m, basis = _fit(d, "l2_intent"), "l2_intent FE"
    if m is None or not np.isfinite(m.bse.get("prev_failed", np.nan)):
        m, basis = _fit(d, "l1_stage", min_n=100), "l1_stage FE (의도 FE 적합 실패)"
    if m is None or not np.isfinite(m.bse.get("prev_failed", np.nan)):
        try:
            m = smf.logit("bad ~ prev_failed", data=d.dropna(
                subset=["bad", "prev_failed"])).fit(disp=0)
            basis = "FE 없음 (교란 미통제 — 참고용)"
        except Exception:
            m, basis = None, "적합 실패"

    if m is not None and np.isfinite(m.bse.get("prev_failed", np.nan)):
        or_ = float(np.exp(m.params["prev_failed"]))
        ci = np.exp(m.conf_int().loc["prev_failed"])
        p = float(m.pvalues["prev_failed"])
    else:
        or_, ci, p = np.nan, (np.nan, np.nan), np.nan

    return {
        "적합 기준": basis,
        "의도수준 상관계수": r,
        "턴수준 오즈비(직전실패→복구성멀티턴)": or_,
        "95%CI": (float(ci[0]), float(ci[1])),
        "p값": p,
        "_by_intent": base.sort_values("싱글턴실패율", ascending=False),
        "주의": ("의도수준 상관은 생태학적 오류 위험이 있어 참고용. "
                 "판단은 의도 고정효과를 넣은 턴수준 오즈비로 하십시오."),
    }


def format_request_share(fu: pd.DataFrame) -> pd.DataFrame:
    """
    형식 재요청 분리 — '내용은 맞았는데 형태가 안 맞은' 건.
    이 비율이 높은 의도는 응답 템플릿(표/차트/길이) 문제이지 모델 문제가 아니다.
    """
    if "query_text" not in fu.columns or fu["query_text"].isna().all():
        return pd.DataFrame({"안내": ["query_text 없음 — 형식 재요청 분리 불가"]})
    d = fu[fu["is_followup"]]
    if d.empty:
        return pd.DataFrame({"안내": ["후속 턴 없음 — 세션 분할 기준을 확인하십시오"]})
    out = d.groupby("l2_intent").apply(
        lambda x: pd.Series({
            "후속턴수": len(x),
            "형식재요청율": float(x["turn_kind"].eq("FORMAT").mean()),
            "재질문율": float(x["turn_kind"].eq("REPEAT").mean()),
        }), include_groups=False)
    if out.empty or "후속턴수" not in out.columns:
        return pd.DataFrame({"안내": ["후속 턴 표본 부족 — 세션 분할(--session-gap) 확인"]})
    out = out[out["후속턴수"] >= 20]
    if out.empty:
        return pd.DataFrame({"안내": ["의도별 후속 턴 20건 미만 — 형식 재요청 분리 불가"]})
    return out.sort_values("형식재요청율", ascending=False).round(3)


def complaint_context(fu: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    불만·이관 요청 직전에 무슨 일이 있었는가.
    REC.complaint / REC.escalate / RISK.distress 앞 lookback 턴의 실패코드 분포를
    전체 기저율과 비교해 lift 로 본다.
    """
    targets = ("REC.complaint", "REC.escalate", "RISK.distress")
    d = fu.sort_values(["session_id", "ts"]).reset_index(drop=True)
    idx = d.index[d["l2_intent"].isin(targets)]
    if len(idx) == 0:
        return pd.DataFrame({"안내": ["불만·이관 질의 없음"]})

    prev = []
    for i in idx:
        sess = d.at[i, "session_id"]
        for j in range(max(0, i - lookback), i):
            if d.at[j, "session_id"] == sess:
                prev.append(j)
    if not prev:
        return pd.DataFrame({"안내": ["직전 턴 없음"]})

    sub = d.loc[prev]
    obs = sub["fail_code"].fillna("성공").value_counts(normalize=True)
    base = d["fail_code"].fillna("성공").value_counts(normalize=True)
    out = pd.DataFrame({"불만직전_비중": obs, "전체_비중": base}).dropna()
    out["lift"] = out["불만직전_비중"] / out["전체_비중"]
    out.attrs["n_complaint"] = int(len(idx))
    return out.sort_values("lift", ascending=False).round(3)
