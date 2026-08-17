"""
세션 단위 결과 · North Star · 의도 포기율 · 수요-공급 맵.

여기까지의 모듈은 전부 '턴'이 관측 단위였다. 그런데 사용자가 경험하는 단위는
세션이다. 3번 재질문 끝에 답을 얻은 세션은 턴 기준으로는 실패 2·성공 1이지만
사용자에게는 '목적을 달성한 한 번'이다. 이 불일치를 여기서 메운다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .turns import BAD_KINDS

SESSION_OUTCOMES = {
    "RESOLVED": "목적 달성 (마찰 없음)",
    "RESOLVED_HARD": "달성했으나 재질문·다턴 소요",
    "ABANDONED": "미달성 후 종료",
    "DEFLECTED": "챗봇 포기 — 상담원·불만으로 전환",
}
SUCCESS_OUTCOMES = ("RESOLVED", "RESOLVED_HARD")
DEFLECT_INTENTS = ("REC.escalate", "REC.complaint")


# ------------------------------------------------------- 1. 세션 단위 결과

def session_outcomes(fu: pd.DataFrame, hard_turns: int = 3,
                     strict: bool = True) -> pd.DataFrame:
    """
    세션별 결과와 노력 비용(effort)을 산출한다.

    strict=True (기본): **분류 실패 후 폴백으로 만들어진 응답은 성공으로 세지 않는다.**
      폴백 구조에서는 질문과 무관한 답도 렌더링되어 outcome=success 가 되므로,
      이를 그대로 세면 세션 해결률이 실제보다 크게 부풀려진다.
      North Star 를 이 위에 세우면 개선을 추적할 수 없다.

    판정 순서: DEFLECTED → RESOLVED → RESOLVED_HARD → ABANDONED
    """
    from .relevance import FALLBACK_TOOLS, OTH_CODES
    d = fu.sort_values(["session_id", "ts"]).copy()
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    d["_fallback"] = False
    if strict:
        if "tool_called" in d.columns:
            tc = d["tool_called"].fillna("").astype(str).str.lower()
            d["_fallback"] = tc.apply(
                lambda x: any(f in x for f in FALLBACK_TOOLS))
        if "intent_pred" in d.columns:
            d["_fallback"] = d["_fallback"] | d["intent_pred"].astype(str) \
                .str.upper().isin([c.upper() for c in OTH_CODES])
        d["_ok"] = d["_ok"] & ~d["_fallback"]
    d["_bad_turn"] = d["turn_kind"].isin(BAD_KINDS).fillna(False)
    d["_deflect"] = d["l2_intent"].isin(DEFLECT_INTENTS).fillna(False)

    g = d.groupby("session_id")
    s = pd.DataFrame({
        "user_id": g["user_id"].first(),
        "start_ts": g["ts"].min(),
        "turns": g.size(),
        "has_success": g["_ok"].any(),
        "repair_turns": g["_bad_turn"].sum(),
        "deflected": g["_deflect"].any(),
        "blocked_any": g["outcome"].apply(lambda x: bool(x.eq("blocked").any())),
        "fallback_turns": g["_fallback"].sum(),
        "first_intent": g["l2_intent"].first(),
        "first_stage": g["l1_stage"].first(),
    })

    # 목적 달성까지의 노력: 첫 성공 턴의 순번과 경과 시간
    d["_rank"] = g.cumcount() + 1
    ok = d[d["_ok"]]
    if len(ok):
        first_ok = ok.groupby("session_id").first()
        s["turns_to_success"] = first_ok["_rank"]
        s["secs_to_success"] = (
            first_ok["ts"] - s.loc[first_ok.index, "start_ts"]).dt.total_seconds()
        s["success_intent"] = first_ok["l2_intent"]
    else:
        s[["turns_to_success", "secs_to_success", "success_intent"]] = np.nan

    easy = s["has_success"] & (s["repair_turns"] == 0) & (s["turns"] < hard_turns)
    s["session_outcome"] = np.select(
        [s["deflected"], easy, s["has_success"]],
        ["DEFLECTED", "RESOLVED", "RESOLVED_HARD"], default="ABANDONED")
    return s


def effort_by_intent(sess: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """
    의도별 노력 비용. 1턴 성공과 4턴 성공은 같은 '성공'이 아니다.
    설문 없이 얻는 CES(Customer Effort Score)의 행동 프록시.
    """
    d = sess[sess["session_outcome"].isin(SUCCESS_OUTCOMES)]
    if d.empty:
        return pd.DataFrame({"안내": ["성공 세션 없음"]})
    g = d.groupby("first_intent").agg(
        n=("turns", "size"),
        평균턴=("turns_to_success", "mean"),
        중앙턴=("turns_to_success", "median"),
        p90턴=("turns_to_success", lambda x: x.quantile(.9)),
        평균초=("secs_to_success", "mean"),
        재질문턴=("repair_turns", "mean"))
    g = g[g["n"] >= min_n]
    g["고비용비율"] = (d.groupby("first_intent")["session_outcome"]
                        .apply(lambda x: float(x.eq("RESOLVED_HARD").mean())))
    return g.sort_values("고비용비율", ascending=False).round(2)


# ------------------------------------------------------- 2. North Star

def north_star(sess: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    추적 지표.

    ★ '주 1회라도 성공한 사용자 비율'은 세션이 여러 번이면 대부분 충족되어
      천장에 붙는다. 1회성 사용자가 다수인 제품에서는 특히 그렇다.
      따라서 **무마찰 해결률(RESOLVED 세션 비율)** 을 주 추적 지표로 쓰고,
      NorthStar 는 보조로만 본다. 컬럼 순서가 그 우선순위를 반영한다.
    """
    d = sess.copy()
    d["period"] = d["start_ts"].dt.to_period(freq).dt.start_time
    d["_ok"] = d["session_outcome"].isin(SUCCESS_OUTCOMES)

    rows = []
    for per, g in d.groupby("period"):
        users = g["user_id"].nunique()
        ok_users = g.loc[g["_ok"], "user_id"].nunique()
        easy = g.loc[g["session_outcome"].eq("RESOLVED"), "user_id"].nunique()
        rows.append({
            "period": per, "진입사용자": users, "세션수": len(g),
            "무마찰해결률": float(g["session_outcome"].eq("RESOLVED").mean()),
            "세션해결률": float(g["_ok"].mean()),
            "무마찰달성사용자": easy / users if users else np.nan,
            "NorthStar(보조)": ok_users / users if users else np.nan,
            "폴백세션비율": (float((g["fallback_turns"] > 0).mean())
                             if "fallback_turns" in g.columns else np.nan),
            "이탈세션률": float(g["session_outcome"].eq("ABANDONED").mean()),
            "전환이탈률": float(g["session_outcome"].eq("DEFLECTED").mean()),
        })
    return pd.DataFrame(rows).set_index("period").round(4)


def turn_vs_session_gap(fu: pd.DataFrame, sess: pd.DataFrame) -> dict:
    """
    턴 기준 성공률과 세션 기준 성공률의 괴리.
    괴리가 크면 '턴은 자주 실패하지만 결국은 해결되는' 상태이고,
    턴 지표만 보고 있으면 실제보다 비관적으로 판단하게 된다.
    """
    turn_rate = float(fu["outcome"].eq("success").mean())
    strict_turn = (float((sess["turns"].sum() - sess["fallback_turns"].sum())
                         / sess["turns"].sum())
                   if "fallback_turns" in sess.columns else np.nan)
    sess_rate = float(sess["session_outcome"].isin(SUCCESS_OUTCOMES).mean())
    hard = float(sess["session_outcome"].eq("RESOLVED_HARD").mean())
    return {"턴 성공률(폴백 포함)": round(turn_rate, 4),
            "폴백 제외 턴 비율": round(strict_turn, 4) if strict_turn == strict_turn else np.nan,
            "세션 성공률": round(sess_rate, 4),
            "괴리": round(sess_rate - turn_rate, 4),
            "고비용 성공 비율": round(hard, 4),
            "해석": ("괴리가 크면 재질문으로 결국 해결되는 구조입니다. "
                     "품질 과제가 '못 답한다'가 아니라 '한 번에 못 답한다'로 바뀝니다.")}


# ------------------------------------------------------- 3. 의도 포기율

def _session_intent_panel(q: pd.DataFrame) -> pd.DataFrame:
    """(사용자, 의도, 세션) 단위 패널 + 세션 순번. 포기 판정의 관측 단위."""
    d = q.sort_values(["user_id", "ts"]).copy()
    sess_start = d.groupby("session_id")["ts"].transform("min")
    order = (d[["user_id", "session_id"]].assign(t=sess_start)
               .drop_duplicates("session_id").sort_values(["user_id", "t"]))
    order["sess_seq"] = order.groupby("user_id").cumcount()
    d = d.merge(order[["session_id", "sess_seq"]], on="session_id", how="left")

    d["_ok"] = d["outcome"].eq("success").fillna(False)
    p = (d.groupby(["user_id", "l2_intent", "sess_seq"])["_ok"]
           .any().rename("성공").reset_index())
    p["max_seq"] = p["user_id"].map(d.groupby("user_id")["sess_seq"].max())
    p = p.sort_values(["user_id", "l2_intent", "sess_seq"])
    # 같은 의도를 '이후 세션'에서 다시 시도했는가 (세션 내 재질문은 제외)
    p["재시도"] = p.groupby(["user_id", "l2_intent"])["sess_seq"].shift(-1).notna()
    # 이후 세션 자체가 없으면 재시도 기회가 없으므로 판정 대상에서 제외
    p["_eligible"] = p["sess_seq"] < p["max_seq"]
    return p


def intent_abandonment(q: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """
    사용자는 서비스를 떠나기 전에 특정 '의도'를 먼저 포기한다.

      재시도율 = 의도 i 를 시도한 (사용자, 세션) 중,
                 **이후 세션에서** i 를 다시 시도한 비율

    ★ 세션 내 재질문은 재시도가 아니라 복구 시도이므로 제외한다.
      (그것은 turns.classify_followups 의 REPEAT 로 따로 측정된다)
    성공 세션의 재시도율을 기준선으로 두고 그 차이를 '포기효과'로 본다.
    """
    p = _session_intent_panel(q)
    e = p[p["_eligible"]]
    if e.empty:
        return pd.DataFrame({"안내": ["이후 세션이 있는 관측이 없습니다"]})

    g = e.groupby(["l2_intent", "성공"])["재시도"].agg(["size", "mean"]).reset_index()
    piv = g.pivot(index="l2_intent", columns="성공", values="mean")
    cnt = g.pivot(index="l2_intent", columns="성공", values="size")
    if piv.shape[1] < 2:
        return pd.DataFrame({"안내": ["성공/실패 양쪽 관측이 필요합니다"]})
    piv.columns = ["실패후_재시도율", "성공후_재시도율"]
    cnt.columns = ["n_실패", "n_성공"]
    out = piv.join(cnt).dropna()
    out = out[out["n_실패"] >= min_n]
    if out.empty:
        return pd.DataFrame({"안내": [f"의도별 실패 세션 {min_n}건 미만"]})
    out["포기효과"] = out["성공후_재시도율"] - out["실패후_재시도율"]
    return out.sort_values("포기효과", ascending=False).round(3)


def abandonment_to_churn(q: pd.DataFrame, end_date, grace_days: int = 30,
                         min_pairs: int = 2) -> dict:
    """
    의도 포기가 사용자 이탈의 선행 신호인지 확인한다.

    실패를 겪은 (의도, 세션) 만을 분모로 사용자별 재시도 비율을 구한 뒤,
    그 비율의 중앙값으로 두 군을 나눠 이탈률을 비교한다.
    (실패 경험이 없는 조합을 분모에 넣으면 전원이 '포기'로 몰린다)
    """
    end_date = pd.Timestamp(end_date)
    p = _session_intent_panel(q)
    fails = p[p["_eligible"] & ~p["성공"]]
    if fails.empty:
        return {"안내": "실패 세션 없음"}

    per_user = fails.groupby("user_id").agg(
        실패조합=("재시도", "size"), 재시도비율=("재시도", "mean"))
    per_user = per_user[per_user["실패조합"] >= min_pairs]
    if len(per_user) < 30:
        return {"안내": f"실패 경험 {min_pairs}건 이상인 사용자가 부족합니다"}

    last = q.groupby("user_id")["ts"].max().dt.normalize()
    per_user["이탈"] = ((end_date - last.reindex(per_user.index)).dt.days
                        > grace_days)
    med = per_user["재시도비율"].median()
    per_user["구분"] = np.where(per_user["재시도비율"] > med,
                                "재시도 많음", "포기 많음")
    tab = per_user.groupby("구분").agg(
        n=("이탈", "size"), 평균재시도율=("재시도비율", "mean"),
        이탈률=("이탈", "mean")).round(4)
    gap = (tab.loc["포기 많음", "이탈률"] - tab.loc["재시도 많음", "이탈률"]
           if len(tab) == 2 else np.nan)
    return {"표": tab, "분할기준(중앙값)": round(float(med), 3),
            "이탈률 차이": round(float(gap), 4) if gap == gap else np.nan,
            "해석": ("차이가 양수로 크면 '의도 포기'가 이탈의 선행 지표입니다. "
                     "이탈을 기다리지 말고 포기율을 조기 경보로 쓰십시오.")}


# ------------------------------------------------------- 4. 수요-공급 맵

def demand_supply_map(q: pd.DataFrame, min_n: int = 30,
                      trend_weeks: int = 8) -> pd.DataFrame:
    """
    의도를 질의량 × 성공률 4사분면에 배치하고, 질의량 추세를 함께 본다.

    ★ 핵심은 '수요 억눌림' 사분면이다.
      계속 실패하면 사용자가 아예 묻지 않게 되어 질의량이 줄고,
      그러면 실패량 기준 우선순위에서 밀린다. 자기실현적 축소가 일어난다.
      성공률이 낮으면서 질의량이 '감소 중'인 의도가 여기에 해당한다.
    """
    d = q.copy()
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    g = d.groupby("l2_intent").agg(
        질의량=("sample_weight", "sum"),
        사용자수=("user_id", "nunique"),
        성공률=("_ok", "mean"))
    g = g[g["질의량"] >= min_n]
    if g.empty:
        return pd.DataFrame({"안내": ["표본 부족"]})

    # 최근 trend_weeks 구간의 주간 질의량 추세 (정규화 기울기)
    d["week"] = d["ts"].dt.to_period("W").dt.start_time
    wk = d.groupby(["l2_intent", "week"])["sample_weight"].sum().reset_index()
    cutoff = wk["week"].max() - pd.Timedelta(weeks=trend_weeks)
    wk = wk[wk["week"] >= cutoff]
    slopes = {}
    for it, gg in wk.groupby("l2_intent"):
        if len(gg) >= 4 and gg["sample_weight"].mean() > 0:
            x = np.arange(len(gg))
            b = np.polyfit(x, gg["sample_weight"].to_numpy(), 1)[0]
            slopes[it] = b / gg["sample_weight"].mean()   # 주당 상대 변화율
    g["질의량추세"] = pd.Series(slopes)

    vol_med, ok_med = g["질의량"].median(), g["성공률"].median()
    hi_vol, hi_ok = g["질의량"] >= vol_med, g["성공률"] >= ok_med
    shrinking = g["질의량추세"] < -0.02

    g["사분면"] = np.select(
        [hi_vol & hi_ok, hi_vol & ~hi_ok,
         ~hi_vol & ~hi_ok & shrinking, ~hi_vol & ~hi_ok],
        ["유지·방어", "★최우선 개선", "⚠수요 억눌림 의심", "저수요·저성공"],
        default="저수요·고성공")
    return g.sort_values(["사분면", "질의량"], ascending=[True, False]).round(3)


def suppressed_demand(dsm: pd.DataFrame) -> pd.DataFrame:
    """수요 억눌림 의심 의도만 추려낸다. 질의량 기준 우선순위에서 놓치는 지점."""
    if "사분면" not in dsm.columns:
        return dsm
    s = dsm[dsm["사분면"].eq("⚠수요 억눌림 의심")]
    return s.sort_values("질의량추세")
