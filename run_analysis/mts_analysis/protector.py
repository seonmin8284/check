"""
Protector 차단 이후 여정 분석.

정책 도입 전후 비교는 불가능하다(차단 없던 기간이 없음).
대신 **차단이 발동한 시점을 기준으로 그 다음에 무슨 일이 일어나는가**를 본다.

세 층으로 나눠 본다.
  층1 즉시 — 차단 직후 다음 턴에서 사용자가 무엇을 하는가
  층2 세션 — 차단이 그 세션 전체를 망가뜨리는가
  층3 이후 — 다음 세션에서 질문하는 방식이 달라지는가

차단 정의는 두 가지를 **동시에** 계산해 교차 검증한다.
  flag : 원본 로그에 'P' 로 표기된 것 (사람이 붙인 표기, 불확실)
  func : 함수 호출·렌더 구조로 파생한 것 (기계적, 규칙 의존)
둘이 크게 어긋나면 어느 쪽도 단독으로 신뢰할 수 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .turns import BAD_KINDS

REACTION = {
    "RETRY": "같은 질문을 다시 (표현만 바꿔)",
    "REFRAME": "같은 대상을 안전한 방식으로 다시 물음 (우회)",
    "SWITCH": "다른 대상·다른 주제로 이동",
    "ESCALATE": "상담원 연결·불만 제기",
    "DROP": "대화 종료",
}
SAFE_LEVELS = ("P0", "P1", "P2")


# ═══════════════════════════════════ 0. 차단 정의 교차 검증

def block_definitions(q: pd.DataFrame) -> dict:
    """
    두 정의를 만들고 일치도를 잰다.

    일치도가 낮으면 각 층 분석을 **두 정의로 각각 돌려** 결론이 뒤집히는지
    확인해야 한다(민감도 분석).
    """
    n = len(q)
    func = q["outcome"].eq("blocked").fillna(False).to_numpy(dtype=bool)
    if "protector_flag" in q.columns and q["protector_flag"].notna().any():
        flag = q["protector_flag"].fillna(False).astype(bool).to_numpy()
        has_flag = True
    else:
        flag = np.zeros(n, dtype=bool)
        has_flag = False

    res = {"함수기반 비율": round(float(func.mean()), 4), "P표기 사용가능": has_flag}
    if has_flag:
        both = flag & func
        either = flag | func
        agree = float((flag == func).mean())
        # Cohen's kappa
        pe = (flag.mean() * func.mean()
              + (1 - flag.mean()) * (1 - func.mean()))
        kappa = (agree - pe) / (1 - pe) if pe < 1 else np.nan
        res |= {"P표기 비율": round(float(flag.mean()), 4),
                "둘 다": round(float(both.mean()), 4),
                "둘 중 하나": round(float(either.mean()), 4),
                "단순일치": round(agree, 4), "kappa": round(float(kappa), 4),
                "P만": int((flag & ~func).sum()),
                "함수만": int((func & ~flag).sum())}
        res["판정"] = (
            "두 정의가 잘 맞습니다. 어느 쪽을 써도 결론이 같습니다."
            if kappa >= .8 else
            "정의 간 불일치가 큽니다. 아래 분석을 두 정의로 각각 확인하고, "
            "결론이 뒤집히면 라벨을 먼저 정리해야 합니다.")
    res["_flag"] = flag
    res["_func"] = func
    return res


def blocked_series(q: pd.DataFrame, how: str = "func") -> pd.Series:
    """how: func | flag | both | either"""
    d = block_definitions(q)
    f, u = d["_flag"], d["_func"]
    arr = {"func": u, "flag": f, "both": f & u, "either": f | u}[how]
    return pd.Series(arr, index=q.index)


# ═══════════════════════════════════ 1. 차단 유형 분류

def block_types(q: pd.DataFrame, blocked: pd.Series,
                alt_markers=("대신", "참고", "확인해", "조회", "안내",
                             "아래", "링크", "메뉴", "다음")) -> pd.Series:
    """
    차단 응답을 사용자가 받는 내용으로 4분류.
    같은 '차단'이라도 대안을 줬는지에 따라 사용자 반응이 완전히 달라진다.
    """
    txt = (q["answer_text"] if "answer_text" in q.columns
           else q.get("ANSWER", pd.Series("", index=q.index))).fillna("").astype(str)
    from .relevance import strip_html
    plain = txt.map(strip_html)
    has_alt = plain.str.contains("|".join(alt_markers), regex=True, na=False)
    long_ = plain.str.len() >= 120
    has_html = txt.str.contains(r"<\s*(?:table|ul|li|div)", regex=True, na=False)

    out = pd.Series(pd.NA, index=q.index, dtype="string")
    b = blocked.fillna(False).to_numpy(dtype=bool)
    ha, lo, hh = has_alt.to_numpy(), long_.to_numpy(), has_html.to_numpy()
    out[b & hh] = "부분응답(자료 동반)"
    out[b & ~hh & ha & lo] = "거절+대체제시"
    out[b & ~hh & ha & ~lo] = "거절+짧은안내"
    out[b & ~hh & ~ha] = "완전거절"
    return out


# ═══════════════════════════════════ 2. 층1 — 즉시 반응

def immediate_reaction(fu: pd.DataFrame, blocked: pd.Series,
                       gap_min: int = 10) -> pd.DataFrame:
    """
    차단 직후 다음 턴에서 사용자가 무엇을 했는가.

    REFRAME(우회) 판정이 핵심: 같은 대상을 유지하면서 컴플라이언스 등급을
    낮춰 다시 물었다면, 사용자가 스스로 대안 경로를 찾은 것이다.
    우회가 적고 DROP 이 많은 의도가 곧 '막다른 길'이다.
    """
    d = fu.sort_values(["session_id", "ts"]).copy()
    d["_blk"] = blocked.reindex(d.index).fillna(False).astype(bool)
    g = d.groupby("session_id")
    d["_n_intent"] = g["l2_intent"].shift(-1)
    d["_n_f4"] = g["f4_compliance"].shift(-1)
    d["_n_ts"] = g["ts"].shift(-1)
    d["_n_kind"] = g["turn_kind"].shift(-1)

    def _slot(v):
        return "|".join(sorted(str(x) for x in v)) if isinstance(v, (list, tuple)) else ""
    d["_slot"] = d["slot_target"].map(_slot)
    d["_n_slot"] = g["_slot"].shift(-1)

    b = d[d["_blk"]].copy()
    if b.empty:
        return pd.DataFrame({"안내": ["차단 건이 없습니다"]})
    within = (b["_n_ts"] - b["ts"]).dt.total_seconds() <= gap_min * 60
    same_slot = b["_slot"].eq(b["_n_slot"]) & b["_slot"].ne("")
    same_intent = b["l2_intent"].eq(b["_n_intent"])
    safer = b["_n_f4"].isin(SAFE_LEVELS)
    esc = b["_n_intent"].isin(["REC.escalate", "REC.complaint"])

    b["반응"] = np.select(
        [b["_n_ts"].isna() | ~within.fillna(False),
         esc.fillna(False),
         (same_slot & safer).fillna(False),
         (same_intent & ~safer).fillna(False)],
        ["DROP", "ESCALATE", "REFRAME", "RETRY"], default="SWITCH")
    return b


def reaction_summary(b: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    if "반응" not in b.columns:
        return pd.DataFrame({"안내": ["차단 건 없음"]})
    if by is None:
        s = b["반응"].value_counts(normalize=True).rename("비율").to_frame()
        s["설명"] = [REACTION.get(i, "") for i in s.index]
        s["건수"] = b["반응"].value_counts()
        return s.round(4)
    t = pd.crosstab(b[by], b["반응"], normalize="index")
    t["건수"] = b.groupby(by).size()
    t = t[t["건수"] >= 20]
    if "REFRAME" in t.columns and "DROP" in t.columns:
        t["막다른길지수"] = t["DROP"] - t["REFRAME"]
        t = t.sort_values("막다른길지수", ascending=False)
    return t.round(3)


# ═══════════════════════════════════ 3. 층2 — 세션 영향

def dead_end(fu: pd.DataFrame, blocked: pd.Series) -> pd.DataFrame:
    """
    차단 이후 그 세션에서 끝내 아무 성공도 못 한 비율(막다른 길)과 회수 비용.
    차단 유형별로 갈라 보면 '대체 안내의 가치'가 숫자로 나온다.
    """
    d = fu.sort_values(["session_id", "ts"]).copy()
    d["_blk"] = blocked.reindex(d.index).fillna(False).astype(bool)
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    d["_rank"] = d.groupby("session_id").cumcount()

    first_blk = d[d["_blk"]].groupby("session_id").first()
    if first_blk.empty:
        return pd.DataFrame({"안내": ["차단 건이 없습니다"]})
    rows = []
    for sid, r in first_blk.iterrows():
        after = d[(d["session_id"] == sid) & (d["_rank"] > r["_rank"])]
        ok_after = after[after["_ok"]]
        rows.append({
            "session_id": sid, "유형": r.get("차단유형"),
            "l2_intent": r["l2_intent"],
            "이후턴수": len(after),
            "막다른길": len(ok_after) == 0,
            "회수턴": (int(ok_after["_rank"].iloc[0] - r["_rank"])
                       if len(ok_after) else np.nan),
            "회수초": (float((ok_after["ts"].iloc[0] - r["ts"]).total_seconds())
                       if len(ok_after) else np.nan),
        })
    return pd.DataFrame(rows)


def dead_end_summary(de: pd.DataFrame, by: str = "유형") -> pd.DataFrame:
    if "막다른길" not in de.columns or by not in de.columns:
        return pd.DataFrame({"안내": ["집계 불가"]})
    g = de.groupby(by, dropna=False).agg(
        건수=("막다른길", "size"), 막다른길비율=("막다른길", "mean"),
        평균회수턴=("회수턴", "mean"), 평균회수초=("회수초", "mean"))
    return g[g["건수"] >= 15].sort_values("막다른길비율", ascending=False).round(3)


def session_shift(fu: pd.DataFrame, blocked: pd.Series) -> dict:
    """차단을 겪은 세션에서, 차단 전 구간과 후 구간의 성격 변화."""
    d = fu.sort_values(["session_id", "ts"]).copy()
    d["_blk"] = blocked.reindex(d.index).fillna(False).astype(bool)
    d["_rank"] = d.groupby("session_id").cumcount()
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    d["_judge"] = d["f4_compliance"].isin(["P2", "P3"]).fillna(False)

    fb = d[d["_blk"]].groupby("session_id")["_rank"].min().rename("blk_rank")
    if fb.empty:
        return {"안내": "차단 건이 없습니다"}
    d = d.join(fb, on="session_id").dropna(subset=["blk_rank"])
    pre = d[d["_rank"] < d["blk_rank"]]
    post = d[d["_rank"] > d["blk_rank"]]
    if pre.empty or post.empty:
        return {"안내": "차단 앞뒤 구간이 모두 있는 세션이 부족합니다"}

    def _agg(x):
        return {"성공률": float(x["_ok"].mean()),
                "판단성비중": float(x["_judge"].mean()),
                "의도가짓수": float(x.groupby("session_id")["l2_intent"]
                                    .nunique().mean())}
    a, b_ = _agg(pre), _agg(post)
    return {"차단 전": a, "차단 후": b_,
            "변화": {k: round(b_[k] - a[k], 4) for k in a},
            "대상세션": int(d["session_id"].nunique()),
            "해석": ("차단 후 성공률과 의도 가짓수가 함께 떨어지면 "
                     "차단이 그 질문 하나가 아니라 세션 전체를 닫는 것입니다.")}


# ═══════════════════════════════════ 4. 층3 — 이후 세션 (이벤트 스터디)

def session_event_study(q: pd.DataFrame, blocked: pd.Series,
                        span: int = 3, min_n: int = 30) -> pd.DataFrame:
    """
    차단이 처음 발생한 세션을 0으로 두고, 상대 세션 번호별 질문 성격 변화.

    -2, -1 구간이 평평해야 차단 이후 변화를 차단 탓으로 볼 수 있다.
    차단 전부터 이미 줄고 있었다면 그것은 자기검열이 아니라 다른 추세다.
    """
    d = q.sort_values(["user_id", "ts"]).copy()
    d["_blk"] = blocked.reindex(d.index).fillna(False).astype(bool)
    start = d.groupby("session_id")["ts"].transform("min")
    order = (d[["user_id", "session_id"]].assign(t=start)
               .drop_duplicates("session_id").sort_values(["user_id", "t"]))
    order["seq"] = order.groupby("user_id").cumcount()
    d = d.merge(order[["session_id", "seq"]], on="session_id", how="left")

    fb = (d[d["_blk"]].groupby("user_id")["seq"].min().rename("blk_seq"))
    if fb.empty:
        return pd.DataFrame({"안내": ["차단 건이 없습니다"]})
    d = d.join(fb, on="user_id").dropna(subset=["blk_seq"])
    d["rel"] = (d["seq"] - d["blk_seq"]).astype(int)
    d = d[d["rel"].between(-span, span)]
    d["_judge"] = d["f4_compliance"].isin(["P2", "P3"]).fillna(False)
    d["_ok"] = d["outcome"].eq("success").fillna(False)

    g = d.groupby("rel").agg(
        질의수=("_judge", "size"), 사용자수=("user_id", "nunique"),
        판단성질의비중=("_judge", "mean"), 성공률=("_ok", "mean"),
        의도가짓수=("l2_intent", "nunique"))
    g = g[g["질의수"] >= min_n]
    if "판단성질의비중" in g.columns and 0 in g.index:
        g["기준대비"] = g["판단성질의비중"] - g.loc[0, "판단성질의비중"]
    return g.round(4)


# ═══════════════════════════════════ 5. 매칭 — 차단 vs 통과

def matched_p3(q: pd.DataFrame, blocked: pd.Series,
               min_cell: int = 5) -> dict:
    """
    같은 P3 질문인데 차단된 건 vs 통과한 건을 층화 매칭해 비교한다.

    Protector 가 GPT-4 mini 기준이라 컴플라이언스 대상 질의도 일부 통과한다.
    통과 여부의 변동이 사용자와 무관하다면 준-무작위 배정이 되어,
    전후 비교 없이도 차단의 효과를 추정할 수 있다.

    층: 의도 × 주차 × 세션 내 턴 순번(0 / 1~2 / 3+)
    """
    d = q.copy()
    d["_blk"] = blocked.reindex(d.index).fillna(False).astype(bool)
    p3 = d[d["f4_compliance"].eq("P3")].copy()
    if p3.empty:
        return {"안내": "P3 라벨 질의가 없습니다"}
    n_b, n_p = int(p3["_blk"].sum()), int((~p3["_blk"]).sum())
    if n_p < 50:
        return {"안내": f"P3 통과 건이 {n_p}건뿐이라 매칭 대조군을 만들 수 없습니다",
                "차단": n_b, "통과": n_p}

    p3["_week"] = p3["ts"].dt.to_period("W").dt.start_time
    p3 = p3.sort_values(["session_id", "ts"])
    p3["_rank"] = p3.groupby("session_id").cumcount()
    p3["_rb"] = pd.cut(p3["_rank"], [-1, 0, 2, 999],
                       labels=["첫턴", "2~3턴", "4턴+"])
    p3["_stratum"] = (p3["l2_intent"].astype(str) + "|"
                      + p3["_week"].astype(str) + "|" + p3["_rb"].astype(str))

    keep = []
    for st, g in p3.groupby("_stratum"):
        if g["_blk"].sum() >= min_cell and (~g["_blk"]).sum() >= min_cell:
            keep.append(g)
    if not keep:
        return {"안내": "차단·통과가 함께 있는 층이 없습니다. "
                        "층 조건을 완화하거나 기술 통계로만 보십시오",
                "차단": n_b, "통과": n_p}
    m = pd.concat(keep)
    return {"매칭표본": int(len(m)), "층수": int(m["_stratum"].nunique()),
            "차단": int(m["_blk"].sum()), "통과": int((~m["_blk"]).sum()),
            "전체차단": n_b, "전체통과": n_p, "_matched": m}


def matched_balance(m: pd.DataFrame) -> pd.DataFrame:
    """매칭 후 두 군이 실제로 비슷한지 확인. 어긋나면 결과 해석에 주의."""
    d = m.copy()
    d["_len"] = (d["query_text"].fillna("").astype(str).str.len()
                 if "query_text" in d.columns else np.nan)
    d["_has_tgt"] = d["slot_target"].map(
        lambda v: isinstance(v, (list, tuple)) and len(v) > 0)
    d["_prev_blk"] = (d.sort_values("ts").groupby("user_id")["_blk"]
                        .transform(lambda s: s.shift(1).fillna(False).cumsum()))
    g = d.groupby("_blk").agg(
        건수=("_len", "size"), 질문길이=("_len", "mean"),
        종목지정비율=("_has_tgt", "mean"), 이전차단횟수=("_prev_blk", "mean"))
    g.index = ["통과", "차단"][: len(g)]
    return g.round(3)


def matched_outcome(m: pd.DataFrame, fu: pd.DataFrame,
                    sess: pd.DataFrame | None = None) -> pd.DataFrame:
    """층별 가중 평균으로 차단군 vs 통과군의 결과 차이를 낸다."""
    d = m.copy()
    nxt = fu.sort_values(["session_id", "ts"]).copy()
    nxt["_n_kind"] = nxt.groupby("session_id")["turn_kind"].shift(-1)
    nxt["_n_ts"] = nxt.groupby("session_id")["ts"].shift(-1)
    d = d.merge(nxt[["query_id", "_n_kind", "_n_ts"]], on="query_id", how="left")
    d["_drop"] = d["_n_ts"].isna()
    d["_bad_next"] = d["_n_kind"].isin(BAD_KINDS).fillna(False)

    if sess is not None and "session_outcome" in sess.columns:
        so = sess["session_outcome"].rename("_so")
        d = d.join(so, on="session_id")
        d["_sess_ok"] = d["_so"].isin(["RESOLVED", "RESOLVED_HARD"])
    else:
        d["_sess_ok"] = np.nan

    rows = []
    for st, g in d.groupby("_stratum"):
        b, p_ = g[g["_blk"]], g[~g["_blk"]]
        if len(b) == 0 or len(p_) == 0:
            continue
        w = len(g)
        for col, lab in [("_drop", "대화 종료율"), ("_bad_next", "되묻기율"),
                         ("_sess_ok", "세션 해결률")]:
            if g[col].notna().any():
                rows.append({"지표": lab, "층": st, "w": w,
                             "차단": float(b[col].mean()),
                             "통과": float(p_[col].mean())})
    if not rows:
        return pd.DataFrame({"안내": ["비교 가능한 층이 없습니다"]})
    t = pd.DataFrame(rows)
    out = (t.groupby("지표")
             .apply(lambda x: pd.Series({
                 "차단군": np.average(x["차단"], weights=x["w"]),
                 "통과군": np.average(x["통과"], weights=x["w"]),
                 "층수": len(x)}), include_groups=False))
    out["차이"] = out["차단군"] - out["통과군"]
    return out.round(4)


# ═══════════════════════════════════ 6. 정책 일관성

def policy_consistency(q: pd.DataFrame, blocked: pd.Series,
                       min_n: int = 30) -> pd.DataFrame:
    """
    같은 성격(P3)의 질문인데 차단률이 시기·의도에 따라 얼마나 흔들리는가.

    분산이 크면 사용자 입장에서는 '어떤 날은 되고 어떤 날은 안 되는' 서비스다.
    준법 관점에서는 통과된 P3 가 곧 점검 대상이다.
    """
    d = q[q["f4_compliance"].eq("P3")].copy()
    if d.empty:
        return pd.DataFrame({"안내": ["P3 질의가 없습니다"]})
    d["_blk"] = blocked.reindex(d.index).fillna(False).astype(bool)
    d["_month"] = d["ts"].dt.to_period("M").dt.start_time

    by_i = d.groupby("l2_intent").agg(건수=("_blk", "size"), 차단률=("_blk", "mean"))
    by_i = by_i[by_i["건수"] >= min_n]
    by_m = d.groupby("_month").agg(건수=("_blk", "size"), 차단률=("_blk", "mean"))
    by_m = by_m[by_m["건수"] >= min_n]
    out = by_i.sort_values("차단률").round(3)
    out.attrs["월별"] = by_m.round(3)
    out.attrs["월별_변동폭"] = (float(by_m["차단률"].max() - by_m["차단률"].min())
                                if len(by_m) > 1 else np.nan)
    out.attrs["통과건수"] = int((~d["_blk"]).sum())
    return out
