"""
B트랙 — 개인 패널 분석 (로그인 ID 필요).

1. shift-share 분해: 의도 비중 변화를 within(행동) / between·entry·exit(구성)로 분리
2. L1 여정 전이 행렬 + 1차 마르코프 가정 검정
3. 생존분석: 차단을 시간가변 공변량으로 넣은 Cox  ← 불멸시간 편향 방어
4. 개인별 자연 사용 주기
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from scipy import stats

from .schema import L1_STAGES


# ------------------------------------------------------- 1. shift-share

def shift_share(q: pd.DataFrame, t0: tuple[str, str], t1: tuple[str, str],
                target_stage: str = "EVALUATE") -> pd.DataFrame:
    """
    집계 비중 변화 ΔP 를 4개 항으로 분해한다.

      within    : 잔존자가 스스로 비중을 바꾼 몫      → 행동 변화 (수요 잔존, 회복 가능)
      between   : 잔존자 간 질의량 가중치 이동        → 구성 변화
      exit      : t0에만 있던 사용자가 빠진 몫        → 구성 변화 (이탈)
      entry     : t1에만 등장한 사용자가 더한 몫      → 구성 변화 (신규)

    within 이 지배적이면 데이터·응답 복구로 회복 가능.
    exit 이 지배적이면 이미 떠난 것 → 재획득 과제.
    """
    d = q.copy()
    d["is_tgt"] = d["l1_stage"].eq(target_stage).astype(float)

    def _agg(lo, hi):
        s = d[(d["ts"] >= pd.Timestamp(lo)) & (d["ts"] < pd.Timestamp(hi))]
        g = s.groupby("user_id").apply(
            lambda x: pd.Series({
                "vol": x["sample_weight"].sum(),
                "p": np.average(x["is_tgt"], weights=x["sample_weight"])}),
            include_groups=False)
        g["w"] = g["vol"] / g["vol"].sum()
        return g

    a, b = _agg(*t0), _agg(*t1)
    stay = a.index.intersection(b.index)
    exit_u = a.index.difference(b.index)
    entry_u = b.index.difference(a.index)

    w0, p0 = a.loc[stay, "w"], a.loc[stay, "p"]
    w1, p1 = b.loc[stay, "w"], b.loc[stay, "p"]

    within = float((w0 * (p1 - p0)).sum())
    between = float(((w1 - w0) * p0).sum())
    inter = float(((w1 - w0) * (p1 - p0)).sum())
    entry = float((b.loc[entry_u, "w"] * b.loc[entry_u, "p"]).sum())
    exit_ = float(-(a.loc[exit_u, "w"] * a.loc[exit_u, "p"]).sum())

    P0 = float((a["w"] * a["p"]).sum())
    P1 = float((b["w"] * b["p"]).sum())
    total = P1 - P0

    rows = [
        ("within (행동변화)", within), ("between (잔존자 가중치이동)", between),
        ("interaction", inter), ("exit (이탈)", exit_), ("entry (신규)", entry),
    ]
    out = pd.DataFrame(rows, columns=["항", "기여"])
    out["기여율"] = out["기여"] / total if total != 0 else np.nan
    out.loc[len(out)] = ["합계(= ΔP)", sum(v for _, v in rows), 1.0]
    out.attrs.update({"P0": P0, "P1": P1, "ΔP": total,
                      "n_stay": len(stay), "n_exit": len(exit_u), "n_entry": len(entry_u)})
    return out


def shift_share_verdict(ss: pd.DataFrame) -> str:
    w = float(ss.loc[ss["항"].str.startswith("within"), "기여"].iloc[0])
    comp = float(ss.loc[ss["항"].str.startswith(("between", "exit", "entry")), "기여"].sum())
    tot = w + comp
    if tot == 0:
        return "변화 없음"
    share = w / tot
    if share > 0.6:
        return (f"행동 변화 우세 ({share:.0%}) — 수요는 남아 있음. "
                "데이터·대체응답 복구로 상당 부분 회복 가능")
    if share < 0.4:
        return (f"구성 변화 우세 ({1-share:.0%}) — 이미 이탈. "
                "재획득 전략 + 진입 시점 기대치 조정 필요")
    return f"행동·구성 혼재 (행동 {share:.0%}) — 두 트랙 병행"


# ------------------------------------------------------- 2. 여정 전이

def transition_matrix(q: pd.DataFrame, level: str = "l1_stage",
                      within_session: bool = True) -> pd.DataFrame:
    d = q.sort_values(["user_id", "ts"])
    key = "session_id" if within_session else "user_id"
    prev = d.groupby(key)[level].shift(1)
    pairs = pd.DataFrame({"from": prev, "to": d[level]}).dropna()
    ct = pd.crosstab(pairs["from"], pairs["to"])
    if level == "l1_stage":
        idx = [s for s in L1_STAGES if s in ct.index]
        col = [s for s in L1_STAGES if s in ct.columns]
        ct = ct.reindex(index=idx, columns=col, fill_value=0)
    return ct.div(ct.sum(axis=1), axis=0).fillna(0.0)


def markov_order_test(q: pd.DataFrame, level: str = "l1_stage") -> dict:
    """
    1차 마르코프 가정 검정 (order-1 vs order-2 우도비).
    기각되면 전이확률 해석에 이력 의존성이 있다는 뜻 → 단순 전이행렬 해석 주의.
    """
    d = q.sort_values(["user_id", "ts"])
    g = d.groupby("session_id")[level]
    s1, s2, s3 = g.shift(2), g.shift(1), d[level]
    tri = pd.DataFrame({"a": s1, "b": s2, "c": s3}).dropna()
    if len(tri) < 200:
        return {"검정": "표본 부족", "p값": np.nan}

    ll2 = ll1 = 0.0
    for b, gb in tri.groupby("b"):
        p1 = gb["c"].value_counts(normalize=True)
        ll1 += float(np.log(gb["c"].map(p1)).sum())
        for a, ga in gb.groupby("a"):
            p2 = ga["c"].value_counts(normalize=True)
            ll2 += float(np.log(ga["c"].map(p2)).sum())
    k = tri["c"].nunique()
    df = (tri["a"].nunique() - 1) * tri["b"].nunique() * (k - 1)
    lr = 2 * (ll2 - ll1)
    p = float(stats.chi2.sf(lr, max(df, 1)))
    return {"LR": lr, "df": df, "p값": p,
            "판정": ("1차 마르코프 기각 — 이력 의존적. 전이행렬만으로 해석 금지"
                     if p < .05 else "1차 마르코프 지지 — 전이행렬 해석 가능")}


# ------------------------------------------------------- 3. 생존분석

def build_survival_panel(q: pd.DataFrame, end_date, grace_days: int = 30,
                         baseline_days: int = 14) -> pd.DataFrame:
    """
    Cox 시간가변 공변량용 long-format 패널.

    - 차단(blocked)은 발생 시점부터 1이 되는 시간가변 변수.
      고정 공변량으로 넣으면 불멸시간 편향이 생긴다.
    - baseline_days 초기 활동량·의도다양성으로 차단의 비무작위 배정을 통제.
    """
    end_date = pd.Timestamp(end_date)
    d = q.sort_values(["user_id", "ts"]).copy()
    d["date"] = d["ts"].dt.normalize()

    g = d.groupby("user_id")
    first = g["date"].min().rename("t0")
    last = g["date"].max().rename("tl")
    base = pd.concat([first, last], axis=1)

    blk = (d[d["outcome"].eq("blocked")].groupby("user_id")["date"].min()
             .rename("t_block"))
    base = base.join(blk)

    # 초기 활동량 / 의도 다양성 (교란 통제)
    d = d.join(first, on="user_id")
    early = d[d["date"] < d["t0"] + pd.Timedelta(days=baseline_days)]
    base = base.join(early.groupby("user_id").size().rename("early_n"))
    base = base.join(early.groupby("user_id")["l2_intent"].nunique().rename("early_div"))
    # 차단은 무작위 배정이 아니다: 투자 지향적인 사용자일수록 차단을 많이 겪고,
    # 그런 사용자는 애초에 이탈 성향이 다르다. 초기 의도 믹스로 그 성향을 통제한다.
    inv = early.assign(
        _i=early["l1_stage"].isin(["DISCOVER", "EVALUATE"]).astype(float))
    base = base.join(inv.groupby("user_id")["_i"].mean().rename("early_invest_share"))
    base[["early_n", "early_div", "early_invest_share"]] = \
        base[["early_n", "early_div", "early_invest_share"]].fillna(0)

    base["dur"] = (base["tl"] - base["t0"]).dt.days.clip(lower=1)
    base["event"] = ((end_date - base["tl"]).dt.days > grace_days).astype(int)

    rows = []
    for uid, r in base.iterrows():
        tb = (r["t_block"] - r["t0"]).days if pd.notna(r["t_block"]) else None
        common = {"user_id": uid, "early_n": np.log1p(r["early_n"]),
                  "early_div": r["early_div"],
                  "early_invest_share": r["early_invest_share"],
                  "_inv_raw": r["early_invest_share"], "_n_raw": r["early_n"]}
        if tb is None or tb <= 0 or tb >= r["dur"]:
            # tb >= dur : 마지막 활동일에 차단된 경우. 이전에는 '미차단'으로
            # 코딩되어 단일세션 사용자가 다수인 표본에서 대량 오분류가 났다.
            rows.append({**common, "start": 0, "stop": r["dur"],
                         "event": r["event"],
                         "blocked": int(tb is not None and (tb <= 0 or tb >= r["dur"]))})
        else:
            rows.append({**common, "start": 0, "stop": tb, "event": 0, "blocked": 0})
            rows.append({**common, "start": tb, "stop": r["dur"],
                         "event": r["event"], "blocked": 1})
    p = pd.DataFrame(rows)
    # 층화용 구간: 선형 공변량은 비선형 교란을 못 잡는다.
    p["stratum"] = (
        pd.qcut(p["_inv_raw"], 4, labels=False, duplicates="drop").astype(str)
        + "_" + pd.qcut(p["_n_raw"], 3, labels=False, duplicates="drop").astype(str))
    return p[p["stop"] > p["start"]].drop(columns=["_inv_raw", "_n_raw"]).reset_index(drop=True)


def cox_block_effect(panel: pd.DataFrame, adjusted: bool = True) -> dict:
    """
    adjusted=False 로 한 번 더 돌려 조정 전후를 비교하십시오.
    두 값의 차이가 크면 차단의 비무작위 배정이 실제로 작동하고 있다는 뜻입니다.
    """
    if not adjusted:
        panel = panel[["user_id", "start", "stop", "event", "blocked"]]
        strata = None
    else:
        strata = ["stratum"] if "stratum" in panel.columns else None
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(panel, id_col="user_id", event_col="event", strata=strata,
            start_col="start", stop_col="stop", show_progress=False)
    s = ctv.summary.loc["blocked"]
    return {"log(HR)": float(s["coef"]), "HR": float(np.exp(s["coef"])),
            "95%CI(HR)": (float(np.exp(s["coef lower 95%"])),
                          float(np.exp(s["coef upper 95%"]))),
            "p값": float(s["p"]), "_fitter": ctv}


def km_curve(q: pd.DataFrame, end_date, grace_days: int = 30) -> pd.DataFrame:
    end_date = pd.Timestamp(end_date)
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    g = d.groupby("user_id")["date"]
    dur = (g.max() - g.min()).dt.days.clip(lower=1)
    ev = ((end_date - g.max()).dt.days > grace_days).astype(int)
    kmf = KaplanMeierFitter().fit(dur, ev)
    sf = kmf.survival_function_.rename(columns={"KM_estimate": "생존확률"})
    return sf.loc[[i for i in [1, 7, 14, 30, 60, 90] if i in sf.index]]


# ------------------------------------------------------- 4. 사용 주기

def usage_cycle(q: pd.DataFrame) -> pd.DataFrame:
    """개인 내 방문 간격 분포. 리텐션 창(D7 vs W4)을 정하는 근거."""
    d = q[["user_id", "ts"]].copy()
    d["date"] = d["ts"].dt.normalize()
    d = d.drop_duplicates(["user_id", "date"]).sort_values(["user_id", "date"])
    d["gap"] = d.groupby("user_id")["date"].diff().dt.days
    gaps = d["gap"].dropna()
    med = d.groupby("user_id")["gap"].median().dropna()
    return pd.DataFrame({
        "지표": ["전체 간격 중앙값", "전체 간격 p75", "전체 간격 p90",
                 "개인별 중앙값의 중앙값", "개인별 중앙값의 p75"],
        "일": [gaps.median(), gaps.quantile(.75), gaps.quantile(.90),
               med.median(), med.quantile(.75)],
    })


def retention_window_advice(uc: pd.DataFrame) -> str:
    p75 = float(uc.loc[uc["지표"].eq("개인별 중앙값의 p75"), "일"].iloc[0])
    if p75 <= 3:
        return "일 단위 사용 — D1/D7/D30 적절"
    if p75 <= 10:
        return "주 단위 사용 — D1은 무의미. W1/W4 권장"
    return "주 단위 이상 — W2/W4/W8 또는 unbounded retention 권장"
