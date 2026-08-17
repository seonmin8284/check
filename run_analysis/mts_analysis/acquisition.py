"""
획득 분석 — "왜 떠났나"에서 "왜 들어오지 않나"로.

세그먼트 시계열이 투자형 → 업무형 전환을 보여주는데, 잔존자가 극소수라
'같은 사람이 바뀐 것'이 아니다. 즉 유입 구성 자체가 바뀐 것이고,
그렇다면 문제는 UX 가 아니라 획득이다.

동시에 진입 질문별 잔존율은 정반대를 가리킨다 —
잔존시키는 진입과 실제 유입이 어긋나 있다면, 그 격차가 곧 기회다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _first_queries(q: pd.DataFrame) -> pd.DataFrame:
    """사용자별 첫 질의(= 진입 질문)."""
    d = q.sort_values("ts")
    f = d.groupby("user_id").first()
    f["첫활동일"] = d.groupby("user_id")["ts"].min().dt.normalize()
    return f


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


# ------------------------------------------------- 1. 유입 구성 추이

def entry_mix_over_time(q: pd.DataFrame, freq: str = "M",
                        level: str = "l1_stage") -> pd.DataFrame:
    """
    **신규 사용자의 첫 질의** 구성 추이.

    전체 질의 구성은 잔존자와 신규가 섞여 원인을 못 가린다.
    첫 질의만 보면 '어떤 사람이 새로 들어오고 있는가'가 그대로 나온다.
    """
    f = _first_queries(q)
    f["period"] = f["첫활동일"].dt.to_period(freq).dt.start_time
    ct = pd.crosstab(f["period"], f[level], normalize="index")
    ct.insert(0, "신규사용자", f.groupby("period").size())
    return ct.round(3)


def entry_mix_shift(q: pd.DataFrame, level: str = "l1_stage",
                    head_months: int = 3, tail_months: int = 3) -> pd.DataFrame:
    """초기 구간과 최근 구간의 유입 구성 차이. 무엇이 늘고 줄었는지."""
    mix = entry_mix_over_time(q, "M", level).drop(columns=["신규사용자"])
    if len(mix) < head_months + tail_months:
        return pd.DataFrame({"안내": ["구간이 짧아 비교 불가"]})
    a = mix.head(head_months).mean()
    b = mix.tail(tail_months).mean()
    out = pd.DataFrame({"초기": a, "최근": b})
    out["변화"] = out["최근"] - out["초기"]
    out["배율"] = (out["최근"] / out["초기"]).replace([np.inf, -np.inf], np.nan)
    return out.sort_values("변화").round(3)


def new_vs_returning(q: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    각 시기에 활동한 사용자 중 신규와 재방문의 비율.

    '매달 새 사람이 채우는 서비스인가, 남은 사람이 쓰는 서비스인가'가
    이후 모든 논의의 전제가 된다. 신규 비중이 높으면 이탈 논의가
    '왜 안 남는가'로, 낮으면 '왜 안 들어오는가'로 향한다.
    """
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    first = d.groupby("user_id")["date"].min().rename("t0")
    d = d.join(first, on="user_id")
    d["period"] = d["date"].dt.to_period(freq).dt.start_time
    d["_first_p"] = d["t0"].dt.to_period(freq).dt.start_time
    act = d.drop_duplicates(["period", "user_id"]).copy()
    act["구분"] = np.where(act["period"].eq(act["_first_p"]), "신규", "재방문")
    ct = pd.crosstab(act["period"], act["구분"])
    for c in ("신규", "재방문"):
        if c not in ct.columns:
            ct[c] = 0
    ct["활동 사용자"] = ct["신규"] + ct["재방문"]
    ct["신규 비중"] = ct["신규"] / ct["활동 사용자"]
    ct["재방문 비중"] = ct["재방문"] / ct["활동 사용자"]
    return ct[["활동 사용자", "신규", "재방문", "신규 비중",
               "재방문 비중"]].round(3)


def composition_verdict(nr: pd.DataFrame) -> str:
    if nr.empty:
        return "판정 불가"
    recent = float(nr["신규 비중"].tail(3).mean())
    if recent >= .6:
        return (f"최근 활동 사용자의 {recent:.0%}가 그 달 처음 온 사람입니다. "
                "매달 새 사용자가 서비스를 채우고 있고, 재방문으로 유지되는 층은 "
                "얇습니다. 이탈 논의는 '왜 안 남는가'에 초점을 맞춰야 합니다.")
    if recent <= .35:
        return (f"신규 비중이 {recent:.0%}로 낮습니다. 기존 사용자가 서비스를 "
                "지탱하고 있으며, 문제는 '왜 새로 들어오지 않는가'입니다.")
    return (f"신규 {recent:.0%} · 재방문 {1-recent:.0%}로 양쪽이 함께 "
            "서비스를 구성합니다.")


# ------------------------------------------------- 2. 진입 질문 × 잔존 (신뢰구간)

def entry_retention(q: pd.DataFrame, end_date, retention_day: int = 30,
                    min_n: int = 50, level: str = "l2_intent") -> pd.DataFrame:
    """
    진입 질문별 잔존율 + Wilson 신뢰구간 + 전체 대비 유의성.

    ★ 표본이 작은 의도의 차이를 근거로 삼지 않도록 CI 와 p값을 함께 낸다.
      (n=53 짜리 의도의 +12%p 는 우연일 수 있다)
    """
    end_date = pd.Timestamp(end_date)
    d = q.sort_values("ts")
    f = _first_queries(d)
    last = d.groupby("user_id")["ts"].max()

    ok = f.index[(end_date - f["첫활동일"]).dt.days >= retention_day]
    f = f.loc[ok]
    ret = ((last.loc[ok] - f["첫활동일"]).dt.days >= retention_day).astype(int)
    base = float(ret.mean())

    rows = []
    for lv, idx in f.groupby(level).groups.items():
        n = len(idx)
        if n < min_n:
            continue
        k = int(ret.loc[idx].sum())
        lo, hi = wilson(k, n)
        se = np.sqrt(base * (1 - base) / n)
        z = (k / n - base) / se if se > 0 else 0.0
        rows.append({level: lv, "n": n, "잔존율": k / n,
                     "CI_lo": lo, "CI_hi": hi,
                     "전체대비": k / n - base,
                     "p값": float(2 * stats.norm.sf(abs(z))),
                     "유의": "★" if 2 * stats.norm.sf(abs(z)) < .05 else ""})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"안내": [f"n≥{min_n} 인 진입 의도가 없습니다"]})
    out.attrs["전체잔존율"] = base
    out.attrs["대상사용자"] = int(len(f))
    return out.sort_values("잔존율", ascending=False).reset_index(drop=True).round(4)


# ------------------------------------------------- 3. 믹스 반사실

def mix_counterfactual(q: pd.DataFrame, end_date, retention_day: int = 30,
                       level: str = "l1_stage", freq: str = "M") -> pd.DataFrame:
    """
    **유입 구성이 초기와 같았다면 전체 잔존율은 얼마였을까.**

    직접표준화: 각 시기의 진입유형별 잔존율은 그대로 두고,
    유입 구성비만 기준 시기 값으로 바꿔 재가중한다.

    관측 잔존율과 반사실 잔존율의 격차 = **유입 구성 변화가 만든 잔존 손실**.
    이 값이 크면 개선 대상은 응답 품질이 아니라 획득 채널이다.
    """
    end_date = pd.Timestamp(end_date)
    d = q.sort_values("ts")
    f = _first_queries(d)
    last = d.groupby("user_id")["ts"].max()
    ok = f.index[(end_date - f["첫활동일"]).dt.days >= retention_day]
    f = f.loc[ok].copy()
    f["ret"] = ((last.loc[ok] - f["첫활동일"]).dt.days >= retention_day).astype(int)
    f["period"] = f["첫활동일"].dt.to_period(freq).dt.start_time
    if f["period"].nunique() < 3:
        return pd.DataFrame({"안내": ["시기 구간이 부족합니다"]})

    # 유형별 잔존율은 전체 평균으로 고정 (시기별 표본이 작아 불안정)
    rate_by_lv = f.groupby(level)["ret"].mean()
    base_period = f["period"].min()
    base_mix = (f[f["period"].eq(base_period)][level]
                .value_counts(normalize=True))

    rows = []
    for per, g in f.groupby("period"):
        mix = g[level].value_counts(normalize=True)
        obs = float(g["ret"].mean())
        common = rate_by_lv.index.intersection(base_mix.index)
        cf = float((base_mix.loc[common] * rate_by_lv.loc[common]).sum()
                   / base_mix.loc[common].sum())
        cur = float((mix.reindex(rate_by_lv.index).fillna(0)
                     * rate_by_lv).sum())
        rows.append({"period": per, "신규": len(g), "관측잔존": obs,
                     "믹스기여_현재": cur, "믹스기여_초기고정": cf,
                     "믹스손실": cur - cf})
    out = pd.DataFrame(rows).set_index("period")
    out.attrs["기준시기"] = str(base_period.date())
    return out.round(4)


# ------------------------------------------------- 4. 1회성 사용자

def oneshot_profile(q: pd.DataFrame, level: str = "l2_intent",
                    min_n: int = 50) -> pd.DataFrame:
    """
    1회성(단일 세션) 사용자와 재방문 사용자의 진입 질문 차이.

    1회성이 다수인 제품에서는 '이탈 방지'보다
    '어떤 진입이 두 번째를 만드는가'가 실질 질문이다.
    """
    sess = q.groupby("user_id")["session_id"].nunique()
    f = _first_queries(q)
    f["재방문"] = (sess.reindex(f.index) > 1).astype(int)
    base = float(f["재방문"].mean())
    g = f.groupby(level)["재방문"].agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "재방문율"})
    g = g[g["n"] >= min_n]
    if g.empty:
        return pd.DataFrame({"안내": ["표본 부족"]})
    g["전체대비"] = g["재방문율"] - base
    ci = [wilson(int(r["재방문율"] * r["n"]), int(r["n"])) for _, r in g.iterrows()]
    g["CI_lo"] = [c[0] for c in ci]
    g["CI_hi"] = [c[1] for c in ci]
    g.attrs["전체재방문율"] = base
    return g.sort_values("재방문율", ascending=False).round(4)


# ------------------------------------------------- 5. 응답 지연의 영향

def latency_impact(q: pd.DataFrame, fu: pd.DataFrame | None = None,
                   bins=(0, 3000, 6000, 10000, 15000, 20000, 1e9)) -> pd.DataFrame:
    """
    응답시간 구간별 세션 종료·재질문·실패율.

    p95 가 15~25초라면 지연은 가드레일이 아니라 1차 UX 문제다.
    ★ 상관이지 인과가 아니다 — 어려운 질의가 느리고 동시에 실패도 잦다.
      그래서 성공 건만으로도 따로 본다(품질 교란 제거).
    """
    d = q.copy()
    if fu is not None and "turn_kind" in fu.columns:
        d = d.merge(fu[["query_id", "turn_kind"]], on="query_id", how="left")
    d = d.sort_values(["session_id", "ts"])
    d["_last"] = d.groupby("session_id")["ts"].transform("max").eq(d["ts"])
    d["_fail"] = ~d["outcome"].eq("success").fillna(False)
    labels = [f"{int(bins[i]/1000)}~{int(bins[i+1]/1000) if bins[i+1] < 1e8 else '∞'}s"
              for i in range(len(bins) - 1)]
    d["지연구간"] = pd.cut(d["latency_ms"], bins=bins, labels=labels, right=False)

    agg = {"n": ("_fail", "size"), "실패율": ("_fail", "mean"),
           "세션종료율": ("_last", "mean")}
    if "turn_kind" in d.columns:
        d["_rep"] = d["turn_kind"].isin(["REPEAT", "FORMAT"]).fillna(False)
        agg["복구성후속률"] = ("_rep", "mean")
    out = d.groupby("지연구간", observed=True).agg(**agg)

    ok = d[~d["_fail"]]
    out["성공건_세션종료율"] = ok.groupby("지연구간", observed=True)["_last"].mean()
    return out.round(4)


# ------------------------------------------------- 6. 진입 배치 시뮬레이션

def entry_reallocation(q: pd.DataFrame, end_date, retention_day: int = 30,
                       shift: float = 0.10, level: str = "l2_intent",
                       min_n: int = 50) -> dict:
    """
    진입 추천질문을 고잔존 의도 쪽으로 shift 만큼 옮겼을 때
    전체 잔존율이 얼마나 오르는가 (직접표준화 기반 상한 근사).

    ★ 강한 가정: 진입 의도를 바꿔도 그 의도의 잔존율은 유지된다.
      실제로는 '원래 그 질문을 하려던 사람'과 '유도되어 하는 사람'이 다르다.
      반드시 A/B 로 검증하십시오. 여기서는 실험 가치 판단용 숫자만 낸다.
    """
    er = entry_retention(q, end_date, retention_day, min_n, level)
    if "안내" in er.columns:
        return {"안내": er["안내"].iloc[0]}
    base = er.attrs["전체잔존율"]
    n_tot = er["n"].sum()
    w = er.set_index(level)["n"] / n_tot
    r = er.set_index(level)["잔존율"]

    top = r.idxmax()
    lows = r.nsmallest(max(len(r) // 3, 1)).index
    move = float(w.loc[lows].sum() * shift)
    w2 = w.copy()
    w2.loc[lows] = w2.loc[lows] * (1 - shift)
    w2.loc[top] = w2.loc[top] + move
    new = float((w2 * r).sum() / w2.sum())
    return {"현재 잔존율": round(base, 4),
            "재배치 후(추정)": round(new, 4),
            "증분": round(new - base, 4),
            "이동 비중": round(move, 4),
            "이동 대상": f"{list(lows)[:4]} → {top}",
            "주의": ("유도된 사용자가 자발적 사용자와 같은 잔존율을 보인다는 "
                     "가정 위의 상한입니다. A/B 로 검증하십시오.")}