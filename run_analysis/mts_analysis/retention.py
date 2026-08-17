"""
코호트 리텐션 커브 · 평탄화 지점 · Activation 역산.

설계 원칙 세 가지를 코드로 옮긴 것:
  1) 리텐션은 코호트 단위로만 본다 (전체 평균은 신규 유입량에 따라 착시)
  2) 절대값보다 커브가 평평해지는 지점이 PMF 판단의 실질적 근거
  3) Activation 은 추측이 아니라 리텐션이 갈리는 행동에서 역산한다
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------- 1. 코호트 리텐션

def cohort_retention(q: pd.DataFrame, end_date, freq: str = "W",
                     horizons: tuple[int, ...] = (1, 7, 14, 30, 45, 60, 90),
                     min_users: int = 20) -> pd.DataFrame:
    """
    가입(첫 활동) 주차별 코호트의 시점별 잔존율.

    ★ 관측 창이 짧아 아직 도달하지 못한 시점은 NaN 으로 둔다.
      이걸 0 으로 채우면 최근 코호트가 급락한 것처럼 보인다(우측 절단 착시).
    """
    end_date = pd.Timestamp(end_date)
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    first = d.groupby("user_id")["date"].min().rename("t0")
    d = d.join(first, on="user_id")
    d["day"] = (d["date"] - d["t0"]).dt.days
    d["cohort"] = d["t0"].dt.to_period(freq).dt.start_time

    rows = []
    for coh, g in d.groupby("cohort"):
        users = g["user_id"].nunique()
        if users < min_users:
            continue
        obs_days = (end_date - coh).days
        rec = {"코호트": coh, "사용자": users}
        for hz in horizons:
            if obs_days < hz:
                rec[f"D{hz}"] = np.nan          # 아직 관측 불가
            else:
                ret = g.loc[g["day"] >= hz, "user_id"].nunique()
                rec[f"D{hz}"] = ret / users
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("코호트")
    return out.round(3)


def plateau_point(q: pd.DataFrame, end_date, max_day: int = 90,
                  flat_eps: float = 0.005, run: int = 5) -> dict:
    """
    리텐션 커브가 평평해지는 지점을 찾는다.

    커브가 0 으로 수렴하면 PMF 없음, 어느 수준에서 평평해지면
    그 층이 실제 가치를 느끼는 사용자다. D7 값보다 이쪽이 중요하다.
    """
    end_date = pd.Timestamp(end_date)
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    first = d.groupby("user_id")["date"].min().rename("t0")
    d = d.join(first, on="user_id")
    d["day"] = (d["date"] - d["t0"]).dt.days

    # 관측 창이 max_day 이상인 사용자만 (우측 절단 방어)
    elig = first[(end_date - first).dt.days >= max_day].index
    if len(elig) < 50:
        # 창이 짧으면 가능한 범위까지만
        max_day = int((end_date - first).dt.days.quantile(.75))
        elig = first[(end_date - first).dt.days >= max_day].index
        if len(elig) < 30 or max_day < 7:
            return {"안내": "관측 창이 짧아 평탄화 판정 불가"}

    e = d[d["user_id"].isin(elig)]
    n = len(elig)
    curve = pd.Series(
        {k: e.loc[e["day"] >= k, "user_id"].nunique() / n
         for k in range(0, max_day + 1)})

    diffs = curve.diff().abs()
    plateau_day, level = None, None
    for k in range(1, max_day - run + 1):
        if not (diffs.iloc[k:k + run] < flat_eps).all():
            continue
        # ★ 국소적으로 완만할 뿐 이후 계속 떨어지면 평탄화가 아니다.
        #   판정 지점 이후 잔여 구간의 총 하락폭이 충분히 작아야 한다.
        tail_drop = float(curve.iloc[k] - curve.iloc[-1])
        rel_drop = tail_drop / max(float(curve.iloc[k]), 1e-9)
        if rel_drop > 0.20:
            continue
        plateau_day, level = k, float(curve.iloc[k])
        break

    total_drop = float(curve.iloc[1] - curve.iloc[-1])
    verdict = (f"평탄화 미도달 — D1 {curve.iloc[1]:.1%} → D{max_day} "
               f"{curve.iloc[-1]:.1%} 로 계속 감소 중(총 -{total_drop:.1%}p). "
               "재방문 가치가 형성된 층이 아직 확인되지 않습니다. "
               "리텐션 목표 대신 세션 내 해결률을 추적 지표로 쓰십시오.")
    if plateau_day is not None:
        if level < 0.05:
            verdict = (f"D{plateau_day} 에서 {level:.1%} 로 평탄화 — 사실상 0 수렴. "
                       "재방문 가치가 형성되지 않았습니다")
        else:
            verdict = (f"D{plateau_day} 에서 {level:.1%} 로 평탄화 — "
                       f"이 층이 실제 가치를 느끼는 사용자입니다. "
                       "개선 목표는 '커브를 올리기'가 아니라 '이 층을 두껍게 하기'")
    return {"대상사용자": int(n), "관측일수": int(max_day),
            "평탄화일": plateau_day, "평탄화수준": level,
            "커브": curve.round(4), "판정": verdict}


# ------------------------------------------------------- 2. Activation 역산

def activation_candidates(q: pd.DataFrame, end_date, window_days: int = 7,
                          retention_day: int = 30,
                          min_group: float = 0.10) -> pd.DataFrame:
    """
    Activation 정의를 데이터에서 역산한다.

    첫 window_days 동안의 여러 행동 후보를 놓고, 각 임계값에서
    D{retention_day} 잔존율이 가장 크게 갈리는 지점을 찾는다.

    ★ 이건 상관이지 인과가 아니다. "그 행동을 시키면 남는다"가 아니라
      "그 행동을 한 사람이 남았다"이다. 확정 전 A/B 로 검증하십시오.
    """
    end_date = pd.Timestamp(end_date)
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    first = d.groupby("user_id")["date"].min().rename("t0")
    d = d.join(first, on="user_id")
    d["day"] = (d["date"] - d["t0"]).dt.days

    # 관측 창 확보된 사용자만
    elig = first[(end_date - first).dt.days >= retention_day].index
    if len(elig) < 100:
        return pd.DataFrame({"안내": ["관측 창이 확보된 사용자가 부족합니다"]})
    d = d[d["user_id"].isin(elig)]

    ret = (d[d["day"] >= retention_day].groupby("user_id").size() > 0)
    ret = ret.reindex(elig, fill_value=False).rename("retained")

    w = d[d["day"] < window_days]
    feat = pd.DataFrame(index=elig)
    feat["세션수"] = w.groupby("user_id")["session_id"].nunique()
    feat["질의수"] = w.groupby("user_id").size()
    feat["활동일수"] = w.groupby("user_id")["date"].nunique()
    feat["성공질의수"] = w[w["outcome"].eq("success")].groupby("user_id").size()
    feat["고유의도수"] = w.groupby("user_id")["l2_intent"].nunique()
    feat["고유단계수"] = w.groupby("user_id")["l1_stage"].nunique()
    feat = feat.fillna(0)

    rows = []
    base = float(ret.mean())
    for col in feat.columns:
        for thr in range(1, 8):
            hit = feat[col] >= thr
            share = float(hit.mean())
            if not (min_group <= share <= 1 - min_group):
                continue
            a, b = float(ret[hit].mean()), float(ret[~hit].mean())
            rows.append({"후보": f"{col} ≥ {thr}", "충족비율": share,
                         "충족시_잔존": a, "미충족시_잔존": b,
                         "격차": a - b, "리프트": a / b if b > 0 else np.nan})

    # 단계·의도 경험 여부 (이분형)
    for stage, g in w.groupby("l1_stage"):
        hit = pd.Series(elig.isin(g["user_id"].unique()), index=elig)
        share = float(hit.mean())
        if not (min_group <= share <= 1 - min_group):
            continue
        a, b = float(ret[hit].mean()), float(ret[~hit].mean())
        rows.append({"후보": f"{stage} 경험", "충족비율": share,
                     "충족시_잔존": a, "미충족시_잔존": b,
                     "격차": a - b, "리프트": a / b if b > 0 else np.nan})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"안내": ["유효한 후보가 없습니다"]})
    out.attrs["전체잔존율"] = base
    return out.sort_values("격차", ascending=False).reset_index(drop=True).round(3)


def activation_verdict(cand: pd.DataFrame) -> str:
    if "안내" in cand.columns or cand.empty:
        return "판정 불가"
    top = cand.iloc[0]
    return (f"Activation 후보 1순위: 「{top['후보']}」 — "
            f"충족 {top['충족비율']:.0%} · 잔존 {top['충족시_잔존']:.1%} vs "
            f"{top['미충족시_잔존']:.1%} (격차 {top['격차']*100:+.1f}%p, "
            f"리프트 {top['리프트']:.2f}배). "
            "상관이므로 확정 전 A/B 검증 필요.")
