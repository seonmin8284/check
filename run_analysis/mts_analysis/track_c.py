"""
C트랙 — 거래 데이터 조인 분석 (로그인 ID 필수).

1. 정보→주문 증분 효과: 개인 × 종목 × 일 패널 + 3원 고정효과
   대조군을 '비챗봇 경로로 조회한 종목'으로 잡아 관심 수준을 통제한다.
   단순 '챗봇 사용자 vs 미사용자' 비교는 효과를 몇 배 부풀린다.
2. 챗봇 이탈 ≠ 고객 이탈 2×2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import EVALUATE_LOOKUP_INTENTS


# ------------------------------------------------- 고정효과 흡수 (교차 투영)

def absorb(df: pd.DataFrame, cols: list[str], fe_cols: list[str],
           tol: float = 1e-9, maxiter: int = 500) -> tuple[np.ndarray, int]:
    """다원 고정효과를 반복 평균차감으로 흡수. 반환: (잔차행렬, 흡수된 파라미터 수)"""
    X = df[cols].to_numpy(dtype=float).copy()
    codes, ns = [], []
    for f in fe_cols:
        c, u = pd.factorize(df[f])
        codes.append(c.astype(np.int64))
        ns.append(len(u))
    for _ in range(maxiter):
        delta = 0.0
        for c, n in zip(codes, ns):
            cnt = np.bincount(c, minlength=n).astype(float)
            for j in range(X.shape[1]):
                m = np.bincount(c, weights=X[:, j], minlength=n) / np.maximum(cnt, 1)
                adj = m[c]
                X[:, j] -= adj
                delta = max(delta, float(np.abs(adj).max()))
        if delta < tol:
            break
    return X, int(sum(ns) - (len(ns) - 1))


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray,
                absorbed: int = 0) -> dict:
    """클러스터-로버스트 표준오차 OLS (고정효과 흡수 후)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    gs, gcodes = np.unique(groups, return_inverse=True)
    G, n, k = len(gs), len(y), X.shape[1]
    meat = np.zeros((k, k))
    Xu = X * u[:, None]
    for g in range(G):
        s = Xu[gcodes == g].sum(axis=0)
        meat += np.outer(s, s)
    dof = max(n - k - absorbed, 1)
    c = (G / max(G - 1, 1)) * ((n - 1) / dof)
    V = XtX_inv @ meat @ XtX_inv * c
    se = np.sqrt(np.diag(V))
    from scipy import stats
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=max(G - 1, 1))
    return {"beta": beta, "se": se, "t": t, "p": p, "G": G, "n": n}


# ------------------------------------------------- 1. 증분 주문 효과

def build_view_order_panel(q: pd.DataFrame, orders: pd.DataFrame,
                           app_views: pd.DataFrame, horizon_days: int = 5,
                           require_success: bool = True) -> pd.DataFrame:
    """
    위험집합 = (사용자, 종목, 일) 중 그날 '어떤 경로로든' 그 종목을 조회한 건.
      treat = 1 : 챗봇으로 조회
      treat = 0 : 비챗봇 경로(종목상세·검색)로만 조회
      y = horizon_days 내 해당 종목 주문 발생 여부
    """
    cb = q[q["l2_intent"].isin(EVALUATE_LOOKUP_INTENTS)].copy()
    cb = cb[cb["slot_target"].map(lambda s: isinstance(s, (list, tuple)) and len(s) > 0)]
    if require_success:
        cb = cb[cb["outcome"].eq("success")]
    cb = cb.explode("slot_target").rename(columns={"slot_target": "ticker"})
    cb["date"] = cb["ts"].dt.normalize()
    cb = cb[["user_id", "ticker", "date"]].drop_duplicates()
    cb["chatbot"] = 1

    nv = app_views.copy()
    nv["date"] = nv["ts"].dt.normalize()
    nv = nv[["user_id", "ticker", "date"]].drop_duplicates()
    nv["chatbot"] = 0

    panel = (pd.concat([cb, nv], ignore_index=True)
               .groupby(["user_id", "ticker", "date"], as_index=False)["chatbot"].max())

    o = orders.copy()
    o["odate"] = o["ts"].dt.normalize()
    o = o[["user_id", "ticker", "odate"]].drop_duplicates()

    m = panel.merge(o, on=["user_id", "ticker"], how="left")
    gap = (m["odate"] - m["date"]).dt.days
    m["hit"] = ((gap >= 0) & (gap < horizon_days)).fillna(False).astype(int)
    out = (m.groupby(["user_id", "ticker", "date", "chatbot"], as_index=False)["hit"]
             .max().rename(columns={"hit": "ordered"}))
    return out


def incremental_order_effect(panel: pd.DataFrame) -> dict:
    """개인 + 종목 + 일자 3원 고정효과, 개인 클러스터 SE."""
    d = panel.dropna(subset=["ordered", "chatbot"]).copy()
    if len(d) < 50 or d["chatbot"].nunique() < 2:
        return {"불가": (f"증분효과 식별 불가 — 관측 {len(d)}건, "
                         f"처치 구분 {d['chatbot'].nunique()}종. "
                         "orders / app_views 데이터가 필요합니다 "
                         "(app_views 가 대조군입니다).")}
    Z, absorbed = absorb(d, ["ordered", "chatbot"], ["user_id", "ticker", "date"])
    y, X = Z[:, 0], Z[:, [1]]
    r = ols_cluster(y, X, d["user_id"].to_numpy(), absorbed=absorbed)
    naive = (d.groupby("chatbot")["ordered"].mean()
               .rename("주문확률").to_frame())
    lift_naive = float(naive.loc[1, "주문확률"] - naive.loc[0, "주문확률"])
    b, se = float(r["beta"][0]), float(r["se"][0])
    return {
        "증분효과(pp)": b, "표준오차": se, "p값": float(r["p"][0]),
        "95%CI": (b - 1.96 * se, b + 1.96 * se),
        "단순차이(pp)": lift_naive,
        "편향(단순-FE)": lift_naive - b,
        "관측수": int(r["n"]), "클러스터(사용자)": int(r["G"]),
        "_naive": naive,
    }


# ------------------------------------------------- 2. 챗봇 이탈 ≠ 고객 이탈

def churn_2x2(q: pd.DataFrame, app_sessions: pd.DataFrame, end_date,
              window_days: int = 30, baseline_days: int = 60) -> pd.DataFrame:
    """
    베이스라인 기간에 챗봇을 쓴 사용자만 대상으로,
    최근 window_days 동안의 챗봇 활동 × 앱 활동 2×2.

    '챗봇 이탈 / 앱 유지'가 크면 챗봇 고유 문제.
    양쪽 다 이탈이 크면 공통 원인 가능성 → 챗봇을 원인으로 단정하면 안 됨.
    """
    end_date = pd.Timestamp(end_date)
    w_start = end_date - pd.Timedelta(days=window_days)
    b_start = w_start - pd.Timedelta(days=baseline_days)

    qd = q.copy()
    qd["date"] = qd["ts"].dt.normalize()
    cohort = set(qd[(qd["date"] >= b_start) & (qd["date"] < w_start)]["user_id"])
    if not cohort:
        return pd.DataFrame()

    cb_recent = set(qd[qd["date"] >= w_start]["user_id"])
    ap = app_sessions.copy()
    ap["date"] = pd.to_datetime(ap["date"]).dt.normalize()
    app_recent = set(ap[ap["date"] >= w_start]["user_id"])

    rec = [{"user_id": u,
            "챗봇": "유지" if u in cb_recent else "이탈",
            "앱": "유지" if u in app_recent else "이탈"} for u in cohort]
    df = pd.DataFrame(rec)
    tab = pd.crosstab(df["챗봇"], df["앱"])
    tab = tab.reindex(index=["유지", "이탈"], columns=["유지", "이탈"], fill_value=0)
    tab["합계"] = tab.sum(axis=1)
    tab.loc["합계"] = tab.sum()
    return tab


def churn_2x2_verdict(tab: pd.DataFrame) -> str:
    if tab.empty:
        return "대상 코호트 없음"
    n = tab.loc["합계", "합계"]
    a = tab.loc["이탈", "유지"]      # 챗봇만 이탈
    b = tab.loc["이탈", "이탈"]      # 둘 다 이탈
    c = tab.loc["유지", "이탈"]      # 챗봇만 유지
    msg = [f"챗봇만 이탈 {a/n:.1%} · 동반 이탈 {b/n:.1%} · 챗봇만 유지 {c/n:.1%}"]
    if a > b:
        msg.append("→ 챗봇 고유 문제가 우세. 고객은 남아 있으므로 회복 여지 큼")
    elif b > a * 1.5:
        msg.append("→ 동반 이탈 우세. 공통 원인 가능성. 챗봇을 원인으로 단정 금지")
    else:
        msg.append("→ 혼재. 두 원인 분리 설계 필요")
    return " ".join(msg)


def horizon_sensitivity(q: pd.DataFrame, orders: pd.DataFrame,
                        app_views: pd.DataFrame,
                        horizons: tuple[int, ...] = (1, 2, 3, 5, 7, 10)) -> pd.DataFrame:
    """
    전환 창(horizon) 민감도.

    짧으면 실제 전환을 놓쳐 과소추정되고, 길면 무관한 주문이 섞여 희석된다.
    추정치가 평평해지는 지점을 본 창으로 쓰고, 그 값을 보고하십시오.
    반복 조회가 많은 종목일수록 처치·대조 셀이 결과를 공유하므로
    이 추정치는 보수적(하한) 성격입니다.
    """
    rows = []
    for h in horizons:
        vp = build_view_order_panel(q, orders, app_views, horizon_days=h)
        r = incremental_order_effect(vp)
        if "불가" in r:
            return pd.DataFrame({"안내": [r["불가"]]})
        rows.append({"창(일)": h, "FE증분": r["증분효과(pp)"],
                     "CI_lo": r["95%CI"][0], "CI_hi": r["95%CI"][1],
                     "단순차이": r["단순차이(pp)"], "관측수": r["관측수"]})
    out = pd.DataFrame(rows)
    out["직전대비변화"] = out["FE증분"].diff()
    return out
