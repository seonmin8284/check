"""
사용자 세그먼트 · 진입 질문 · 이탈 지점 · 리텐션 지점.

세그먼트는 미리 정의하지 않고 의도 벡터에서 도출한다.
"투자형/업무형"이 실제로 2개인지, 경계가 어디인지가 예상과 다를 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------ 1. 세그먼트

def intent_vectors(q: pd.DataFrame, level: str = "l1_stage",
                   min_queries: int = 5, period: tuple | None = None) -> pd.DataFrame:
    d = q
    if period:
        d = d[(d["ts"] >= pd.Timestamp(period[0])) & (d["ts"] < pd.Timestamp(period[1]))]
    n = d.groupby("user_id").size()
    keep = n[n >= min_queries].index
    d = d[d["user_id"].isin(keep)]
    v = pd.crosstab(d["user_id"], d[level], normalize="index")
    return v


def segment_users(q: pd.DataFrame, level: str = "l1_stage",
                  k_range: tuple[int, ...] = (2, 3, 4, 5, 6),
                  seed: int = 0) -> dict:
    """
    의도 구성 벡터에 KMeans. k 는 실루엣으로 고른다.
    라벨은 각 군집에서 기저율 대비 과대표집된 의도로 자동 명명한다.
    """
    V = intent_vectors(q, level)
    if len(V) < 50:
        return {"오류": "세그먼트 도출에 표본 부족"}
    X = StandardScaler().fit_transform(V.values)

    scores = {}
    for k in k_range:
        if k >= len(V):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        scores[k] = float(silhouette_score(X, km.labels_))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)

    V = V.copy()
    V["seg"] = km.labels_
    prof = V.groupby("seg").mean()
    overall = V.drop(columns="seg").mean()
    lift = prof / overall
    names = {s: "+".join(lift.loc[s].sort_values(ascending=False).head(2).index)
             for s in prof.index}
    prof.index = [f"S{s}:{names[s]}" for s in prof.index]

    size = V["seg"].value_counts(normalize=True).sort_index()
    size.index = [f"S{s}:{names[s]}" for s in size.index]

    return {"k": best_k, "실루엣": scores, "프로파일": prof.round(3),
            "규모": size.round(3), "_labels": pd.Series(km.labels_, index=V.index),
            "_names": names}


def segment_over_time(q: pd.DataFrame, seg: dict, freq: str = "M") -> pd.DataFrame:
    """시기별 세그먼트 구성비 추이. 사용자가 '바뀐 것'인지 '갈린 것'인지 본다."""
    if "_labels" not in seg:
        return pd.DataFrame()
    lab = seg["_labels"].rename("seg")
    d = q.join(lab, on="user_id").dropna(subset=["seg"])
    d["period"] = d["ts"].dt.to_period(freq).dt.start_time
    act = d.drop_duplicates(["period", "user_id"])
    ct = pd.crosstab(act["period"], act["seg"], normalize="index")
    ct.columns = [f"S{int(c)}:{seg['_names'][int(c)]}" for c in ct.columns]
    return ct.round(3)


# ------------------------------------------------------------ 2. 진입 질문

def entry_questions(q: pd.DataFrame, end_date, retention_days: int = 30,
                    min_n: int = 20) -> pd.DataFrame:
    """
    사용자의 '첫 질문'별 잔존율.
    어떤 첫 경험이 남는 사용자를 만드는지 — 온보딩 추천 질문의 근거.
    """
    end_date = pd.Timestamp(end_date)
    d = q.sort_values("ts")
    first = d.groupby("user_id").first()
    last = d.groupby("user_id")["ts"].max()

    # 관측 창이 충분한 사용자만 (우측 절단 방어)
    ok = first.index[(end_date - first["ts"]).dt.days >= retention_days]
    first = first.loc[ok]
    ret = ((last.loc[ok] - first["ts"]).dt.days >= retention_days).astype(int)

    out = pd.DataFrame({"l2_intent": first["l2_intent"],
                        "outcome": first["outcome"],
                        "ret": ret})
    g = out.groupby("l2_intent").agg(n=("ret", "size"), 잔존율=("ret", "mean"))
    g = g[g["n"] >= min_n]
    g["전체대비"] = g["잔존율"] - out["ret"].mean()

    byo = out.groupby("outcome").agg(n=("ret", "size"), 잔존율=("ret", "mean"))
    g.attrs["첫경험_결과별"] = byo.round(3)
    g.attrs["전체잔존율"] = float(out["ret"].mean())
    return g.sort_values("잔존율", ascending=False).round(3)


# ------------------------------------------------------------ 3. 이탈 지점

def exit_points(q: pd.DataFrame, end_date, grace_days: int = 30,
                min_n: int = 30) -> pd.DataFrame:
    """
    마지막 질의의 의도·결과 분포를 '종료 위험 lift'로 본다.

    주의: 마지막 질의는 정의상 마지막이므로 단순 분포는 의미가 없다.
    각 (의도, 결과) 조합이 '그 질의가 마지막이 될 확률'을 얼마나 올리는지
    전체 기저율 대비 lift 로 계산한다.
    """
    end_date = pd.Timestamp(end_date)
    d = q.sort_values(["user_id", "ts"]).copy()
    last_ts = d.groupby("user_id")["ts"].transform("max")
    churned = d.groupby("user_id")["ts"].transform(
        lambda s: (end_date - s.max()).days > grace_days)
    d["is_exit"] = (d["ts"].eq(last_ts) & churned).astype(int)

    base = d["is_exit"].mean()
    g = d.groupby(["l2_intent"]).agg(n=("is_exit", "size"), 종료율=("is_exit", "mean"))
    g = g[g["n"] >= min_n]
    g["lift"] = g["종료율"] / base

    byf = d.groupby(d["fail_code"].fillna("성공")).agg(
        n=("is_exit", "size"), 종료율=("is_exit", "mean"))
    byf["lift"] = byf["종료율"] / base
    g.attrs["실패코드별"] = byf.sort_values("lift", ascending=False).round(3)
    g.attrs["기저_종료율"] = float(base)
    return g.sort_values("lift", ascending=False).round(3)


# ------------------------------------------------------------ 4. 리텐션 지점

def retention_drivers(q: pd.DataFrame, window_days: int = 14,
                      min_n: int = 50) -> pd.DataFrame:
    """
    '이 의도를 성공적으로 경험하면 돌아오는가'를 개인 내에서 본다.

    사용자-일 단위로 (그날 경험한 의도 × 성공여부) → window_days 내 재방문 여부.
    사용자 고정효과 대신 사용자별 평균 재방문율을 차감해 개인 성향을 제거한다.
    """
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    days = d.drop_duplicates(["user_id", "date"])[["user_id", "date"]].sort_values(
        ["user_id", "date"])
    days["next_gap"] = days.groupby("user_id")["date"].diff(-1).dt.days.abs()
    days["returned"] = (days["next_gap"] <= window_days).fillna(False).astype(int)

    ud = d.merge(days[["user_id", "date", "returned"]], on=["user_id", "date"])
    ud["ok"] = ud["outcome"].eq("success")
    # 개인 성향 차감 (within-person)
    umean = ud.groupby("user_id")["returned"].transform("mean")
    ud["ret_dm"] = ud["returned"] - umean

    g = ud.groupby(["l2_intent", "ok"])["ret_dm"].agg(["size", "mean"]).reset_index()
    g = g[g["size"] >= min_n]
    piv = g.pivot(index="l2_intent", columns="ok", values="mean")
    cnt = g.pivot(index="l2_intent", columns="ok", values="size")
    if piv.shape[1] < 2:
        return pd.DataFrame({"안내": [
            "성공/실패 양쪽 관측이 있는 의도가 없습니다 — 성공효과 산출 불가"]})
    piv = piv.reindex(columns=[False, True])
    cnt = cnt.reindex(columns=[False, True])
    piv.columns = ["실패시", "성공시"]
    cnt.columns = ["n_실패", "n_성공"]
    out = piv.join(cnt).dropna()
    out["성공효과"] = out["성공시"] - out["실패시"]
    return out.sort_values("성공효과", ascending=False).round(4)
