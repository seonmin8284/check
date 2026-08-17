"""
응답 지연이 사용자 경험에 미치는 영향.

ELAPSED_TIME 은 사용자가 체감하는 전체 응답 시간이다(차단이든 정상이든).
같은 축의 값이므로 비교는 가능하지만, **경로가 다르면 지연의 의미가 다르다.**

  차단      → 정책 판정만 하고 즉시 반환       (빠름이 곧 성의 없음일 수 있음)
  폴백      → 무관한 답을 빠르게 반환          (가장 나쁜 조합)
  정상 1스텝 → 툴 1회 + 생성                   (진짜 성능)
  정상 다스텝 → 툴 연쇄                        (복잡도 비용)
  기술 실패  → 오류 반환                        (지연과 무관)

이를 섞으면 '빠른 응답의 실패율이 100%' 같은 역인과가 나온다.
따라서 ① 경로를 먼저 나누고 ② 정상 층 안에서만 지연 효과를 본다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .relevance import FALLBACK_TOOLS, OTH_CODES
from .track_a import tag_session
from .track_c import absorb

PATHS = ["차단", "폴백", "정상 1스텝", "정상 다스텝", "기술 실패"]
NORMAL = ("정상 1스텝", "정상 다스텝")


# ═══════════════════════════════ 1. 경로 분류

def classify_path(q: pd.DataFrame) -> pd.Series:
    """응답이 어떤 경로로 만들어졌는가. 지연 해석의 전제."""
    d = q
    blocked = d["outcome"].eq("blocked").fillna(False).to_numpy(dtype=bool)

    fb = np.zeros(len(d), dtype=bool)
    if "tool_called" in d.columns:
        tc = d["tool_called"].fillna("").astype(str).str.lower()
        fb |= tc.apply(lambda s: any(f in s for f in FALLBACK_TOOLS)).to_numpy()
    if "intent_pred" in d.columns:
        fb |= d["intent_pred"].astype(str).str.upper().isin(
            [c.upper() for c in OTH_CODES]).to_numpy()

    ok = d["outcome"].eq("success").fillna(False).to_numpy(dtype=bool)
    steps = (pd.to_numeric(d["tool_steps"], errors="coerce").fillna(0)
             if "tool_steps" in d.columns else pd.Series(0, index=d.index))
    multi = (steps >= 2).to_numpy()

    return pd.Series(np.select(
        [blocked, fb, ok & multi, ok & ~multi],
        ["차단", "폴백", "정상 다스텝", "정상 1스텝"],
        default="기술 실패"), index=d.index)


# ═══════════════════════════════ 2. 결과 변수

def add_outcomes(q: pd.DataFrame, fu: pd.DataFrame,
                 sess: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    사용자 관점 결과 세 가지를 붙인다.

      다음턴발생 — 대화가 이어졌는가
      후속간격   — 응답을 읽었는가, 기다리다 지쳐 넘겼는가
      세션해결   — 결국 목적을 이뤘는가
    """
    d = q.copy()
    n = fu.sort_values(["session_id", "ts"]).copy()
    g = n.groupby("session_id")
    n["_next_ts"] = g["ts"].shift(-1)
    n["_next_kind"] = g["turn_kind"].shift(-1)
    n["_gap"] = (n["_next_ts"] - n["ts"]).dt.total_seconds()
    d = d.merge(n[["query_id", "_next_ts", "_next_kind", "_gap"]],
                on="query_id", how="left")
    d["다음턴발생"] = d["_next_ts"].notna().astype(float)
    d["후속간격"] = d["_gap"]
    d["되묻기"] = d["_next_kind"].isin(["REPEAT", "FORMAT"]).fillna(False).astype(float)
    if sess is not None and "session_outcome" in sess.columns:
        d = d.join(sess["session_outcome"].rename("_so"), on="session_id")
        d["세션해결"] = d["_so"].isin(["RESOLVED", "RESOLVED_HARD"]).astype(float)
    else:
        d["세션해결"] = np.nan
    d["경로"] = classify_path(d)
    d["시간대"] = d["ts"].dt.hour.map(tag_session)
    d["초"] = pd.to_numeric(d["latency_ms"], errors="coerce") / 1000
    return d


# ═══════════════════════════════ 3. 표1 — 경로별 지연과 결과

def path_summary(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("경로").agg(
        건수=("초", "size"),
        중앙지연=("초", "median"),
        p95지연=("초", lambda x: x.quantile(.95)),
        다음턴발생률=("다음턴발생", "mean"),
        되묻기율=("되묻기", "mean"),
        세션해결률=("세션해결", "mean"))
    return g.reindex([p for p in PATHS if p in g.index]).round(3)


def path_share_of_latency(d: pd.DataFrame) -> dict:
    """
    전체 지연 분산 중 '경로'와 '스텝 수'가 설명하는 몫.
    스텝 수가 대부분을 설명하면 속도 문제가 아니라 계획 복잡도 문제다.
    """
    x = np.log1p(pd.to_numeric(d["초"], errors="coerce"))
    m = x.notna()
    if m.sum() < 200:
        return {}
    tot = float(x[m].var())
    res = {}
    for col, lab in [("경로", "경로"), ("l2_intent", "질문 유형")]:
        if col in d.columns:
            within = float(x[m].groupby(d.loc[m, col]).transform("mean").var())
            res[f"{lab} 설명력"] = round(within / tot, 3) if tot else np.nan
    if "tool_steps" in d.columns:
        st = pd.to_numeric(d["tool_steps"], errors="coerce")
        mm = m & st.notna()
        if mm.sum() > 200:
            within = float(x[mm].groupby(st[mm]).transform("mean").var())
            res["호출 스텝수 설명력"] = round(within / tot, 3) if tot else np.nan
            res["스텝별 중앙지연"] = (d[mm].groupby(st[mm])["초"].median()
                                      .round(2).to_dict())
    return res


# ═══════════════════════════════ 4. 표2 — 정상 층 내 지연 효과

def residual_latency(d: pd.DataFrame,
                     fe=("l2_intent", "tool_steps")) -> pd.Series:
    """
    질문 유형·호출 스텝수를 흡수하고 남은 지연.

    '같은 유형, 같은 스텝인데 유독 느린 건'만 남으므로,
    난이도 교란을 뺀 순수 인프라 지연에 가깝다.
    """
    x = d.copy()
    x["_y"] = np.log1p(pd.to_numeric(x["초"], errors="coerce"))
    cols = [c for c in fe if c in x.columns]
    x = x.dropna(subset=["_y"] + cols)
    if len(x) < 200 or not cols:
        return pd.Series(np.nan, index=d.index)
    for c in cols:
        x[c] = x[c].astype(str)
    Z, _ = absorb(x, ["_y"], cols)
    return pd.Series(Z[:, 0], index=x.index).reindex(d.index)


def latency_effect(d: pd.DataFrame, use_residual: bool = True,
                   q_bins: int = 6, min_n: int = 100) -> pd.DataFrame:
    """
    정상 응답 안에서만, 교란을 흡수한 뒤 지연 분위별 결과.
    """
    n = d[d["경로"].isin(NORMAL)].copy()
    if len(n) < min_n * 2:
        return pd.DataFrame({"안내": ["정상 응답 표본이 부족합니다"]})
    if use_residual:
        n["_r"] = residual_latency(n)
        base = "_r"
        if n["_r"].isna().all():
            base, use_residual = "초", False
    else:
        base = "초"
    n = n.dropna(subset=[base])
    try:
        n["분위"] = pd.qcut(n[base], q_bins, labels=[f"Q{i}" for i in
                                                    range(1, q_bins + 1)],
                            duplicates="drop")
    except ValueError:
        return pd.DataFrame({"안내": ["지연 분포가 좁아 분위를 나눌 수 없습니다"]})
    g = n.groupby("분위", observed=True).agg(
        건수=("초", "size"), 중앙지연=("초", "median"),
        다음턴발생률=("다음턴발생", "mean"),
        후속간격중앙=("후속간격", "median"),
        되묻기율=("되묻기", "mean"),
        세션해결률=("세션해결", "mean"))
    g.attrs["기준"] = ("질문유형·스텝수 흡수 후 잔차" if use_residual else "원시 지연")
    return g.round(3)


def latency_by_session(d: pd.DataFrame, min_n: int = 80,
                       bins=(0, 3, 6, 10, 15, 20, 1e9)) -> pd.DataFrame:
    """
    시간대 × 지연 구간. 같은 5초라도 개장과 야간의 체감이 다르다.
    시간대별 SLA 차등의 근거.
    """
    n = d[d["경로"].isin(NORMAL)].copy()
    labels = [f"{int(bins[i])}~{int(bins[i+1]) if bins[i+1] < 1e8 else '∞'}s"
              for i in range(len(bins) - 1)]
    n["지연구간"] = pd.cut(n["초"], bins=bins, labels=labels, right=False)
    g = (n.groupby(["시간대", "지연구간"], observed=True)
           .agg(건수=("초", "size"), 다음턴발생률=("다음턴발생", "mean"))
           .reset_index())
    g = g[g["건수"] >= min_n]
    if g.empty:
        return pd.DataFrame({"안내": ["시간대×지연 셀 표본 부족"]})
    piv = g.pivot(index="시간대", columns="지연구간", values="다음턴발생률")
    order = ["장전", "개장", "장중", "장후", "야간"]
    return piv.reindex([o for o in order if o in piv.index]).round(3)


# ═══════════════════════════════ 5. 인내 한계 탐색

def patience_threshold(d: pd.DataFrame, by: str | None = None,
                       max_sec: int = 30, min_n: int = 150) -> pd.DataFrame:
    """
    몇 초부터 문제인가.

    1초 단위로 누적하며 '다음 턴 발생률'의 기울기가 꺾이는 지점을 찾는다.
    꺾임이 없으면 그것도 답이다 — 관측 범위 안에서 지연이 이탈을 만들지 않는다.
    """
    n = d[d["경로"].isin(NORMAL)].dropna(subset=["초"]).copy()
    groups = [(None, n)] if by is None else list(n.groupby(by))
    rows = []
    for key, g in groups:
        if len(g) < min_n:
            continue
        g = g[g["초"] <= max_sec]
        pts = []
        for t in range(1, max_sec):
            sub = g[(g["초"] >= t - 1) & (g["초"] < t + 1)]
            if len(sub) >= 30:
                pts.append((t, float(sub["다음턴발생"].mean()), len(sub)))
        if len(pts) < 5:
            continue
        arr = pd.DataFrame(pts, columns=["초", "발생률", "n"])
        arr["기울기"] = arr["발생률"].diff()
        base = arr["기울기"].iloc[1:4].mean()
        knee = None
        for i in range(3, len(arr) - 1):
            if arr["기울기"].iloc[i] < min(base * 3, -0.02) and \
                    arr["기울기"].iloc[i + 1] < 0:
                knee = float(arr["초"].iloc[i])
                break
        rows.append({(by or "전체"): key if key is not None else "전체",
                     "관측": int(len(g)),
                     "발생률 최고": float(arr["발생률"].max()),
                     "발생률 최저": float(arr["발생률"].min()),
                     "낙차": float(arr["발생률"].max() - arr["발생률"].min()),
                     "꺾이는 지점(초)": knee})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"안내": ["임계값 탐색 표본 부족"]})
    return out.round(3)


def threshold_verdict(pt: pd.DataFrame) -> str:
    if "안내" in pt.columns or pt.empty:
        return "판정 불가"
    knee = pt["꺾이는 지점(초)"].dropna()
    drop = float(pt["낙차"].max())
    if knee.empty:
        return (f"관측 범위 안에서 뚜렷한 인내 한계가 나타나지 않음 "
                f"(최대 낙차 {drop:.1%}p). 속도가 이탈의 주된 원인이라는 근거는 "
                "확인되지 않았으며, 속도 개선의 성과 지표로 이탈 감소를 "
                "설정하는 것은 적절하지 않음.")
    return (f"인내 한계가 관측됨 — {knee.min():.0f}~{knee.max():.0f}초 구간에서 "
            f"대화 지속률이 하락함(최대 낙차 {drop:.1%}p). 시간대별 응답 목표를 "
            "해당 값 이하로 설정할 필요가 있음.")


# ═══════════════════════════════ 6. 빠른 거절

def fast_rejection(d: pd.DataFrame, path: str = "차단",
                   bins=(0, 1, 2, 3, 5, 1e9)) -> pd.DataFrame:
    """
    거절이 너무 빨리 오는 것이 오히려 성의 없게 느껴지는가.

    매우 빠른 차단일수록 이탈이 높다면 이는 속도 문제가 아니라
    **응답 경험 설계** 문제다. 대체 안내를 붙이면 자연히 느려지면서
    이탈도 줄 수 있다. 폴백도 같은 논리로 본다.
    """
    n = d[d["경로"].eq(path)].dropna(subset=["초"]).copy()
    if len(n) < 100:
        return pd.DataFrame({"안내": [f"'{path}' 경로 표본이 부족합니다"]})
    labels = [f"{bins[i]:g}~{bins[i+1]:g}s" if bins[i+1] < 1e8
              else f"{bins[i]:g}s+" for i in range(len(bins) - 1)]
    n["지연구간"] = pd.cut(n["초"], bins=bins, labels=labels, right=False)
    g = n.groupby("지연구간", observed=True).agg(
        건수=("초", "size"), 다음턴발생률=("다음턴발생", "mean"),
        되묻기율=("되묻기", "mean"), 세션해결률=("세션해결", "mean"))
    g = g[g["건수"] >= 30]
    if len(g) >= 2:
        g.attrs["추세"] = float(g["다음턴발생률"].iloc[-1]
                                - g["다음턴발생률"].iloc[0])
    return g.round(3)