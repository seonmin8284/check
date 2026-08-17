"""
어노테이션 산출물 자체를 분석한다.

confidence / needs_review / secondary 는 라벨의 부산물이 아니라
**택소노미 설계의 진단 데이터**다. 저확신이 몰리는 지점이 곧 경계 문제이고,
secondary 공기 패턴이 곧 통합·분할 후보다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import INTENT_TO_STAGE


def review_summary(q: pd.DataFrame) -> dict:
    """needs_review 규모와 분포. 본 집계에서 분리해야 할 대상."""
    if "needs_review" not in q.columns:
        return {"안내": "needs_review 없음"}
    r = q["needs_review"].fillna(False).astype(bool)
    by_intent = (q.assign(_r=r).groupby("l2_intent")["_r"]
                   .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "검토율"}))
    by_intent = by_intent[by_intent["n"] >= 20].sort_values("검토율", ascending=False)
    by_stage = (q.assign(_r=r).groupby("l1_stage")["_r"]
                  .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "검토율"}))
    return {"전체_검토율": round(float(r.mean()), 4),
            "검토건수": int(r.sum()),
            "단계별": by_stage.round(4),
            "의도별_상위": by_intent.head(12).round(4),
            "해석": ("검토율이 높은 의도는 택소노미 경계가 모호하다는 신호입니다. "
                     "본 집계에서는 분리하고, 가이드 개정 1순위로 두십시오.")}


def confidence_profile(q: pd.DataFrame, low: float = 0.7) -> dict:
    """
    확신도 분포. 저확신이 특정 의도에 몰리면 그것이 경계 문제의 위치다.
    facet 별로도 봐서 어느 축이 흔들리는지 짚는다.
    """
    if "confidence" not in q.columns or q["confidence"].isna().all():
        return {"안내": "confidence 없음"}
    c = pd.to_numeric(q["confidence"], errors="coerce")
    d = q.assign(_c=c, _low=(c < low))

    by_intent = (d.groupby("l2_intent")["_low"].agg(["size", "mean"])
                   .rename(columns={"size": "n", "mean": "저확신율"}))
    by_intent["평균확신"] = d.groupby("l2_intent")["_c"].mean()
    by_intent = by_intent[by_intent["n"] >= 20].sort_values("저확신율", ascending=False)

    facet_cols = [c_ for c_ in ["f1_target_type", "f2_tense", "f3_personal",
                                "f4_compliance", "f5_response", "f6_turn"]
                  if c_ in d.columns]
    by_facet = {fc: d.groupby(fc)["_c"].mean().round(3).to_dict() for fc in facet_cols}

    # 저확신 건의 실패율 — 라벨 불확실이 실제 실패와 겹치는지
    link = None
    if "outcome" in d.columns:
        link = (d.groupby("_low")["outcome"]
                  .apply(lambda s: float((~s.eq("success")).mean())).round(4).to_dict())

    return {"평균확신": round(float(c.mean()), 3),
            f"저확신율(<{low})": round(float(d["_low"].mean()), 4),
            "의도별_상위": by_intent.head(12).round(3),
            "facet별_평균확신": by_facet,
            "저확신여부별_실패율": link,
            "해석": ("저확신 구간의 실패율이 눈에 띄게 높으면, 모델이 어려워하는 발화와 "
                     "어노테이터가 어려워하는 발화가 같다는 뜻입니다. "
                     "택소노미 개정이 곧 성능 개선으로 이어질 여지가 큽니다.")}


def secondary_cooccurrence(q: pd.DataFrame, min_n: int = 15) -> pd.DataFrame:
    """
    primary × secondary 공기 행렬.

    특정 쌍이 반복해서 함께 나오면 둘 중 하나다.
      (a) 두 의도가 실은 하나 — 통합 후보
      (b) 하나가 다른 하나의 하위 — 계층 재배치 후보
    같은 단계 안에서의 공기는 (a), 단계를 가로지르면 (b) 쪽일 때가 많다.
    """
    if "secondary" not in q.columns:
        return pd.DataFrame({"안내": ["secondary 없음"]})
    d = q[["l2_intent", "secondary"]].copy()
    d["secondary"] = d["secondary"].map(
        lambda v: v if isinstance(v, (list, tuple)) else [])
    d = d.explode("secondary").dropna(subset=["secondary"])
    d = d[d["secondary"].astype(str).str.len() > 0]
    if d.empty:
        return pd.DataFrame({"안내": ["secondary 값 없음"]})

    g = d.groupby(["l2_intent", "secondary"]).size().rename("공기건수").reset_index()
    g = g[g["공기건수"] >= min_n]
    base = q.groupby("l2_intent").size().rename("primary건수")
    g = g.join(base, on="l2_intent")
    g["공기율"] = g["공기건수"] / g["primary건수"]
    g["같은단계"] = (g["l2_intent"].map(INTENT_TO_STAGE)
                     == g["secondary"].map(INTENT_TO_STAGE))
    g["후보"] = np.where(g["같은단계"], "통합 검토", "계층 재배치 검토")
    return g.sort_values("공기율", ascending=False).reset_index(drop=True).round(3)


def split_review(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """본 집계용 / 검토 대기용으로 분리."""
    if "needs_review" not in q.columns:
        return q, q.iloc[0:0]
    r = q["needs_review"].fillna(False).astype(bool)
    return q[~r].copy(), q[r].copy()


def weight_by_confidence(q: pd.DataFrame, floor: float = 0.5) -> pd.Series:
    """
    확신도를 표본 가중에 반영한 값을 반환한다(선택).

    저확신 라벨을 버리지 않되 영향력을 줄이는 절충안입니다.
    민감도 분석용으로만 쓰고, 본 집계는 검토건 분리 방식을 우선하십시오.
    """
    if "confidence" not in q.columns or q["confidence"].isna().all():
        return q["sample_weight"]
    c = pd.to_numeric(q["confidence"], errors="coerce").fillna(1.0).clip(floor, 1.0)
    return q["sample_weight"] * c
