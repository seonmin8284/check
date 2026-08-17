"""
인적 평가 ↔ 자동 프록시 캘리브레이션.

1,000건은 최종 답이 아니라 **프록시 검증용 골드셋**이다.
어느 자동 지표가 인적 판정을 예측하는지 재서, 검증된 것만 전체 로그에 확장한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROXIES = {
    "rel_coverage": "질문 내용어 적중률",
    "rel_jaccard": "내용어 자카드",
    "rel_bigram": "문자 bigram 자카드",
    "st_해석문장수": "해석 문장 수",
    "st_수치밀도": "수치 밀도(역방향 기대)",
    "st_본문길이": "본문 길이",
}
BINARY_PROXIES = {
    "st_요약선행문": "요약 선행문 유무",
    "_eff_effective": "실질 성공(EFFECTIVE)",
    "_next_bad": "다음 턴 재질문·형식재요청",
    "_is_fallback": "폴백 툴 호출",
}


def merge_review(sheet_done: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """평가 완료 시트와 숨김 키(정답·프록시)를 review_id 로 결합."""
    d = sheet_done.merge(keys, on="review_id", how="inner", suffixes=("", "_k"))
    for c in ("R_관련성", "R_정확성", "R_완결성", "R_구조성", "R_위해성"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].replace({"N": np.nan, "": np.nan}),
                                 errors="coerce")
    return d


def review_summary(d: pd.DataFrame) -> dict:
    """평가 결과 기술 통계. 층별로 나눠 보는 것이 핵심."""
    res = {}
    rel = d["R_관련성"]
    res["전체 관련성 0 비율"] = round(float(rel.eq(0).mean()), 4)
    if "stratum" in d.columns:
        by = d.groupby("stratum").agg(
            n=("R_관련성", "size"),
            관련성0비율=("R_관련성", lambda x: float(x.eq(0).mean())),
            정확성=("R_정확성", "mean"),
            완결성=("R_완결성", "mean"),
            구조성=("R_구조성", "mean"),
            위해성비율=("R_위해성", "mean"))
        res["층별"] = by.round(3)
    ok = d[d["R_관련성"].eq(1)]
    res["관련성 통과분 평균"] = {
        "정확성": round(float(ok["R_정확성"].mean()), 3),
        "완결성": round(float(ok["R_완결성"].mean()), 3),
        "구조성": round(float(ok["R_구조성"].mean()), 3)}
    if "R_정확성" in d.columns:
        res["정확성 검증불가(N) 비율"] = round(
            float(ok["R_정확성"].isna().mean()), 4)
    return res


def calibrate(d: pd.DataFrame, target: str = "R_관련성",
              positive_is_high: bool = True) -> pd.DataFrame:
    """
    각 프록시가 인적 판정을 얼마나 예측하는가.

    이진 타깃은 AUC, 순서형 타깃은 스피어만 상관으로 본다.
    AUC 0.80 이상이면 전체 로그 자동 채점 후보.
    """
    y = d[target]
    binary = set(y.dropna().unique()) <= {0, 1}
    rows = []
    for col, label in {**PROXIES, **BINARY_PROXIES}.items():
        if col not in d.columns:
            continue
        x = pd.to_numeric(d[col], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 50 or x[m].nunique() < 2:
            continue
        if binary:
            try:
                auc = float(roc_auc_score(y[m], x[m]))
            except ValueError:
                continue
            rows.append({"프록시": col, "설명": label, "n": int(m.sum()),
                         "AUC": auc, "방향": "정" if auc >= .5 else "역",
                         "AUC(방향보정)": max(auc, 1 - auc)})
        else:
            r = float(x[m].corr(y[m], method="spearman"))
            rows.append({"프록시": col, "설명": label, "n": int(m.sum()),
                         "스피어만": r, "절대값": abs(r)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"안내": ["계산 가능한 프록시가 없습니다"]})
    key = "AUC(방향보정)" if "AUC(방향보정)" in out.columns else "절대값"
    return out.sort_values(key, ascending=False).reset_index(drop=True).round(3)


def calibration_verdict(cal: pd.DataFrame, thr: float = 0.80) -> str:
    if "안내" in cal.columns or cal.empty:
        return "판정 불가"
    key = "AUC(방향보정)" if "AUC(방향보정)" in cal.columns else "절대값"
    top = cal.iloc[0]
    v = float(top[key])
    if key.startswith("AUC") and v >= thr:
        return (f"✅ 「{top['프록시']}」 AUC {v:.3f} — 전체 로그 자동 채점 가능. "
                "아래 임계값 표에서 운영점을 고르십시오.")
    if key.startswith("AUC"):
        return (f"⚠ 최고 AUC {v:.3f} (< {thr}) — 자동 채점 불가. "
                "행동·어휘 신호만으로는 내용 품질을 판정할 수 없다는 뜻이며, "
                "샘플링 검수를 정례화해야 합니다.")
    return (f"최고 상관 {v:.3f} ({top['프록시']}) — "
            f"{'실용 가능' if v >= .5 else '약함. 보조 지표로만'}")


def threshold_table(d: pd.DataFrame, proxy: str = "rel_coverage",
                    target: str = "R_관련성",
                    grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    """
    자동 채점 운영점 선택용. 임계값별 정밀도·재현율.

    무관 응답을 잡아내는 것이 목적이므로 **재현율을 우선**하십시오.
    놓친 무관 응답은 그대로 사용자에게 나갑니다.
    """
    x = pd.to_numeric(d[proxy], errors="coerce")
    y = d[target]
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    bad = y.eq(0)          # 무관 = 양성으로 둔다
    rows = []
    for t in grid:
        pred = x <= t
        tp = int((pred & bad).sum())
        fp = int((pred & ~bad).sum())
        fn = int((~pred & bad).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if prec and rec and (prec + rec) else np.nan)
        rows.append({"임계값": t, "탐지건수": tp + fp, "정밀도": prec,
                     "재현율": rec, "F1": f1})
    return pd.DataFrame(rows).round(3)


def iaa(a: pd.DataFrame, b: pd.DataFrame,
        cols=("R_관련성", "R_정확성", "R_완결성", "R_구조성", "R_위해성")) -> pd.DataFrame:
    """파일럿 이중 평가 일치도. 본 평가 전에 반드시 확인."""
    from sklearn.metrics import cohen_kappa_score
    m = a.merge(b, on="review_id", suffixes=("_A", "_B"))
    rows = []
    for c in cols:
        ca, cb = f"{c}_A", f"{c}_B"
        if ca not in m or cb not in m:
            continue
        x = pd.to_numeric(m[ca].replace({"N": np.nan, "": np.nan}), errors="coerce")
        y = pd.to_numeric(m[cb].replace({"N": np.nan, "": np.nan}), errors="coerce")
        k = x.notna() & y.notna()
        if k.sum() < 10:
            continue
        agree = float((x[k] == y[k]).mean())
        try:
            kappa = float(cohen_kappa_score(x[k].astype(int), y[k].astype(int)))
        except Exception:
            kappa = np.nan
        target = 0.80 if c == "R_관련성" else 0.60
        rows.append({"축": c, "n": int(k.sum()), "단순일치": agree,
                     "kappa": kappa, "목표": target,
                     "판정": "OK" if kappa >= target else "미달 — 가이드 보정 필요"})
    return pd.DataFrame(rows).round(3)
