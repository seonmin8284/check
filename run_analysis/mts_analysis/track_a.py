"""
A트랙 — 집계 분석 (사용자 ID 불필요).

1. 일중 의도 프로파일
2. 믹스 고정(mix-adjusted) 실패율  ← 심슨의 역설 방어
3. FnGuide 의도군 DiD + 평행추세(이벤트 스터디) 검정
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .schema import FNGUIDE_INTENTS, CONTROL_INTENTS

MARKET_SESSION = [
    (0, 8, "야간"), (8, 9, "장전"), (9, 10, "개장"),
    (10, 15, "장중"), (15, 18, "장후"), (18, 24, "야간"),
]


def tag_session(hour: int) -> str:
    for lo, hi, name in MARKET_SESSION:
        if lo <= hour < hi:
            return name
    return "야간"


def intraday_profile(q: pd.DataFrame, by: str = "l1_stage") -> pd.DataFrame:
    """시간대 × 의도(또는 단계) 구성비. 진입화면 추천질문 시간대 배치의 근거."""
    d = q.copy()
    d["hour"] = d["ts"].dt.hour
    ct = d.pivot_table(index="hour", columns=by, values="sample_weight",
                       aggfunc="sum", fill_value=0.0)
    return ct.div(ct.sum(axis=1), axis=0)


def session_profile(q: pd.DataFrame, by: str = "l1_stage") -> pd.DataFrame:
    """시장 세션(장전/개장/장중/장후/야간) × 의도 구성비."""
    d = q.copy()
    d["sess"] = d["ts"].dt.hour.map(tag_session)
    ct = d.pivot_table(index="sess", columns=by, values="sample_weight",
                       aggfunc="sum", fill_value=0.0)
    ct = ct.div(ct.sum(axis=1), axis=0)
    return ct.reindex([s for s in ["장전", "개장", "장중", "장후", "야간"] if s in ct.index])


def top_intent_by_session(q: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    p = session_profile(q, by="l2_intent")
    out = {}
    for sess in p.index:
        out[sess] = ", ".join(f"{i}({v:.0%})" for i, v in
                              p.loc[sess].sort_values(ascending=False).head(k).items())
    return pd.DataFrame({"상위 의도": out})


# ------------------------------------------------------------- 믹스 고정

def mix_adjusted_rate(q: pd.DataFrame, base_period: tuple[str, str],
                      freq: str = "W", metric: str = "is_fail") -> pd.DataFrame:
    """
    직접표준화. 기준기 의도 구성비를 고정한 뒤 기간별 실패율을 재가중한다.

    조 실패율(crude)과 믹스 고정 실패율이 반대 방향으로 움직이면
    → 개선의 실체는 성능이 아니라 의도 믹스 이동이다.
    """
    d = q.copy()
    if metric not in d.columns:
        d["is_fail"] = (d["fail_code"].notna() & ~d["fail_code"].isin(["C1"])).astype(float)
    d["period"] = d["ts"].dt.to_period(freq).dt.start_time

    lo, hi = pd.Timestamp(base_period[0]), pd.Timestamp(base_period[1])
    base = d[(d["ts"] >= lo) & (d["ts"] < hi)]
    if base.empty:
        raise ValueError("기준기에 데이터가 없습니다.")
    w_base = base.groupby("l2_intent")["sample_weight"].sum()
    w_base = w_base / w_base.sum()

    cell = (d.groupby(["period", "l2_intent"])
              .apply(lambda x: pd.Series({
                  "rate": np.average(x[metric], weights=x["sample_weight"]),
                  "vol": x["sample_weight"].sum()}), include_groups=False)
              .reset_index())

    rows = []
    for per, g in cell.groupby("period"):
        crude = np.average(g["rate"], weights=g["vol"])
        gg = g.set_index("l2_intent")
        common = w_base.index.intersection(gg.index)
        w = w_base.loc[common]
        adj = float((gg.loc[common, "rate"] * w).sum() / w.sum())
        rows.append({"period": per, "조_실패율": crude, "믹스고정_실패율": adj,
                     "질의량": g["vol"].sum()})
    out = pd.DataFrame(rows).set_index("period")
    out["괴리"] = out["믹스고정_실패율"] - out["조_실패율"]
    return out


# ------------------------------------------------------------- DiD

def _prep_did(q: pd.DataFrame, outage_date, treat, control, freq="D") -> pd.DataFrame:
    d = q[q["l2_intent"].isin(list(treat) + list(control))].copy()
    d["is_fail"] = (d["fail_code"].notna() & ~d["fail_code"].isin(["C1"])).astype(float)
    d["treat"] = d["l2_intent"].isin(treat).astype(int)
    d["date"] = d["ts"].dt.to_period(freq).dt.start_time
    cell = (d.groupby(["l2_intent", "treat", "date"])
              .apply(lambda x: pd.Series({
                  "fail_rate": np.average(x["is_fail"], weights=x["sample_weight"]),
                  "n": x["sample_weight"].sum()}), include_groups=False)
              .reset_index())
    cell["post"] = (cell["date"] >= pd.Timestamp(outage_date)).astype(int)
    cell["rel"] = ((cell["date"] - pd.Timestamp(outage_date)).dt.days // 7).astype(int)
    return cell


def did(q: pd.DataFrame, outage_date, treat=FNGUIDE_INTENTS,
        control=CONTROL_INTENTS, freq: str = "D") -> dict:
    """
    이중차분: 실패율 ~ 의도FE + 날짜FE + treat×post
    클러스터는 의도 단위(수가 적음 → 결과 해석 시 주의 경고 포함).
    """
    cell = _prep_did(q, outage_date, treat, control, freq)
    n_t = cell.loc[cell["treat"].eq(1), "l2_intent"].nunique()
    n_c = cell.loc[cell["treat"].eq(0), "l2_intent"].nunique()
    pre = int((cell["post"] == 0).sum())
    post = int((cell["post"] == 1).sum())
    if n_t < 1 or n_c < 1 or n_t + n_c < 3 or pre == 0 or post == 0:
        return {"불가": (f"DiD 식별 불가 — 처치 의도 {n_t}종 / 대조 {n_c}종, "
                         f"사전 {pre}셀 / 사후 {post}셀. "
                         "schema.py 의 FNGUIDE_INTENTS·CONTROL_INTENTS 가 실제 로그의 "
                         "의도와 맞는지, --outage 가 데이터 기간 안에 있는지 확인하십시오."),
                "클러스터수": n_t + n_c}
    m = smf.wls("fail_rate ~ treat:post + C(l2_intent) + C(date)",
                data=cell, weights=cell["n"]).fit(
        cov_type="cluster", cov_kwds={"groups": cell["l2_intent"]})
    key = [k for k in m.params.index if "treat:post" in k][0]
    n_clu = cell["l2_intent"].nunique()
    return {
        "효과(실패율 pp)": float(m.params[key]),
        "표준오차": float(m.bse[key]),
        "p값": float(m.pvalues[key]),
        "95%CI": (float(m.conf_int().loc[key, 0]), float(m.conf_int().loc[key, 1])),
        "클러스터수": n_clu,
        "경고": ("클러스터 수가 적어 표준오차가 과소추정될 수 있음. "
                 "wild bootstrap 병행 권장." if n_clu < 15 else ""),
        "_model": m, "_cell": cell,
    }


def event_study(q: pd.DataFrame, outage_date, treat=FNGUIDE_INTENTS,
                control=CONTROL_INTENTS, span: int = 6) -> pd.DataFrame:
    """
    평행추세 검정. rel=-1 을 기준으로 lead 계수가 0 근처면 가정 충족.
    lead 가 유의하면 DiD 결과를 신뢰할 수 없다.
    """
    cell = _prep_did(q, outage_date, treat, control, "D")
    cell = cell[cell["rel"].between(-span, span)].copy()
    if cell["l2_intent"].nunique() < 3 or cell["treat"].nunique() < 2:
        out = pd.DataFrame(columns=["주차(rel)", "계수", "SE", "CI_lo", "CI_hi", "구간"])
        out.attrs["불가"] = "이벤트스터디 식별 불가 — 처치·대조 의도가 부족합니다"
        return out

    # rel = -1 을 기준으로 명시적 상호작용 더미 생성
    ks = sorted(k for k in cell["rel"].unique() if k != -1)
    terms = []
    for k in ks:
        col = f"ev_{'m' if k < 0 else 'p'}{abs(k)}"
        cell[col] = cell["treat"] * (cell["rel"] == k).astype(int)
        terms.append((k, col))

    formula = ("fail_rate ~ " + " + ".join(c for _, c in terms)
               + " + C(l2_intent) + C(date)")
    m = smf.wls(formula, data=cell, weights=cell["n"]).fit(
        cov_type="cluster", cov_kwds={"groups": cell["l2_intent"]})

    rows = []
    for k, col in terms:
        if col not in m.params.index:
            continue
        ci = m.conf_int().loc[col]
        rows.append({"주차(rel)": k, "계수": float(m.params[col]),
                     "SE": float(m.bse[col]),
                     "CI_lo": float(ci.iloc[0]), "CI_hi": float(ci.iloc[1])})
    rows.append({"주차(rel)": -1, "계수": 0.0, "SE": 0.0, "CI_lo": 0.0, "CI_hi": 0.0})
    out = pd.DataFrame(rows).sort_values("주차(rel)").reset_index(drop=True)
    out["구간"] = np.where(out["주차(rel)"] < 0, "사전(lead)", "사후(lag)")

    # 사전 계수 결합검정 — 개별 계수를 각각 보면 다중비교로 위양성이 난다
    leads = [c for k, c in terms if k < 0 and c in m.params.index]
    if leads:
        ft = m.f_test(" = 0, ".join(leads) + " = 0")
        out.attrs["pretrend_F"] = float(np.squeeze(ft.fvalue))
        out.attrs["pretrend_p"] = float(np.squeeze(ft.pvalue))
        # 결합검정은 기준주차 선택에 민감하다. DiD 를 실제로 위협하는 것은
        # '수준 차이'가 아니라 '추세'이므로 사전 계수의 기울기를 따로 검정한다.
        L = out[out["주차(rel)"] < 0].copy()
        L = L[L["SE"] > 0]
        if len(L) >= 3:
            wls = smf.wls("계수 ~ Q('주차(rel)')", data=L,
                          weights=1.0 / L["SE"] ** 2).fit()
            key = [k for k in wls.params.index if "주차" in k][0]
            out.attrs["pretrend_slope"] = float(wls.params[key])
            out.attrs["pretrend_slope_p"] = float(wls.pvalues[key])
    return out


def parallel_trend_verdict(es: pd.DataFrame) -> str:
    """개별 계수가 아니라 결합검정으로 판정한다(다중비교 위양성 방지)."""
    lead = es[es["주차(rel)"] < 0]
    if lead.empty:
        return "사전 구간 없음 — 평행추세 검정 불가"
    p = es.attrs.get("pretrend_p")
    sp = es.attrs.get("pretrend_slope_p")
    if p is None:
        return "결합검정 불가 — 개별 계수로만 판단"
    joint = f"결합 p={p:.3f}"
    slope = f"기울기 p={sp:.3f}" if sp is not None else "기울기 검정 불가"
    if p >= .05:
        return f"✅ 평행추세 지지 ({joint}, {slope})"
    if sp is not None and sp >= .05:
        return (f"△ 결합검정은 기각되나 추세는 없음 ({joint}, {slope}) — "
                "기준주차 수준차 성격. DiD 사용 가능하되 기준주차 민감도 확인 권장")
    return (f"⚠️ 평행추세 위배 ({joint}, {slope}) — "
            "DiD 대신 이벤트스터디 또는 합성대조군 고려")
