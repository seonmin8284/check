"""
식별 가능성 사전 점검(guards).

분석이 '돌아가는 것'과 '해석할 수 있는 것'은 다르다.
사전 구간이 없는 DiD, 슬롯이 비어 있는 재질문 판정, 분산이 0인 지표는
숫자를 내놓지만 그 숫자는 아무것도 의미하지 않는다.

각 함수는 (통과여부, 사유) 를 돌려주고, run_analysis 가 이를 근거로
해당 분석을 건너뛰거나 결과에 경고를 붙인다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _res(ok: bool, msg: str, **kw) -> dict:
    return {"통과": ok, "사유": msg, **kw}


# ------------------------------------------------- 정책 효과 식별

def policy_pre_period(q: pd.DataFrame, policy_date, min_days: int = 21) -> dict:
    """
    정책(Protector) 도입 이전 구간이 존재하는가.

    데이터 시작일 ≈ 정책 도입일이면 '정책 없는 상태'가 관측되지 않는다.
    이 경우 차단의 효과는 **원리적으로 식별 불가**하다.
    차단 경험자/미경험자 비교는 정책 효과가 아니라
    '판단성 질문을 하는 사용자'와 '안 하는 사용자'의 차이를 재는 것이 된다.
    """
    if policy_date is None:
        return _res(True, "정책일 미지정 — 점검 생략")
    pol = pd.Timestamp(policy_date)
    start = q["ts"].min()
    pre_days = (pol - start).days
    pre_n = int((q["ts"] < pol).sum())
    if pre_days < min_days or pre_n < 200:
        return _res(False,
                    f"정책 도입 이전 구간이 {max(pre_days,0)}일 / {pre_n:,}건뿐입니다. "
                    "차단의 인과 효과는 식별 불가 — Cox·자기검열 결과를 "
                    "'정책 효과'로 해석하지 마십시오.",
                    사전일수=max(pre_days, 0), 사전건수=pre_n)
    return _res(True, f"사전 구간 {pre_days}일 / {pre_n:,}건 확보",
                사전일수=pre_days, 사전건수=pre_n)


def event_window(q: pd.DataFrame, event_date, end_date,
                 min_side_days: int = 21) -> dict:
    """
    이벤트(소스 중단) 전후 구간이 충분한가.
    사후가 짧으면 DiD·shift-share 가 극소 표본 위에서 계산된다.
    """
    ev, end = pd.Timestamp(event_date), pd.Timestamp(end_date)
    pre = int(((q["ts"] >= ev - pd.Timedelta(days=min_side_days)) & (q["ts"] < ev)).sum())
    post_days = (min(end, q["ts"].max()) - ev).days
    post = int((q["ts"] >= ev).sum())
    if post_days < min_side_days:
        return _res(False,
                    f"사후 구간이 {post_days}일뿐입니다(사후 {post:,}건). "
                    f"최소 {min_side_days}일 필요 — DiD·shift-share 결과는 "
                    "표본 아티팩트일 가능성이 높습니다.",
                    사후일수=post_days, 사후건수=post)
    return _res(True, f"사전 {pre:,}건 / 사후 {post_days}일 {post:,}건",
                사후일수=post_days, 사후건수=post)


# ------------------------------------------------- 라벨·컬럼 적합성

def slot_coverage(q: pd.DataFrame) -> dict:
    """
    slot_target 이 실제로 채워져 있는가.

    비어 있으면 재질문(REPEAT) 판정이 '같은 의도 반복'으로 퇴화하고,
    세부화(REFINE)는 아예 발생할 수 없다. 멀티턴 분석 전체가 무너진다.
    """
    if "slot_target" not in q.columns:
        return _res(False, "slot_target 컬럼 없음 — REPEAT/REFINE 구분 불가")
    filled = q["slot_target"].map(
        lambda v: isinstance(v, (list, tuple)) and len(v) > 0).mean()
    if filled < 0.05:
        return _res(False,
                    f"slot_target 이 {filled:.1%}만 채워져 있습니다. "
                    "REPEAT 판정이 '같은 의도 반복'으로 퇴화하고 REFINE 은 "
                    "0%가 됩니다 — 멀티턴 분석을 신뢰하지 마십시오. "
                    "query_text 유사도 폴백이 적용됩니다.",
                    충전율=round(float(filled), 4))
    return _res(True, f"slot_target 충전율 {filled:.1%}", 충전율=round(float(filled), 4))


def column_variance(q: pd.DataFrame, col: str, label: str = "") -> dict:
    """분산이 없는 지표는 판별력이 없다(예: 폴백 강제 호출로 cited 가 전부 True)."""
    if col not in q.columns or q[col].isna().all():
        return _res(False, f"{col} 없음")
    v = q[col]
    uniq = v.dropna().nunique()
    if uniq <= 1:
        return _res(False,
                    f"{label or col} 가 단일값({v.dropna().iloc[0] if len(v.dropna()) else 'NA'})입니다 — "
                    "판별력 없음. 폴백 강제 호출 구조에서는 모든 건에 툴이 붙어 "
                    "근거인용률·환각 프록시가 무의미해집니다.",
                    고유값=int(uniq))
    return _res(True, f"{col} 고유값 {uniq}", 고유값=int(uniq))


# ------------------------------------------------- 표본 구조

def panel_feasibility(q: pd.DataFrame, min_multi_share: float = 0.30) -> dict:
    """
    개인 패널 분석이 가능한 표본 구조인가.
    1회성 사용자가 압도적이면 리텐션·생존분석의 해상도가 사라진다.
    """
    per_user = q.groupby("user_id")["session_id"].nunique()
    multi = float((per_user > 1).mean())
    spu = float(per_user.mean())
    if multi < min_multi_share:
        return _res(False,
                    f"2세션 이상 사용자가 {multi:.1%}뿐입니다(세션/사용자 {spu:.2f}). "
                    "리텐션 중심 프레임이 맞지 않습니다 — 추적 지표를 "
                    "**세션 내 해결률**로 바꾸십시오.",
                    다회비율=round(multi, 4), 세션당사용자=round(spu, 3))
    return _res(True, f"2세션 이상 {multi:.1%}", 다회비율=round(multi, 4))


def group_significance(k: int, n: int, base: float, alpha: float = 0.05) -> dict:
    """
    부분군 비율이 전체 대비 유의하게 다른가 (정규 근사 1표본 검정).
    n 이 작은 부분군의 차이를 근거로 삼기 전에 반드시 통과시켜야 한다.
    """
    if n < 10:
        return _res(False, f"n={n} — 검정 불가")
    p = k / n
    se = np.sqrt(base * (1 - base) / n)
    z = (p - base) / se if se > 0 else 0.0
    from scipy import stats
    pv = float(2 * stats.norm.sf(abs(z)))
    return _res(pv < alpha,
                f"p={p:.3f} vs 기저 {base:.3f} · z={z:.2f} · p값={pv:.3f}"
                + ("" if pv < alpha else " — 유의하지 않음. 근거로 쓰지 마십시오"),
                비율=round(p, 4), z=round(z, 3), p값=round(pv, 4))


def detect_service_open(q: pd.DataFrame, freq: str = "W",
                        min_share: float = 0.25, run: int = 4) -> dict:
    """
    파일럿/내부테스트 구간을 탐지해 실질 서비스 오픈 시점을 추정한다.

    오픈 전 구간은 사용자 수가 극소수이고 리텐션이 0에 가까워, 그대로 두면
    코호트 비교·믹스 반사실 기준시기·shift-share 전반부를 통째로 오염시킨다.

    판정: 주간 신규 사용자 수가 정상기 중앙값의 min_share 이상으로
          run 주 연속 유지되는 첫 주를 오픈 시점으로 본다.
    """
    d = q.copy()
    d["date"] = d["ts"].dt.normalize()
    first = d.groupby("user_id")["date"].min()
    wk = first.dt.to_period(freq).dt.start_time.value_counts().sort_index()
    if len(wk) < run + 2:
        return _res(True, "구간이 짧아 오픈 시점 탐지 생략")

    med = float(wk.median())
    thr = med * min_share
    ok = (wk >= thr).to_numpy()
    idx = None
    for i in range(len(ok) - run + 1):
        if ok[i:i + run].all():
            idx = i
            break
    if idx is None or idx == 0:
        return _res(True, "파일럿 구간 없음 — 전체 기간 사용 가능",
                    제안시작일=None)

    open_wk = wk.index[idx]
    pre_users = int(wk.iloc[:idx].sum())
    pre_q = int((d["ts"] < open_wk).sum())
    return _res(False,
                f"{open_wk:%Y-%m-%d} 이전이 파일럿 구간으로 보입니다 "
                f"(신규 {pre_users:,}명 · 질의 {pre_q:,}건, "
                f"주간 신규가 정상기 중앙값 {med:,.0f}의 {min_share:.0%} 미만). "
                f"--start {open_wk:%Y-%m-%d} 로 잘라 재실행하십시오.",
                제안시작일=open_wk, 사전사용자=pre_users, 사전질의=pre_q)


def run_all(q: pd.DataFrame, policy_date=None, event_date=None,
            end_date=None) -> pd.DataFrame:
    """전체 점검을 한 번에 실행."""
    checks = {
        "정책 사전구간": policy_pre_period(q, policy_date),
        "이벤트 창": (event_window(q, event_date, end_date)
                      if event_date is not None and end_date is not None
                      else _res(True, "미지정 — 생략")),
        "슬롯 충전율": slot_coverage(q),
        "근거인용 분산": column_variance(q, "cited", "근거인용"),
        "패널 가용성": panel_feasibility(q),
        "서비스 오픈 시점": detect_service_open(q),
    }
    return pd.DataFrame([
        {"점검": k, "통과": "OK" if v["통과"] else "실패", "사유": v["사유"]}
        for k, v in checks.items()]), checks
