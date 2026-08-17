"""
문제별 이탈 기여도.

"차단이 X%, 데이터 부족이 Y%" 같은 단일 분해는 만들 수 없다.
  · 이탈이 단일 사건이 아니고 (대화 중단 / 기능 포기 / 서비스 이탈)
  · 한 사용자가 여러 문제를 동시에 겪어 기여도가 겹치고
  · '그 문제가 없었다면'을 관측할 반사실이 없기 때문이다.

대신 성격이 다른 네 가지 답을 낸다.
  A 대화 중단 귀속   — 세션을 끝낸 '마지막 실패'로 귀속하면 배타적으로 나뉜다
  B 노출-반응        — 겪은 사용자와 안 겪은 사용자의 차이 (합산하지 않는다)
  C 단독 해결 효과   — 각 문제를 하나씩만 없앴을 때의 상한
  D 문제 개수 누적   — 여러 문제를 겪을수록 나빠지는가

각 결과에는 근거 등급(준-인과 / 상관 / 가정)을 함께 붙인다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .relevance import FALLBACK_TOOLS, OTH_CODES

# 문제 정의 — 실패코드를 배타적 그룹으로 묶는다
PROBLEMS = {
    "분류 실패(무관한 답)": {"codes": (), "fallback": True,
                             "처방": "라우팅 개선", "담당": "AI"},
    "차단 후 대안 없음": {"codes": ("C3",), "처방": "대체 안내 설계", "담당": "준법·AI"},
    "자료 부재": {"codes": ("D1", "D2", "D3"), "처방": "조달·복구", "담당": "프로덕트"},
    "기능 미호출·미구현": {"codes": ("T1", "T2"), "처방": "개발·라우팅", "담당": "엔지니어링"},
    "인증 미처리": {"codes": ("A1",), "처방": "인증 게이트", "담당": "UX"},
    "대화 맥락 유실": {"codes": ("S1", "X1"), "처방": "되묻기 설계", "담당": "대화설계"},
    "모델 응답 오류": {"codes": ("M1",), "처방": "학습·프롬프트", "담당": "AI"},
}


def _fallback_mask(q: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=q.index)
    if "tool_called" in q.columns:
        tc = q["tool_called"].fillna("").astype(str).str.lower()
        m |= tc.apply(lambda s: any(f in s for f in FALLBACK_TOOLS))
    if "intent_pred" in q.columns:
        m |= q["intent_pred"].astype(str).str.upper().isin(
            [c.upper() for c in OTH_CODES])
    return m


def tag_problem(q: pd.DataFrame) -> pd.Series:
    """
    각 질의를 하나의 문제로만 귀속한다(배타적).
    분류 실패(폴백)를 가장 먼저 잡는다 — 성공으로 집계되어 있어 다른 코드가 없다.
    """
    out = pd.Series(pd.NA, index=q.index, dtype="string")
    fb = _fallback_mask(q).to_numpy(dtype=bool)
    out[fb] = "분류 실패(무관한 답)"
    code = q["fail_code"].fillna("").astype(str).to_numpy(dtype=object)
    for name, spec in PROBLEMS.items():
        if not spec.get("codes"):
            continue
        sel = np.isin(code, list(spec["codes"])) & out.isna().to_numpy()
        out[sel] = name
    return out


# ═══════════════════════════════ A. 대화 중단 귀속

def session_attribution(fu: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    """
    실패로 끝난 세션을, **세션을 끝낸 마지막 실패**의 원인으로 귀속한다.
    마지막 실패는 세션당 하나뿐이므로 합이 100%가 된다.

    여기에 '종료 위험 배수'(그 문제 뒤에 대화가 끝날 확률 ÷ 전체 평균)를
    곱해 빈도와 치명도를 함께 본다.
    """
    d = fu.sort_values(["session_id", "ts"]).copy()
    d["문제"] = tag_problem(d)
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    last_ts = d.groupby("session_id")["ts"].transform("max")
    d["_is_last"] = d["ts"].eq(last_ts)

    base_end = float(d["_is_last"].mean())
    risk = (d[d["문제"].notna()].groupby("문제")["_is_last"].mean() / base_end
            ).rename("종료위험배수")

    bad = sess[sess["session_outcome"].isin(["ABANDONED", "DEFLECTED"])]
    if bad.empty:
        return pd.DataFrame({"안내": ["실패로 끝난 세션이 없습니다"]})
    fail_turns = d[d["session_id"].isin(bad.index) & d["문제"].notna()]
    if fail_turns.empty:
        return pd.DataFrame({"안내": ["실패 원인이 귀속된 턴이 없습니다"]})
    last_fail = fail_turns.groupby("session_id").last()

    g = last_fail["문제"].value_counts().rename("세션수").to_frame()
    g["기여율"] = g["세션수"] / g["세션수"].sum()
    g = g.join(risk)
    g["겪은세션비율"] = (d[d["문제"].notna()].groupby("문제")["session_id"]
                          .nunique() / d["session_id"].nunique())
    g["처방"] = [PROBLEMS.get(i, {}).get("처방", "") for i in g.index]
    g["담당"] = [PROBLEMS.get(i, {}).get("담당", "") for i in g.index]
    g.attrs["실패세션수"] = int(len(bad))
    g.attrs["전체세션수"] = int(len(sess))
    return g.sort_values("기여율", ascending=False).round(4)


# ═══════════════════════════════ B. 노출-반응

def exposure_response(q: pd.DataFrame, end_date, grace_days: int = 30,
                      min_exposed: int = 50) -> pd.DataFrame:
    """
    문제를 겪은 사용자와 안 겪은 사용자의 이후 행동 차이.

    ★ 각 문제를 따로 계산하며 **합산하지 않는다**. 한 사용자가 여러 문제를
      겪으므로 합하면 중복된다. 또한 노출이 무작위가 아니어서(자료 부재를
      겪으려면 재무 질문을 해야 한다) 대부분 '상관'이다.
    """
    end_date = pd.Timestamp(end_date)
    d = q.copy()
    d["문제"] = tag_problem(d)
    d["date"] = d["ts"].dt.normalize()

    last = d.groupby("user_id")["date"].max()
    first = d.groupby("user_id")["date"].min()
    churn = ((end_date - last).dt.days > grace_days)
    sess_n = d.groupby("user_id")["session_id"].nunique()
    span = (last - first).dt.days

    rows = []
    for name in PROBLEMS:
        users = set(d.loc[d["문제"].eq(name), "user_id"])
        if len(users) < min_exposed:
            continue
        exp = churn.index.isin(users)
        if exp.sum() < min_exposed or (~exp).sum() < min_exposed:
            continue
        rows.append({
            "문제": name, "겪은 사용자": int(exp.sum()),
            "겪은쪽 이탈률": float(churn[exp].mean()),
            "안겪은쪽 이탈률": float(churn[~exp].mean()),
            "이탈률 차이": float(churn[exp].mean() - churn[~exp].mean()),
            "겪은쪽 세션수": float(sess_n[exp].mean()),
            "안겪은쪽 세션수": float(sess_n[~exp].mean()),
            "겪은쪽 이용기간(일)": float(span[exp].mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"안내": ["노출 표본이 부족합니다"]})
    out["근거"] = "상관"
    return out.sort_values("이탈률 차이", ascending=False).round(4)


def exposure_caution() -> str:
    return ("겪은 쪽이 더 오래·자주 쓴 사용자일 수 있어, 이탈률 차이가 음수로 "
            "나오기도 합니다. '세션수'와 '이용기간' 열이 그 선택 편향의 크기입니다. "
            "두 군의 활동량이 크게 다르면 이탈률 차이를 인과로 읽지 마십시오.")


# ═══════════════════════════════ C. 단독 해결 시뮬레이션

def single_fix(q: pd.DataFrame) -> pd.DataFrame:
    """
    각 문제를 **하나씩만** 해결했다고 가정했을 때의 무마찰 해결률 변화.
    순차 누적이 아니므로 문제 간 비교가 가능하다.

    ★ '고치면 그 질의가 성공한다'는 강한 가정 위의 상한이다.
    """
    from .sessions import session_outcomes
    from .turns import classify_followups

    def _rate(w):
        s = session_outcomes(classify_followups(w))
        return (float(s["session_outcome"].eq("RESOLVED").mean()),
                float(s["session_outcome"].isin(["RESOLVED", "RESOLVED_HARD"]).mean()))

    base_r, base_s = _rate(q)
    prob = tag_problem(q)
    rows = [{"시나리오": "현재", "치환건수": 0,
             "무마찰해결률": base_r, "세션해결률": base_s,
             "Δ무마찰": 0.0, "Δ세션": 0.0}]
    for name in PROBLEMS:
        sel = prob.eq(name).fillna(False)
        n = int(sel.sum())
        if n < 50:
            continue
        w = q.copy()
        w.loc[sel, "outcome"] = "success"
        w.loc[sel, "fail_code"] = pd.NA
        if name.startswith("분류 실패"):
            # 폴백은 이미 success 이므로, 라우팅이 고쳐지면 '제대로 된 성공'이 된다.
            # 세션 판정에서 폴백으로 제외되던 것이 성공으로 잡히게 툴명을 지운다.
            if "tool_called" in w.columns:
                w.loc[sel, "tool_called"] = "fixed_tool"
            if "intent_pred" in w.columns:
                w.loc[sel, "intent_pred"] = "FIXED"
        r, s_ = _rate(w)
        rows.append({"시나리오": f"{name} 해결", "치환건수": n,
                     "무마찰해결률": r, "세션해결률": s_,
                     "Δ무마찰": r - base_r, "Δ세션": s_ - base_s})
    out = pd.DataFrame(rows)
    return out.sort_values("Δ무마찰", ascending=False).round(4)


# ═══════════════════════════════ D. 문제 개수 누적

def problem_count_churn(q: pd.DataFrame, end_date, grace_days: int = 30,
                        min_n: int = 50) -> pd.DataFrame:
    """
    한 사용자가 겪은 서로 다른 문제의 개수와 이탈률.
    개수가 늘수록 나빠지면 누적 효과가 있다는 뜻이다.
    """
    end_date = pd.Timestamp(end_date)
    d = q.copy()
    d["문제"] = tag_problem(d)
    d["date"] = d["ts"].dt.normalize()
    cnt = (d[d["문제"].notna()].groupby("user_id")["문제"].nunique()
           .reindex(d["user_id"].unique(), fill_value=0).rename("문제수"))
    last = d.groupby("user_id")["date"].max()
    churn = ((end_date - last).dt.days > grace_days).rename("이탈")
    q_n = d.groupby("user_id").size().rename("질의수")
    j = pd.concat([cnt, churn, q_n], axis=1).dropna()
    j["구간"] = pd.cut(j["문제수"], [-1, 0, 1, 2, 3, 99],
                       labels=["0개", "1개", "2개", "3개", "4개+"])
    g = j.groupby("구간", observed=True).agg(
        사용자=("이탈", "size"), 이탈률=("이탈", "mean"),
        평균질의수=("질의수", "mean"))
    g = g[g["사용자"] >= min_n]
    g.attrs["주의"] = ("질의를 많이 한 사용자가 문제도 많이 겪습니다. "
                       "'평균질의수' 열이 그 교란의 크기이며, 개수가 늘수록 "
                       "질의수도 함께 늘면 순수 누적 효과로 볼 수 없습니다.")
    return g.round(4)


# ═══════════════════════════════ 통합 표

def combined(att: pd.DataFrame, exp: pd.DataFrame, fix: pd.DataFrame,
             matched_block: dict | None = None) -> pd.DataFrame:
    """네 결과를 한 표로. 근거 등급을 반드시 함께 표기한다."""
    if "안내" in att.columns:
        return att
    t = att.reset_index().rename(columns={"index": "문제"})
    keep = ["문제", "겪은세션비율", "종료위험배수", "기여율", "처방", "담당"]
    t = t[[c for c in keep if c in t.columns]]

    if "안내" not in exp.columns:
        t = t.merge(exp[["문제", "이탈률 차이"]], on="문제", how="left")
    if not fix.empty:
        f = fix.copy()
        f["문제"] = f["시나리오"].str.replace(" 해결", "", regex=False)
        t = t.merge(f[["문제", "Δ무마찰"]], on="문제", how="left")

    t["근거"] = "상관"
    if matched_block and matched_block.get("차이") is not None:
        t.loc[t["문제"].eq("차단 후 대안 없음"), "근거"] = "준-인과(매칭)"
    t["근거"] = np.where(t["Δ무마찰"].notna() if "Δ무마찰" in t.columns else False,
                         t["근거"] + " · 효과는 가정 위 상한", t["근거"])
    return t.round(4)


def headline(att: pd.DataFrame) -> str:
    """현업이 '그래서 몇 %냐'고 물을 때의 단일 문장. 정의를 좁혀서 답한다."""
    if "안내" in att.columns:
        return "실패로 끝난 세션이 없어 산출 불가"
    fixable = ["차단 후 대안 없음", "자료 부재", "기능 미호출·미구현",
               "인증 미처리", "대화 맥락 유실", "분류 실패(무관한 답)"]
    share = float(att.loc[att.index.isin(fixable), "기여율"].sum())
    model = float(att.loc[att.index.isin(["모델 응답 오류"]), "기여율"].sum())
    return (f"실패로 끝난 대화의 {share:.0%}가 자료·기능·인증·차단 후속 처리 등 "
            f"구조적으로 고칠 수 있는 원인으로 설명됩니다. "
            f"나머지 {model:.0%}는 모델 응답 자체의 문제입니다.")
