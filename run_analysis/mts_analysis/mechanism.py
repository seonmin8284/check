"""
메커니즘 분석 — 진단을 결론으로 잇는 층.

지금까지의 모듈은 "무엇이 실패했는가"를 셌다. 여기서는
"왜 그렇게 됐고, 고치면 어디까지 가는가"를 묻는다.

핵심 두 가지:
  ① 자기검열  — 차단이 그 질문만 막은 게 아니라 질문하는 행동 자체를 줄였는가
  ② 실질 성공 — 'HTML 이 렌더됐다'는 시스템 성공이지 사용자 성공이 아니다
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .schema import FNGUIDE_INTENTS
from .turns import BAD_KINDS

JUDGMENT_LEVELS = ("P2", "P3")     # 해석·전망·매매판단 = 억눌리기 쉬운 질의


# =================================================== ① 자기검열 지수

def self_censorship(q: pd.DataFrame, min_each: int = 3) -> dict:
    """
    차단 경험 전후로 개인의 판단성 질의(P2/P3) 비중이 어떻게 변하는가.

    차단은 그 질문 하나를 막는다. 그런데 사용자가 '못 하는 챗봇'으로 학습하면
    비슷한 질문 전체를 줄인다. 그러면 수요가 사라진 게 아니라 **억눌린** 것이고,
    질의량 기준 우선순위에서는 영영 보이지 않게 된다.

    개인 내 전후 비교(paired)라 사용자 이질성은 자동 통제된다.
    미차단 사용자를 대조군으로 두어 시간 추세도 함께 뺀다.
    """
    d = q.sort_values(["user_id", "ts"]).copy()
    d["_judge"] = d["f4_compliance"].isin(JUDGMENT_LEVELS).fillna(False)

    blk = (d[d["outcome"].eq("blocked")].groupby("user_id")["ts"].min()
             .rename("t_block"))
    if blk.empty:
        return {"안내": "차단 경험 사용자가 없습니다"}
    d = d.join(blk, on="user_id")

    treat = d[d["t_block"].notna()].copy()
    treat["_after"] = treat["ts"] > treat["t_block"]
    g = (treat.groupby(["user_id", "_after"])["_judge"]
              .agg(["size", "mean"]).reset_index())
    piv = g.pivot(index="user_id", columns="_after", values="mean")
    cnt = g.pivot(index="user_id", columns="_after", values="size")
    if piv.shape[1] < 2:
        return {"안내": "차단 전후 양쪽 관측이 있는 사용자가 없습니다"}
    piv.columns, cnt.columns = ["before", "after"], ["n_before", "n_after"]
    paired = piv.join(cnt).dropna()
    paired = paired[(paired["n_before"] >= min_each) & (paired["n_after"] >= min_each)]
    if len(paired) < 20:
        return {"안내": f"전후 각 {min_each}건 이상인 사용자가 부족합니다"}

    paired["변화"] = paired["after"] - paired["before"]
    t, p = stats.ttest_rel(paired["after"], paired["before"])

    # 대조군: 미차단 사용자의 같은 기간 전후 변화 (시간 추세 제거)
    ctrl_delta = np.nan
    ctrl = d[d["t_block"].isna()]
    if len(ctrl):
        med_t = treat["t_block"].median()
        c = ctrl.assign(_after=ctrl["ts"] > med_t)
        cg = c.groupby(["user_id", "_after"])["_judge"].agg(["size", "mean"]).reset_index()
        cp = cg.pivot(index="user_id", columns="_after", values="mean")
        cc = cg.pivot(index="user_id", columns="_after", values="size")
        if cp.shape[1] == 2:
            cp.columns, cc.columns = ["before", "after"], ["n_before", "n_after"]
            cpair = cp.join(cc).dropna()
            cpair = cpair[(cpair["n_before"] >= min_each) & (cpair["n_after"] >= min_each)]
            if len(cpair) >= 20:
                ctrl_delta = float((cpair["after"] - cpair["before"]).mean())

    raw = float(paired["변화"].mean())
    adj = raw - ctrl_delta if ctrl_delta == ctrl_delta else np.nan
    shrink = float((paired["변화"] < -0.05).mean())

    if adj == adj:
        verdict = ("억눌린 수요 — 차단이 해당 질문만이 아니라 판단성 질의 전반을 "
                   "줄였습니다. 질의량 기준 우선순위로는 보이지 않습니다."
                   if adj < -0.03 else
                   "자기검열 신호 약함 — 차단 이후에도 판단성 질의 비중이 유지됩니다")
    else:
        verdict = "대조군 확보 실패 — 원시 변화만 참고하십시오"

    return {"대상사용자": int(len(paired)),
            "차단전_판단성비중": round(float(paired["before"].mean()), 4),
            "차단후_판단성비중": round(float(paired["after"].mean()), 4),
            "원시변화": round(raw, 4),
            "대조군변화": round(ctrl_delta, 4) if ctrl_delta == ctrl_delta else np.nan,
            "보정변화": round(adj, 4) if adj == adj else np.nan,
            "5%p이상 감소한 사용자 비율": round(shrink, 4),
            "paired t": round(float(t), 3), "p값": float(p),
            "판정": verdict}


# =================================================== ② 실질 성공 신호

def effective_success(fu: pd.DataFrame, chars_per_sec: float = 12.0,
                      min_read_sec: float = 3.0,
                      skim_ratio: float = 0.4) -> pd.DataFrame:
    """
    렌더 성공을 사용자 관점으로 재분류한다.

      EFFECTIVE     — 충분히 머문 뒤 다른 의도로 진행 (실질 성공)
      SHALLOW_FAIL  — 곧바로 재질문·형식 재요청 (실질 실패)
      SKIMMED       — 읽는 데 필요한 시간에 못 미치고 넘어감
      TERMINAL      — 세션 종료 (즉답 만족일 수도, 포기일 수도 — 양가)

    읽기 시간은 response_len 으로 추정한다(한국어 약 12자/초).
    TERMINAL 은 판정 불가이므로 실질 성공률을 **구간**으로 낸다.
    """
    d = fu.sort_values(["session_id", "ts"]).copy()
    g = d.groupby("session_id")
    d["_next_ts"] = g["ts"].shift(-1)
    d["_next_kind"] = g["turn_kind"].shift(-1)
    d["_gap"] = (d["_next_ts"] - d["ts"]).dt.total_seconds()

    if "response_len" in d.columns and d["response_len"].notna().any():
        need = (d["response_len"].fillna(0) / chars_per_sec).clip(lower=min_read_sec)
    else:
        need = pd.Series(min_read_sec, index=d.index)
    d["_need_sec"] = need

    ok = d["outcome"].eq("success").fillna(False).to_numpy(dtype=bool)
    terminal = d["_next_ts"].isna().to_numpy(dtype=bool)
    bad_next = d["_next_kind"].isin(BAD_KINDS).fillna(False).to_numpy(dtype=bool)
    skimmed = (d["_gap"] < d["_need_sec"] * skim_ratio).fillna(False).to_numpy(dtype=bool)

    d["eff_kind"] = np.select(
        [~ok, terminal, bad_next, skimmed],
        ["NOT_SUCCESS", "TERMINAL", "SHALLOW_FAIL", "SKIMMED"],
        default="EFFECTIVE")
    return d


def effective_success_summary(eff: pd.DataFrame) -> dict:
    d = eff[eff["eff_kind"].ne("NOT_SUCCESS")]
    if d.empty:
        return {"안내": "성공 건이 없습니다"}
    n_all = len(eff)
    vc = d["eff_kind"].value_counts()
    sys_rate = len(d) / n_all
    lo = float(vc.get("EFFECTIVE", 0)) / n_all
    hi = float(vc.get("EFFECTIVE", 0) + vc.get("TERMINAL", 0)) / n_all
    return {"시스템 성공률": round(sys_rate, 4),
            "실질 성공률 하한": round(lo, 4),
            "실질 성공률 상한": round(hi, 4),
            "구간폭(TERMINAL 비중)": round(hi - lo, 4),
            "분포": (vc / n_all).round(4).to_dict(),
            "해석": ("시스템 성공률과 실질 하한의 간격이 곧 '렌더는 됐지만 "
                     "사용자에게 닿지 않은' 응답의 크기입니다. "
                     "TERMINAL 은 즉답 만족과 포기가 섞여 있어 구간으로 둡니다.")}


def effective_by_intent(eff: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    d = eff.copy()
    d["_sys_ok"] = d["eff_kind"].ne("NOT_SUCCESS")
    d["_eff_ok"] = d["eff_kind"].eq("EFFECTIVE")
    d["_shallow"] = d["eff_kind"].isin(["SHALLOW_FAIL", "SKIMMED"])
    g = d.groupby("l2_intent").agg(
        n=("_sys_ok", "size"), 시스템성공률=("_sys_ok", "mean"),
        실질성공률=("_eff_ok", "mean"), 헛성공률=("_shallow", "mean"))
    g = g[g["n"] >= min_n]
    g["괴리"] = g["시스템성공률"] - g["실질성공률"]
    return g.sort_values("괴리", ascending=False).round(3)


# =================================================== ③ 실패 직후 전이

def transition_by_outcome(fu: pd.DataFrame, level: str = "l1_stage",
                          min_n: int = 30, drop_self: bool = True) -> pd.DataFrame:
    """
    직전 턴의 성패에 따라 **다른 단계로의 이동**이 어떻게 달라지는가.

    EVALUATE 실패 후 SERVICE 로 넘어갈 확률이 성공 후보다 높다면,
    '투자 목적으로 왔다가 업무 목적으로 이동'의 세션 내 미시 증거다.

    ★ 두 가지를 반드시 걷어낸다.
      · 같은 단계로의 전이(대각선) — 실패하면 재질문하므로 당연히 높다.
        그건 이미 REPEAT 으로 측정 중이라 여기서 보면 중복이고 신호를 덮는다.
      · REPEAT/FORMAT 턴 — 복구 시도이지 '이동'이 아니다.
    """
    d = fu.sort_values(["session_id", "ts"]).copy()
    g = d.groupby("session_id")
    d["prev_lv"] = g[level].shift(1)
    d["prev_ok"] = g["outcome"].shift(1).eq("success")
    p = d.dropna(subset=["prev_lv"])
    p = p[~p["turn_kind"].isin(BAD_KINDS).fillna(False)]
    if drop_self:
        p = p[p[level] != p["prev_lv"]]
    if p.empty:
        return pd.DataFrame({"안내": ["이동 전이 관측 없음"]})

    rows = []
    for (frm, ok), gg in p.groupby(["prev_lv", "prev_ok"]):
        if len(gg) < min_n:
            continue
        for to, share in gg[level].value_counts(normalize=True).items():
            rows.append({"from": frm, "직전성공": bool(ok), "to": to,
                         "전이확률": share, "n": len(gg)})
    if not rows:
        return pd.DataFrame({"안내": ["표본 부족"]})
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["from", "to"], columns="직전성공",
                        values="전이확률")
    if piv.shape[1] < 2:
        return pd.DataFrame({"안내": ["성공/실패 양쪽 관측 필요"]})
    piv.columns = ["실패후", "성공후"]
    piv = piv.dropna()
    piv["차이"] = piv["실패후"] - piv["성공후"]
    return piv.sort_values("차이", ascending=False).round(3)


# =================================================== ④ 의존도 기반 반사실

def dependency_shock(q: pd.DataFrame, outage_date, end_date,
                     dep_intents: tuple[str, ...] = tuple(FNGUIDE_INTENTS),
                     pre_days: int = 42, post_days: int = 42,
                     min_pre: int = 5) -> dict:
    """
    소스 중단을 자연실험으로 쓴다.

    처치 강도 = 중단 전 그 사용자의 '의존 의도' 질의 비중 (연속 변수).
    결과      = 중단 후 총 질의량 변화 / 이탈 / 업무(SERVICE) 비중 변화

    실패율 DiD 가 '무엇이 깨졌나'라면, 이건 '사용자에게 무슨 일이 일어났나'다.
    회복 상한(C3)의 실증 근거가 여기서 나온다.
    """
    outage, end = pd.Timestamp(outage_date), pd.Timestamp(end_date)
    d = q.copy()
    pre = d[(d["ts"] >= outage - pd.Timedelta(days=pre_days)) & (d["ts"] < outage)]
    post = d[(d["ts"] >= outage) & (d["ts"] < outage + pd.Timedelta(days=post_days))]
    if pre.empty or post.empty:
        return {"안내": "중단일 전후 구간에 데이터가 부족합니다"}

    pre_n = pre.groupby("user_id").size().rename("pre_n")
    dep = (pre.assign(_d=pre["l2_intent"].isin(dep_intents))
              .groupby("user_id")["_d"].mean().rename("의존도"))
    svc_pre = (pre.assign(_s=pre["l1_stage"].eq("SERVICE"))
                  .groupby("user_id")["_s"].mean().rename("svc_pre"))
    post_n = post.groupby("user_id").size().rename("post_n")
    svc_post = (post.assign(_s=post["l1_stage"].eq("SERVICE"))
                    .groupby("user_id")["_s"].mean().rename("svc_post"))

    u = pd.concat([pre_n, dep, svc_pre], axis=1).dropna()
    u = u[u["pre_n"] >= min_pre]
    if len(u) < 50:
        return {"안내": f"중단 전 {min_pre}건 이상 사용자가 부족합니다"}
    u = u.join(post_n).join(svc_post)
    u["post_n"] = u["post_n"].fillna(0)
    u["이탈"] = u["post_n"].eq(0)
    u["질의량변화율"] = (u["post_n"] - u["pre_n"]) / u["pre_n"]
    u["업무비중변화"] = u["svc_post"] - u["svc_pre"]

    # 의존도 5분위별 결과
    u["분위"] = pd.qcut(u["의존도"].rank(method="first"), 5,
                        labels=[f"Q{i}" for i in range(1, 6)])
    tab = u.groupby("분위", observed=True).agg(
        n=("의존도", "size"), 평균의존도=("의존도", "mean"),
        이탈률=("이탈", "mean"), 질의량변화율=("질의량변화율", "mean"),
        업무비중변화=("업무비중변화", "mean")).round(4)

    # 강도-반응: 의존도 → 결과 상관 및 Q5-Q1 격차
    res = {}
    for col in ("이탈", "질의량변화율", "업무비중변화"):
        v = u[col].astype(float)
        m = v.notna()
        r = float(np.corrcoef(u.loc[m, "의존도"], v[m])[0, 1]) if m.sum() > 10 else np.nan
        gap = (float(tab.loc["Q5", col if col != "이탈" else "이탈률"])
               - float(tab.loc["Q1", col if col != "이탈" else "이탈률"]))
        res[col] = {"상관": round(r, 3), "Q5-Q1": round(gap, 4)}

    verdict = ("의존도가 높을수록 이탈·축소가 뚜렷 — 정보 기능 상실이 "
               "사용자 이탈로 직결됩니다. 복구의 기대 효과가 큽니다."
               if res["이탈"]["Q5-Q1"] > 0.03 else
               "의존도와 이탈의 관계가 약함 — 정보 기능 상실이 "
               "사용자를 떠나게 하지는 않았습니다. 복구 우선순위 재검토 필요.")
    return {"분위표": tab, "강도반응": res, "대상사용자": int(len(u)),
            "판정": verdict}


# =================================================== ⑤ 회복 상한

def recovery_ceiling(q: pd.DataFrame, code_groups: dict[str, tuple[str, ...]] | None = None
                     ) -> pd.DataFrame:
    """
    실패코드 그룹을 순차적으로 '성공'으로 치환했을 때 North Star 가 어디까지 가는가.

    ★ 강한 가정 위의 시뮬레이션입니다. 고치면 그 질의가 성공한다고 가정할 뿐,
      성공했을 때 사용자가 남는지는 별개입니다. 상한의 상한으로만 읽으십시오.
      그래도 '6개월 써서 몇 %p 인가'를 묻는 자리에는 이 숫자가 필요합니다.
    """
    from .sessions import SUCCESS_OUTCOMES, session_outcomes
    from .turns import classify_followups

    if code_groups is None:
        code_groups = {
            "현재": (),
            "+데이터(D1·D2·D3)": ("D1", "D2", "D3"),
            "+툴(T1·T2)": ("D1", "D2", "D3", "T1", "T2"),
            "+대화·인증(S1·X1·A1)": ("D1", "D2", "D3", "T1", "T2", "S1", "X1", "A1"),
            "+차단회수(C3)": ("D1", "D2", "D3", "T1", "T2", "S1", "X1", "A1", "C3"),
        }
    rows = []
    for label, codes in code_groups.items():
        w = q.copy()
        if codes:
            fix = w["fail_code"].isin(codes).fillna(False)
            w.loc[fix, "outcome"] = "success"
            w.loc[fix, "fail_code"] = pd.NA
        fu = classify_followups(w)
        sess = session_outcomes(fu)
        n_user = sess["user_id"].nunique()
        users_ok = (sess[sess["session_outcome"].isin(SUCCESS_OUTCOMES)]["user_id"]
                    .nunique())
        # 사용자별 무마찰 달성 세션 보유 여부 — 천장 효과가 없는 엄격 지표
        clean = sess[sess["session_outcome"].eq("RESOLVED")]["user_id"].nunique()
        rows.append({
            "시나리오": label,
            "치환건수": int(w["outcome"].eq("success").sum() - q["outcome"].eq("success").sum()),
            "턴성공률": float(w["outcome"].eq("success").mean()),
            "세션성공률": float(sess["session_outcome"].isin(SUCCESS_OUTCOMES).mean()),
            "무마찰세션률": float(sess["session_outcome"].eq("RESOLVED").mean()),
            "NorthStar": users_ok / n_user,
            "무마찰달성사용자": clean / n_user,
        })
    out = pd.DataFrame(rows)
    for c in ("NorthStar", "무마찰세션률", "세션성공률"):
        out[f"Δ{c}"] = out[c] - out.loc[0, c]
    # 천장 효과 경고: North Star 가 이미 95% 이상이면 추적 지표로 부적합
    out.attrs["천장경고"] = bool(out.loc[0, "NorthStar"] >= 0.95)
    return out.round(4)


# =================================================== ⑥ 세그먼트 교차

def segment_crosstab(q: pd.DataFrame, seg: dict, fu: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    핵심 지표를 세그먼트별로 재산출한다.
    의도 축에서는 심슨의 역설을 막았지만 사용자 축에서는 아직 안 막았다.
    """
    if "_labels" not in seg:
        return pd.DataFrame({"안내": ["세그먼트 도출 실패"]})
    lab = seg["_labels"].rename("seg")
    d = q.join(lab, on="user_id").dropna(subset=["seg"])
    if d.empty:
        return pd.DataFrame({"안내": ["세그먼트 매칭 실패"]})
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    d["_blk"] = d["outcome"].eq("blocked").fillna(False)
    d["_judge"] = d["f4_compliance"].isin(JUDGMENT_LEVELS).fillna(False)

    g = d.groupby("seg").agg(
        사용자=("user_id", "nunique"), 질의=("_ok", "size"),
        성공률=("_ok", "mean"), 차단률=("_blk", "mean"),
        판단성질의비중=("_judge", "mean"),
        의도다양성=("l2_intent", "nunique"))
    g["주의도"] = d.groupby("seg")["l1_stage"].agg(
        lambda x: x.value_counts().index[0])
    g.index = [f"S{int(i)}:{seg['_names'][int(i)]}" for i in g.index]
    return g.round(3)


# =================================================== ⑦ 비용-편익

def cost_benefit(q: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """
    의도별 운영 비용(응답시간·툴 호출)과 편익(성공 질의량)의 사분면.

    고비용·저편익 의도는 개선 대상이 아니라 **범위 축소 후보**일 수 있다.
    """
    d = q.copy()
    d["_ok"] = d["outcome"].eq("success").fillna(False)
    d["_ntool"] = (d["tool_called"].fillna("").astype(str).str.count(r"\|") + 1
                   ).where(d["tool_called"].notna(), 0) if "tool_called" in d else 0
    g = d.groupby("l2_intent").agg(
        질의량=("_ok", "size"), 성공률=("_ok", "mean"),
        평균응답ms=("latency_ms", "mean"), p95응답ms=("latency_ms", lambda x: x.quantile(.95)))
    if "tool_called" in d.columns:
        g["평균툴호출"] = d.groupby("l2_intent")["_ntool"].mean()
    g = g[g["질의량"] >= min_n]
    if g.empty:
        return pd.DataFrame({"안내": ["표본 부족"]})

    g["총비용"] = g["질의량"] * g["평균응답ms"] / 1000 / 3600     # 누적 응답시간(시간)
    g["편익"] = g["질의량"] * g["성공률"]
    c_med, b_med = g["총비용"].median(), g["편익"].median()
    g["사분면"] = np.select(
        [(g["총비용"] >= c_med) & (g["편익"] >= b_med),
         (g["총비용"] >= c_med) & (g["편익"] < b_med),
         (g["총비용"] < c_med) & (g["편익"] >= b_med)],
        ["고비용·고편익 (효율화)", "★고비용·저편익 (축소 검토)",
         "저비용·고편익 (유지)"], default="저비용·저편익")
    return g.sort_values(["사분면", "총비용"], ascending=[True, False]).round(3)


# =================================================== ⑧ 대체 가능성

def substitutability(q: pd.DataFrame) -> dict:
    """
    챗봇이 기존 화면을 '대체'하는가 '보완'하는가.

    같은 답을 두어 번의 탭으로 얻을 수 있다면 챗봇의 고유 가치는 낮다.
    navigation 응답 비중과 단순 조회 비중이 높을수록 대체재 성격이 강하다.
    """
    d = q.copy()
    n = len(d)
    nav_resp = float(d["f5_response"].eq("navigation").mean()) if "f5_response" in d else np.nan
    nav_intent = float(d["l2_intent"].eq("EXEC.nav").mean())
    proc = float(d["f5_response"].eq("procedure").mean()) if "f5_response" in d else np.nan
    fact = float(d["f5_response"].eq("fact").mean()) if "f5_response" in d else np.nan
    simple = np.nan
    if "response_len" in d.columns and d["response_len"].notna().any():
        thr = d["response_len"].quantile(.33)
        simple = float((d["response_len"] <= thr).mean())

    sub_score = np.nansum([nav_resp, nav_intent, proc]) / 3
    verdict = ("대체재 성격 강함 — 화면 탐색·절차 안내가 주 용도입니다. "
               "챗봇의 가치가 '더 빠른 길찾기'에 머물러 있습니다."
               if sub_score > .30 else
               "보완재 성격 — 화면으로 대체하기 어려운 질의가 다수입니다")
    return {"navigation 응답 비중": round(nav_resp, 4),
            "EXEC.nav 의도 비중": round(nav_intent, 4),
            "procedure 응답 비중": round(proc, 4),
            "fact 응답 비중": round(fact, 4),
            "짧은 응답 비중(하위 33%)": round(simple, 4) if simple == simple else np.nan,
            "대체성 점수": round(float(sub_score), 4),
            "판정": verdict}
