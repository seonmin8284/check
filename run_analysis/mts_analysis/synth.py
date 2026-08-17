"""
검증용 합성 데이터 생성기.

실데이터를 붙이기 전에 파이프라인이 '알려진 정답'을 복원하는지 확인하는 용도입니다.
아래 TRUE_* 상수가 심어놓은 참값이며, run_analysis.py 가 추정치와 대조합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    ALL_INTENTS, INTENT_TO_STAGE, FNGUIDE_INTENTS, CONTROL_INTENTS,
    EVALUATE_LOOKUP_INTENTS,
)

# ------------------------------------------------ 심어놓은 참값 (검증 대조용)
TRUE_DID_EFFECT = 0.45        # FnGuide 중단이 처치 의도군 실패율에 미친 효과
TRUE_BLOCK_LOG_HR = np.log(1.8)   # 차단 경험의 이탈 위험비
TRUE_CHATBOT_ORDER_LIFT = 0.06    # 챗봇 조회의 주문확률 증분
TRUE_WITHIN_SHARE = 0.40      # 투자의도 비중 하락 중 '행동변화' 기여 비중(대략)
TRUE_REPEAT_P_FAIL = 0.38     # 직전 턴 실패 시 재질문 확률
TRUE_REPEAT_P_OK = 0.04       # 직전 턴 성공 시 재질문 확률
TRUE_OTH_RATE = 0.14          # 의도분류 실패 → OTH → 폴백 함수 강제 호출 비율

OUTAGE_DATE = pd.Timestamp("2026-04-15")   # FnGuide 중단일
PROTECTOR_DATE = pd.Timestamp("2026-03-01")  # Protector 도입일
START = pd.Timestamp("2026-02-01")
END = pd.Timestamp("2026-06-30")

# 시간대별 의도 가중 (일중 프로파일 심기)
_HOUR_PROFILE = {
    "LEARN": {8: .5, 9: .3, 12: 1.0, 15: 1.0, 20: 3.0, 22: 3.0},
    "DISCOVER": {8: 3.0, 9: 2.5, 12: 1.0, 15: .8, 20: 1.2, 22: 1.0},
    "EVALUATE": {8: 2.0, 9: 3.0, 12: 1.5, 15: 1.0, 20: 1.0, 22: .8},
    "EXECUTE": {8: 1.5, 9: 3.5, 12: 1.5, 15: 1.2, 20: .3, 22: .2},
    "MONITOR": {8: .5, 9: 1.0, 12: 1.0, 15: 2.5, 20: 2.0, 22: 1.5},
    "SETTLE": {8: .5, 9: .5, 12: 1.0, 15: 2.0, 20: 1.5, 22: 1.2},
    "SERVICE": {8: 1.0, 9: .8, 12: 1.2, 15: 1.2, 20: 2.0, 22: 2.5},
    "RECOVER": {8: 1.0, 9: 1.2, 12: 1.0, 15: 1.0, 20: 1.0, 22: 1.0},
}


def _hour_weight(stage: str, hour: int) -> float:
    prof = _HOUR_PROFILE[stage]
    keys = sorted(prof)
    k = min(keys, key=lambda x: abs(x - hour))
    return prof[k]


def generate(n_users: int = 1200, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    tickers = [f"{i:06d}" for i in rng.choice(range(5000, 9000), 120, replace=False)]

    # --- 사용자 유형: 투자형 / 업무형 / 혼합형
    utype = rng.choice(["invest", "service", "mixed"], n_users, p=[.45, .30, .25])
    users = pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(n_users)],
                          "utype": utype})
    users["join_date"] = rng.choice(dates[:100], n_users)
    # 기본 이탈 해저드
    users["base_hazard"] = np.where(users.utype == "invest", .020,
                            np.where(users.utype == "service", .010, .014))

    rows, order_rows, view_rows, appsess_rows = [], [], [], []

    for _, u in users.iterrows():
        uid, ut = u.user_id, u.utype
        alive_from = u.join_date

        # 1) 활동일 후보를 먼저 뽑는다
        act_p = {"invest": .28, "service": .12, "mixed": .20}[ut]
        full_span = pd.date_range(alive_from, END, freq="D")
        cand_days = full_span[rng.random(len(full_span)) < act_p]

        # 2) 차단 시점 = Protector 도입 이후 첫 활동일 중 P3 질의가 나온 날.
        #    관측되는 '첫 차단 질의'와 위험요인이 동일해야 생존분석이 검증 가능하다.
        p_block_day = {"invest": .22, "mixed": .10, "service": .015}[ut]
        block_date = None
        for dte in cand_days:
            if dte >= PROTECTOR_DATE and rng.random() < p_block_day:
                block_date = dte
                break

        # 3) 이탈일 (차단 이후 해저드 상승 = 심어놓은 효과)
        churn_date, d = None, alive_from
        while d <= END:
            h = u.base_hazard
            if block_date is not None and d >= block_date:
                h *= np.exp(TRUE_BLOCK_LOG_HR)
            if rng.random() < h:
                churn_date = d
                break
            d += pd.Timedelta(days=1)
        last_day = churn_date if churn_date is not None else END
        if block_date is not None and block_date > last_day:
            block_date = None

        act_days = cand_days[cand_days <= last_day]
        # 이탈일에 마지막 활동이 찍히게 해 '관측 최종활동일 = 잠재 이탈일'로 맞춘다.
        # (실데이터에서는 이 둘이 어긋나며, 그 측정오차가 HR 을 감쇠시킨다 — README 참조)
        if churn_date is not None and churn_date not in set(act_days):
            act_days = pd.DatetimeIndex(sorted(set(act_days) | {churn_date}))

        for dte in act_days:
            appsess_rows.append({"user_id": uid, "date": dte})
            n_q = int(rng.integers(1, 5))
            sid = f"{uid}-{dte:%Y%m%d}"
            hour = int(rng.choice([8, 9, 10, 11, 13, 14, 15, 16, 20, 21, 22],
                                  p=[.08, .18, .12, .08, .08, .08, .10, .08, .08, .06, .06]))
            qts = dte + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 50)))
            prev_ok, prev_intent, prev_tgt = None, None, None
            for qi in range(n_q):
                if qi > 0:                       # 세션 내 턴은 1~6분 간격
                    qts = qts + pd.Timedelta(minutes=int(rng.integers(1, 7)))
                # 유형 × 시간대 기반 stage 추첨
                if ut == "invest":
                    base = {"LEARN": .06, "DISCOVER": .18, "EVALUATE": .34,
                            "EXECUTE": .16, "MONITOR": .12, "SETTLE": .05,
                            "SERVICE": .06, "RECOVER": .03}
                elif ut == "service":
                    base = {"LEARN": .12, "DISCOVER": .05, "EVALUATE": .10,
                            "EXECUTE": .12, "MONITOR": .08, "SETTLE": .13,
                            "SERVICE": .36, "RECOVER": .04}
                else:
                    base = {"LEARN": .10, "DISCOVER": .12, "EVALUATE": .22,
                            "EXECUTE": .14, "MONITOR": .12, "SETTLE": .09,
                            "SERVICE": .18, "RECOVER": .03}
                # 심어놓은 '행동변화': 중단 이후 투자형의 EVALUATE 비중 자체가 하락
                base = dict(base)
                if dte >= OUTAGE_DATE and ut in ("invest", "mixed"):
                    shift = 0.10 * TRUE_WITHIN_SHARE / 0.40
                    base["EVALUATE"] = max(.02, base["EVALUATE"] - shift)
                    base["SERVICE"] += shift
                w = np.array([base[s] * _hour_weight(s, hour) for s in base])
                stage = list(base)[int(rng.choice(len(w), p=w / w.sum()))]

                intents = [i for i in ALL_INTENTS if INTENT_TO_STAGE[i] == stage]
                intent = str(rng.choice(intents))

                # 관측되는 '첫 차단'이 block_date 와 일치하도록 정합화:
                # block_date 이전에는 P3 의도가 등장하지 않게 하고,
                # block_date 당일에는 반드시 하나 등장시킨다.
                P3_SET = {"EVAL.verdict", "DISC.recommend_open",
                          "MON.rebalance", "MON.loss_reaction"}
                if block_date is not None and dte == block_date and qi == 0:
                    intent = str(rng.choice(sorted(P3_SET)))
                    stage = INTENT_TO_STAGE[intent]
                elif intent in P3_SET and (block_date is None or dte < block_date):
                    alt = [i for i in intents if i not in P3_SET]
                    intent = str(rng.choice(alt)) if alt else intent

                # ★ 심어놓은 효과: 직전 턴이 실패·차단이면 같은 질문을 다시 하거나
                #    형식을 바꿔 재요청할 확률이 크게 오른다.
                repeat_mode = None
                if qi > 0 and prev_intent is not None:
                    p_rep = TRUE_REPEAT_P_FAIL if not prev_ok else TRUE_REPEAT_P_OK
                    p_fmt = .18 if not prev_ok else .05
                    r = rng.random()
                    if r < p_rep:
                        repeat_mode, intent = "REPEAT", prev_intent
                    elif r < p_rep + p_fmt:
                        repeat_mode, intent = "FORMAT", prev_intent
                    stage = INTENT_TO_STAGE[intent]

                # Facet
                f4 = "P0"
                if intent in ("EVAL.verdict", "DISC.recommend_open",
                              "MON.rebalance", "MON.loss_reaction"):
                    f4 = "P3"
                elif intent in ("EVAL.outlook", "EVAL.causal", "EVAL.interpret"):
                    f4 = "P2"
                elif stage in ("EVALUATE", "DISCOVER", "MONITOR", "SETTLE"):
                    f4 = "P1"
                f3 = "account_required" if stage == "MONITOR" or intent in (
                    "EXEC.order_status", "EXEC.margin", "SETL.settlement",
                    "SETL.record") else "none"
                f2 = "future" if intent in ("EVAL.outlook", "DISC.ipo_pipeline") else "present"
                f6 = "followup" if qi > 0 and rng.random() < .3 else "new"

                # 대상 종목
                tgt = []
                if repeat_mode == "REPEAT" and prev_tgt:
                    tgt = list(prev_tgt)
                elif stage in ("EVALUATE",) and rng.random() < .85:
                    tgt = [str(rng.choice(tickers))]
                    if intent == "EVAL.compare":
                        tgt.append(str(rng.choice(tickers)))

                # --- 결과 결정 (여기에 DiD 효과를 심음)
                base_fail = .08
                if intent == "EXEC.nav":
                    base_fail = .55                 # 딥링크 카탈로그 부재
                if intent in ("DISC.theme", "DISC.related"):
                    base_fail = .25
                if intent == "REC.followup" or f6 == "followup":
                    base_fail += .07                # 컨텍스트 상속 실패
                if intent in FNGUIDE_INTENTS and dte >= OUTAGE_DATE:
                    base_fail += TRUE_DID_EFFECT    # ★ 심어놓은 DiD 효과
                if f3 == "account_required" and rng.random() < .10:
                    base_fail += .30                # 인증 미충족

                if f4 == "P3" and dte >= PROTECTOR_DATE:
                    outcome, answerable = "blocked", "blocked"
                elif rng.random() < min(base_fail, .95):
                    outcome = "fail"
                    if intent in FNGUIDE_INTENTS and dte >= OUTAGE_DATE:
                        answerable = "no_source"
                    elif intent == "EXEC.nav":
                        answerable = "no_source"
                    elif intent in ("DISC.theme", "DISC.related"):
                        answerable = "no_tool"
                    elif f3 == "account_required":
                        answerable = "no_auth"
                    elif f6 == "followup":
                        answerable = "no_slot"
                    else:
                        answerable = "yes"          # → M1 잔차
                else:
                    outcome, answerable = "success", "yes"

                # --- 질문/응답 원문 (관련성·구조 분석용)
                _NOUN = {"EVAL": "영업이익", "DISC": "종목", "EXEC": "주문",
                         "MON": "잔고", "SETL": "배당", "SVC": "계좌",
                         "LEARN": "용어", "REC": "문의", "OOS": "기타",
                         "RISK": "상담"}
                _tk = tgt[0] if tgt else str(rng.choice(tickers))
                _noun = _NOUN.get(intent.split(".")[0], "정보")
                qtext_base = f"{_tk} {_noun} 알려줘"

                # ★ 심어놓은 구조적 결함: 분류 실패 → OTH → 폴백 함수 강제 호출.
                #   응답은 나오지만 질문과 무관하다. 그런데 렌더는 되므로 성공으로 집계된다.
                is_oth = rng.random() < TRUE_OTH_RATE and outcome != "blocked"
                if is_oth:
                    outcome = "success"
                    atext = ("<div class=\"news\"><p>최근 증시 동향 요약입니다. "
                             "코스피는 전일 대비 상승 마감했으며 외국인 순매수가 "
                             "이어졌습니다.</p></div>")
                elif outcome == "success":
                    _rows = "".join(
                        f"<tr><td>{y}</td><td>{int(rng.integers(100, 9999)):,}</td></tr>"
                        for y in range(2020, 2025))
                    _lead = ("<p>요청하신 내용을 정리했습니다.</p>"
                             if rng.random() < .35 else "")
                    atext = (f"<div>{_lead}<p>{_tk} {_noun} 조회 결과</p>"
                             f"<table>{_rows}</table></div>")
                else:
                    atext = "<span>요청을 처리하지 못했습니다.</span>"

                # --- 선택 컬럼 (운영 분류·툴·응답 품질)
                CONFUSE = {"EVAL.outlook": "EVAL.verdict",
                           "EVAL.verdict": "EVAL.outlook",
                           "DISC.recommend_open": "EVAL.verdict",
                           "EXEC.nav": "SVC.app_setting",
                           "LEARN.process": "EXEC.order_howto",
                           "EVAL.causal": "EVAL.interpret"}
                if rng.random() < (.30 if intent in CONFUSE else .04):
                    intent_pred = CONFUSE.get(intent, str(rng.choice(intents)))
                else:
                    intent_pred = intent

                needs_tool = stage not in ("LEARN", "RECOVER")
                # 툴 없이 답하는 비율 = 환각 위험군
                p_notool = .28 if stage in ("DISCOVER", "EVALUATE") else .10
                tool_called = (f"tool_{intent.split('.')[1]}"
                               if (needs_tool and outcome == "success"
                                   and rng.random() > p_notool) else None)
                cited = bool(tool_called) and rng.random() < .85
                rlen = float(rng.gamma(4, 120) * (1.6 if stage == "EVALUATE" else 1.0))
                csat = (float(np.clip(rng.normal(4.2 if outcome == "success" else 2.4, .9),
                                      1, 5)) if rng.random() < .08 else np.nan)
                # 형식 재요청은 후속 턴에서만 발생
                if repeat_mode == "FORMAT":
                    qtext = str(rng.choice(["표로 보여줘", "더 짧게 정리해",
                                            "차트로 보여줘", "자세히 설명해줘"]))
                else:
                    qtext = qtext_base

                if is_oth:
                    intent_pred, tool_called = "OTH", "get_news_and_work"
                    cited = False

                rows.append({
                    "answer_text": atext,
                    "intent_pred": intent_pred, "tool_called": tool_called,
                    "response_len": float(len(atext)), "cited": cited, "csat": csat,
                    "query_text": qtext, "overblock": (
                        bool(rng.random() < .12) if outcome == "blocked" else None),
                    "halluc_audit": (
                        bool(rng.random() < (.45 if not tool_called else .05))
                        if rng.random() < .10 and outcome == "success" else None),
                    "query_id": f"Q{len(rows):08d}", "session_id": sid, "user_id": uid,
                    "ts": qts,
                    "l1_stage": stage, "l2_intent": intent,
                    "f1_target_type": "stock" if tgt else "market",
                    "f2_tense": f2, "f3_personal": f3, "f4_compliance": f4,
                    "f5_response": "fact", "f6_turn": f6,
                    "slot_target": tgt,
                    "tool_expected": f"tool_{intent.split('.')[1]}",
                    "source_expected": (["fnguide"] if intent in FNGUIDE_INTENTS
                                        else ["quote"] if intent in CONTROL_INTENTS
                                        else ["internal"]),
                    "answerable": answerable, "outcome": outcome,
                    "latency_ms": float(rng.gamma(3, 400)),
                    "sample_stratum": "base", "sample_weight": 1.0,
                })

                prev_ok = (outcome == "success")
                prev_intent, prev_tgt = intent, list(tgt)

                # --- 종목 조회 → 주문 (심어놓은 증분 효과)
                if tgt and intent in EVALUATE_LOOKUP_INTENTS:
                    tk = tgt[0]
                    p_order = .10 + (TRUE_CHATBOT_ORDER_LIFT if outcome == "success" else 0)
                    p_order += {"invest": .06, "mixed": .02, "service": -.04}[ut]
                    if rng.random() < max(p_order, .01):
                        order_rows.append({
                            "user_id": uid, "ticker": tk,
                            "ts": dte + pd.Timedelta(days=int(rng.integers(0, 4))),
                            "order_amt": float(rng.lognormal(14, 1)), "filled": True})

            # 비챗봇 경로 조회 (대조군)
            for _ in range(int(rng.integers(0, 4))):
                tk = str(rng.choice(tickers))
                view_rows.append({"user_id": uid, "ticker": tk,
                                  "ts": dte + pd.Timedelta(hours=int(rng.integers(9, 16))),
                                  "channel": "detail"})
                p_order = .10 + {"invest": .06, "mixed": .02, "service": -.04}[ut]
                if rng.random() < max(p_order, .01):
                    order_rows.append({
                        "user_id": uid, "ticker": tk,
                        "ts": dte + pd.Timedelta(days=int(rng.integers(0, 4))),
                        "order_amt": float(rng.lognormal(14, 1)), "filled": True})

        # 챗봇 이탈 후에도 앱은 계속 쓰는 사용자 (챗봇 이탈 ≠ 고객 이탈)
        if churn_date is not None and rng.random() < .55:
            tail = pd.date_range(churn_date, END, freq="D")
            for dte in tail[rng.random(len(tail)) < .15]:
                appsess_rows.append({"user_id": uid, "date": dte})

    return {
        "queries": pd.DataFrame(rows),
        "orders": pd.DataFrame(order_rows),
        "app_views": pd.DataFrame(view_rows),
        "app_sessions": pd.DataFrame(appsess_rows).drop_duplicates(),
        "users_truth": users,
    }
