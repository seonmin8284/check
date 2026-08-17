"""
prep_data.py → run_analysis.py 파이프라인 검증용 가상 원본 로그 생성기.

    python generate_sample_data.py                    # 기본값으로 생성
    python generate_sample_data.py --users 700 --seed 42

생성물
    labeled_all.csv        원본 로그 스키마 + 재어노테이션 컬럼 (prep_data.py 입력)
    data/orders.pkl        C트랙 주문 데이터 (+ .csv 사본)
    data/app_views.pkl     비챗봇 경로 종목조회 (대조군)
    data/app_sessions.pkl  앱 전체 세션 (챗봇 이탈 ≠ 고객 이탈 판별용)

실데이터가 아니라 파이프라인 검증용입니다. 아래 효과를 의도적으로 심어두었으므로
run_analysis.py 가 이것들을 되찾아내는지로 분석 코드를 확인할 수 있습니다.

  ① DiD        — FnGuide 의존 의도의 실패율이 소스 중단일 이후 급등
  ② 정책       — Protector 도입일 이후에만 P3 의도가 차단됨 (도입 전 구간 확보)
  ③ 멀티턴     — 직전 턴 실패 시 같은 질문 재시도·형식 재요청 확률 상승
  ④ OTH 폴백   — 분류 실패건이 무관한 폴백 응답을 받고도 '성공'으로 집계됨
  ⑤ C트랙      — 챗봇 조회 성공 후 해당 종목 주문 확률 상승
  ⑥ 생존       — 차단 경험자의 이탈 위험 상승

주의: 택소노미 코드(l1_stage / l2_intent)는 mts_analysis/schema.py 에서 직접
가져옵니다. 임의의 라벨을 쓰면 분석 모듈 전체가 빈 결과를 냅니다.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from mts_analysis.schema import (
    ALL_INTENTS,
    CONTROL_INTENTS,
    EVALUATE_LOOKUP_INTENTS,
    FNGUIDE_INTENTS,
    INTENT_TO_STAGE,
    L2_INTENTS,
    LEGACY_CROSSWALK,
)

# ---------------------------------------------------------------- 심어놓는 효과
TRUE_DID_EFFECT = 0.35          # 소스 중단 후 FnGuide 의도 실패율 증가폭
TRUE_OTH_RATE = 0.07            # 분류 실패 → 폴백 응답 비율
TRUE_REPEAT_P_FAIL = 0.32       # 직전 실패 후 재질문 확률
TRUE_REPEAT_P_OK = 0.06         # 직전 성공 후 재질문 확률
TRUE_ORDER_LIFT = 0.06          # 챗봇 조회 성공 → 주문 확률 증가분
TRUE_BLOCK_CHURN_MULT = 1.9     # 차단 경험자의 이탈 해저드 배수

# ---------------------------------------------------------------- 기준일
START_DATE = pd.Timestamp("2026-01-05")
PROTECTOR_DATE = pd.Timestamp("2026-03-01")   # run_analysis.py --protector 기본값
OUTAGE_DATE = pd.Timestamp("2026-04-15")      # run_analysis.py --outage 기본값
END_DATE = pd.Timestamp("2026-06-30")         # run_analysis.py --end 기본값

TICKERS = [
    ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("035420", "NAVER"),
    ("035720", "카카오"), ("005380", "현대차"), ("373220", "LG에너지솔루션"),
    ("005490", "POSCO홀딩스"), ("051910", "LG화학"), ("207940", "삼성바이오로직스"),
    ("068270", "셀트리온"), ("012330", "현대모비스"), ("055550", "신한지주"),
    ("105560", "KB금융"), ("003670", "포스코퓨처엠"), ("247540", "에코프로비엠"),
    ("086520", "에코프로"), ("028260", "삼성물산"), ("015760", "한국전력"),
]
TICKER_NAME = dict(TICKERS)
CODES = [c for c, _ in TICKERS]

# ---------------------------------------------------------------- Facet 규칙
# synth.py 의 어휘를 그대로 따릅니다. 값이 다르면 mechanism.py 등이 인식하지 못합니다.
P3_INTENTS = {"EVAL.verdict", "DISC.recommend_open", "MON.rebalance", "MON.loss_reaction"}
P2_INTENTS = {"EVAL.outlook", "EVAL.causal", "EVAL.interpret", "RISK.distress"}
ACCOUNT_INTENTS = set(L2_INTENTS["MONITOR"]) | {
    "EXEC.order_status", "EXEC.margin", "SETL.settlement", "SETL.record",
    "SVC.account", "SVC.auth", "SVC.transfer", "SVC.loan",
}
FUTURE_INTENTS = {"EVAL.outlook", "EVAL.verdict", "DISC.ipo_pipeline", "SETL.rights"}
PAST_INTENTS = {"EVAL.causal", "SETL.record", "EXEC.order_status", "MON.performance"}
# 종목 슬롯이 붙는 의도
TICKER_INTENTS = set(EVALUATE_LOOKUP_INTENTS) | {
    "EVAL.outlook", "EVAL.verdict", "DISC.related", "MON.alert", "MON.watchlist",
    "SETL.dividend", "SETL.rights",
}
NAV_RESPONSE = {"EXEC.nav", "SVC.app_setting", "SVC.channel"}
PROC_RESPONSE = {"LEARN.process", "EXEC.order_howto", "EXEC.ipo_subscribe",
                 "SVC.account", "SVC.auth", "SVC.transfer", "SETL.tax"}

# 의도별 기저 실패율. 나머지는 STAGE 기본값.
BASE_FAIL = {
    "EXEC.nav": 0.52,           # 딥링크 카탈로그 부재
    "DISC.screen": 0.30,
    "DISC.theme": 0.24,
    "DISC.related": 0.24,
    "EVAL.causal": 0.20,
    "EVAL.interpret": 0.18,
    "MON.rebalance": 0.15,
    "SETL.record": 0.20,
    "SVC.channel": 0.16,
    "REC.error": 0.12,
}
STAGE_FAIL = {"LEARN": .06, "DISCOVER": .12, "EVALUATE": .09, "EXECUTE": .12,
              "MONITOR": .10, "SETTLE": .11, "SERVICE": .13, "RECOVER": .10}

# 사용자 유형별 L1 단계 선호
STAGE_W = {
    "invest":  {"LEARN": .04, "DISCOVER": .17, "EVALUATE": .42, "EXECUTE": .10,
                "MONITOR": .14, "SETTLE": .04, "SERVICE": .05, "RECOVER": .04},
    "mixed":   {"LEARN": .10, "DISCOVER": .12, "EVALUATE": .28, "EXECUTE": .12,
                "MONITOR": .14, "SETTLE": .07, "SERVICE": .12, "RECOVER": .05},
    "service": {"LEARN": .15, "DISCOVER": .05, "EVALUATE": .10, "EXECUTE": .12,
                "MONITOR": .10, "SETTLE": .12, "SERVICE": .31, "RECOVER": .05},
}

# ---------------------------------------------------------------- 기존 운영 분류 역매핑
_REV_LEGACY: dict[str, list[str]] = {}
for _k, _vs in LEGACY_CROSSWALK.items():
    if "." not in _k:
        continue
    for _v in _vs:
        _REV_LEGACY.setdefault(_v, []).append(_k)
_LEGACY_KEYS = [k for k in LEGACY_CROSSWALK if "." in k and k != "OTH.OTHER"]


def legacy_code(intent: str, rng: random.Random) -> tuple[str, str]:
    """신규 의도 → 기존 운영 분류 코드(그룹, 서브). 오분류·미포괄을 재현한다."""
    cands = _REV_LEGACY.get(intent)
    if not cands:
        # 기존 체계가 포괄하지 못하던 신규 의도 → 대부분 OTH 로 흘러감
        key = "OTH.OTHER" if rng.random() < 0.65 else rng.choice(_LEGACY_KEYS)
    elif rng.random() < 0.20:
        key = rng.choice(_LEGACY_KEYS)          # 운영 분류 오분류
    else:
        key = rng.choice(cands)
    g, s = key.split(".", 1)
    return g, s


# ---------------------------------------------------------------- 툴 정의
def tool_for(intent: str) -> str:
    return "get_" + intent.split(".", 1)[1]


def tool_args(intent: str, ticker: str | None, rng: random.Random) -> dict:
    """
    툴 호출 인자. 키 이름은 prep_data.ARG_TO_SLOT 에 등록된 것만 슬롯으로 복원된다.
    (target→ticker, period→base_d/start_d/end_d/base_q, metric, sort→order, count→top_n)
    """
    a: dict = {}
    if ticker:
        a["ticker"] = ticker
    if intent == "EVAL.compare" and ticker:
        a["ticker"] = [ticker, rng.choice([c for c in CODES if c != ticker])]
    if intent in ("EVAL.financials", "EVAL.valuation"):
        a["base_q"] = rng.choice(["2025Q4", "2026Q1"])
    if intent in ("EVAL.price", "EVAL.supply_demand", "EVAL.causal"):
        a["start_d"] = "2026-01-02"
        a["end_d"] = "2026-06-30"
    if intent in ("DISC.rank_event", "DISC.rank_metric", "DISC.trending"):
        a["base_d"] = "2026-06-30"
        a["top_n"] = rng.choice([10, 20, 30])
        a["order"] = rng.choice(["desc", "asc"])
    if intent in ("DISC.rank_metric", "EVAL.valuation", "EVAL.score"):
        a["metric"] = rng.choice(["PER", "PBR", "ROE", "영업이익증가율"])
    if intent.startswith(("LEARN.", "SVC.", "RGP")) or intent == "DISC.screen":
        a["query"] = rng.choice(["공모주 청약 절차", "증거금률", "배당락일",
                                 "ROE 높은 종목", "이체 한도 변경"])
    return a


# ---------------------------------------------------------------- 원문 생성
QTPL = {
    "EVAL.price": "{name} 지금 주가 얼마야",
    "EVAL.financials": "{name} 작년 영업이익 알려줘",
    "EVAL.valuation": "{name} PER 비싼 편이야",
    "EVAL.consensus": "{name} 목표주가 컨센서스 보여줘",
    "EVAL.score": "{name} 종목 점수 어때",
    "EVAL.profile": "{name} 무슨 회사야",
    "EVAL.supply_demand": "{name} 외국인 순매수 추이 알려줘",
    "EVAL.news": "{name} 최근 뉴스 정리해줘",
    "EVAL.disclosure": "{name} 공시 뭐 떴어",
    "EVAL.compare": "{name} 이랑 비교해줘",
    "EVAL.causal": "{name} 왜 떨어졌어",
    "EVAL.interpret": "{name} 이번 공시 호재야 악재야",
    "EVAL.outlook": "{name} 앞으로 오를까",
    "EVAL.verdict": "{name} 지금 사도 될까",
    "DISC.rank_event": "오늘 상한가 종목 알려줘",
    "DISC.rank_metric": "거래량 상위 종목 보여줘",
    "DISC.screen": "PER 낮고 배당 주는 종목 찾아줘",
    "DISC.theme": "이차전지 테마 어때",
    "DISC.related": "{name} 관련주 뭐 있어",
    "DISC.trending": "지금 사람들 많이 보는 종목",
    "DISC.ipo_pipeline": "이번 달 공모주 일정 알려줘",
    "DISC.recommend_open": "지금 살 만한 종목 추천해줘",
    "MON.holdings": "내 보유 종목 알려줘",
    "MON.performance": "내 수익률 얼마야",
    "MON.rebalance": "내 포트폴리오 어떻게 바꿔야 해",
    "MON.loss_reaction": "지금 손실인데 손절해야 하나",
    "EXEC.nav": "주문 화면 어디 있어",
    "EXEC.order_howto": "예약주문 어떻게 해",
    "EXEC.order_status": "아까 낸 주문 체결됐어",
    "SETL.dividend": "{name} 배당금 언제 들어와",
    "SETL.tax": "해외주식 양도세 어떻게 계산해",
    "SVC.auth": "공동인증서 재발급 어떻게 해",
    "SVC.transfer": "출금 한도 어떻게 올려",
    "LEARN.term": "PBR 이 뭐야",
    "REC.escalate": "상담원 연결해줘",
    "REC.complaint": "답변이 계속 이상한데요",
}
FORMAT_REQ = ["표로 보여줘", "더 짧게 정리해줘", "차트로 보여줘", "자세히 설명해줘",
              "한 줄로 요약해줘"]


def question_text(intent: str, ticker: str | None, rng: random.Random) -> str:
    name = TICKER_NAME.get(ticker or "", "코스피")
    tpl = QTPL.get(intent)
    if tpl is None:
        tpl = "{name} " + intent.split(".", 1)[1].replace("_", " ") + " 알려줘"
    return tpl.format(name=name)


def answer_json(outcome: str, intent: str, ticker: str | None, is_oth: bool,
                rng: random.Random) -> tuple[str, str]:
    """(ANSWER JSON 문자열, 내부 text) 반환. 성공만 HTML 렌더된다."""
    name = TICKER_NAME.get(ticker or "", "코스피")
    if is_oth:
        # ★ 폴백: 렌더는 되지만 질문과 무관한 응답 → 성공으로 집계되는 구조적 결함
        text = ('<div class="news"><p>최근 증시 동향 요약입니다. 코스피는 전일 대비 '
                '상승 마감했으며 외국인 순매수가 이어졌습니다.</p></div>')
    elif outcome == "success":
        rows = "".join(
            f"<tr><td>{y}</td><td>{rng.randint(100, 9999):,}</td></tr>"
            for y in range(2021, 2026))
        lead = "<p>요청하신 내용을 정리했습니다.</p>" if rng.random() < 0.35 else ""
        text = (f'<div class="answer">{lead}<p>{name} 조회 결과입니다.</p>'
                f"<table><tbody>{rows}</tbody></table></div>")
    elif outcome == "blocked":
        text = rng.choice([
            "투자 판단은 도와드리기 어렵습니다. 관련 정보만 안내해 드릴 수 있습니다.",
            "특정 종목에 대한 투자 권유는 제공해 드릴 수 없습니다.",
            "매매 시점에 대한 조언해 드릴 수 없습니다. 투자 결정은 고객님 책임입니다.",
        ])
    else:
        text = rng.choice([
            "요청하신 정보를 찾지 못했습니다.",
            "일시적인 오류로 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            "해당 기능은 아직 준비 중입니다.",
        ])
    return json.dumps({"id": f"a{rng.randint(10**7, 10**8)}", "type": "bot",
                       "text": text}, ensure_ascii=False), text


def kdate(dt) -> str:
    """한국어 오전/오후 표기. prep_data.parse_kdatetime 이 읽는 형식."""
    ampm = "오전" if dt.hour < 12 else "오후"
    h12 = dt.hour % 12 or 12
    return f"{dt:%Y-%m-%d} {ampm} {h12}:{dt.minute:02d}:{dt.second:02d}"


# ---------------------------------------------------------------- 생성 본체
def generate(n_users: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    days = pd.date_range(START_DATE, END_DATE, freq="D")
    n_weeks = max(1, len(days) // 7)

    # --- 사용자 생성: 초기 유입 + 이후 꾸준한 신규 유입(파일럿 오탐 방지)
    users = []
    for i in range(n_users):
        if i < int(n_users * 0.45):
            first = START_DATE + timedelta(days=rng.randint(0, 20))
        else:
            first = START_DATE + timedelta(days=rng.randint(0, len(days) - 25))
        utype = rng.choices(["invest", "mixed", "service"], weights=[.42, .38, .20])[0]
        users.append({
            "uid": f"U{100000 + i}",
            "first": first,
            "type": utype,
            "intensity": float(np.clip(nrng.lognormal(-0.9, 0.7), 0.04, 0.85)),
            "dependency": float(nrng.beta(2, 3)),   # FnGuide 의존 성향
            "alive": True,
            "blocked_ever": False,
            "recent_fail": 0.0,
        })

    rows, order_rows, view_rows, sess_rows = [], [], [], []
    qi = 0

    for day in days:
        wd = day.weekday()
        weekend = wd >= 5
        for u in users:
            if not u["alive"] or day < u["first"]:
                continue

            # --- 앱 세션 (챗봇을 안 써도 앱은 열 수 있음)
            p_app = min(0.95, u["intensity"] * 2.6) * (0.45 if weekend else 1.0)
            opened_app = rng.random() < p_app
            if opened_app:
                sess_rows.append({"user_id": u["uid"], "date": day})

            p_use = u["intensity"] * (0.35 if weekend else 1.0)
            if not (opened_app and rng.random() < p_use):
                _churn_step(u, day, rng, used=False)
                continue

            n_sessions = 1 if rng.random() < 0.82 else 2
            for s in range(n_sessions):
                base_hour = rng.choices(
                    [8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 21, 22],
                    weights=[4, 12, 14, 11, 7, 9, 10, 9, 6, 6, 7, 5])[0]
                t = day + timedelta(hours=base_hour + 5 * s,
                                    minutes=rng.randint(0, 59),
                                    seconds=rng.randint(0, 59))
                _emit_session(u, t, day, rows, order_rows, rng, nrng,
                              lambda: _next_qid(rows))

            # --- 비챗봇 경로 조회 (C트랙 대조군)
            for _ in range(rng.randint(0, 3)):
                tk = rng.choice(CODES)
                vt = day + timedelta(hours=rng.randint(9, 16), minutes=rng.randint(0, 59))
                view_rows.append({"user_id": u["uid"], "ticker": tk, "ts": vt,
                                  "channel": rng.choice(["detail", "search", "watchlist"])})
                if rng.random() < _base_order_p(u["type"]):
                    order_rows.append({
                        "user_id": u["uid"], "ticker": tk,
                        "ts": vt + timedelta(days=rng.randint(0, 3)),
                        "order_amt": float(nrng.lognormal(14, 1)), "filled": True})

            _churn_step(u, day, rng, used=True)

    q = pd.DataFrame(rows)
    orders = pd.DataFrame(order_rows,
                          columns=["user_id", "ticker", "ts", "order_amt", "filled"])
    views = pd.DataFrame(view_rows, columns=["user_id", "ticker", "ts", "channel"])
    sessions = pd.DataFrame(sess_rows, columns=["user_id", "date"]).drop_duplicates()
    return q, orders, views, sessions


def _next_qid(rows) -> str:
    return f"MSG{len(rows) + 1:08d}"


def _base_order_p(utype: str) -> float:
    return {"invest": .13, "mixed": .09, "service": .04}[utype]


def _churn_step(u: dict, day, rng: random.Random, used: bool) -> None:
    """일 단위 이탈 해저드. 차단 경험·연속 실패가 위험을 올린다."""
    h = 0.0055
    if u["blocked_ever"]:
        h *= TRUE_BLOCK_CHURN_MULT
    h *= (1.0 + 1.2 * min(u["recent_fail"], 1.0))
    if not used:
        h *= 1.3
    if rng.random() < h:
        u["alive"] = False
    u["recent_fail"] *= 0.85          # 나쁜 경험의 기억은 서서히 옅어진다


def _pick_intent(u: dict, rng: random.Random) -> str:
    stage = rng.choices(list(STAGE_W[u["type"]]),
                        weights=list(STAGE_W[u["type"]].values()))[0]
    pool = L2_INTENTS[stage]
    w = []
    for it in pool:
        base = 1.0
        if it in FNGUIDE_INTENTS:
            base *= 0.6 + 2.4 * u["dependency"]     # 의존도가 높은 사용자일수록 자주
        if it in CONTROL_INTENTS:
            base *= 1.6
        if it in ("RISK.distress",):
            base *= 0.05
        if it in ("OOS.out_of_domain", "OOS.chitchat"):
            base *= 0.6
        w.append(base)
    return rng.choices(pool, weights=w)[0]


def _emit_session(u, t0, day, rows, order_rows, rng, nrng, _qid) -> None:
    n_turns = rng.choices([1, 2, 3, 4, 5], weights=[38, 27, 18, 11, 6])[0]
    prev_intent, prev_ticker, prev_ok = None, None, True
    t = t0

    for turn in range(n_turns):
        # --- 의도 결정 (직전 실패 시 재질문·형식 재요청 확률 상승)
        repeat_mode = None
        if turn > 0 and prev_intent is not None:
            p_rep = TRUE_REPEAT_P_FAIL if not prev_ok else TRUE_REPEAT_P_OK
            p_fmt = 0.16 if not prev_ok else 0.04
            r = rng.random()
            if r < p_rep:
                repeat_mode, intent = "REPEAT", prev_intent
            elif r < p_rep + p_fmt:
                repeat_mode, intent = "FORMAT", prev_intent
            else:
                intent = _pick_intent(u, rng)
        else:
            intent = _pick_intent(u, rng)
        stage = INTENT_TO_STAGE[intent]

        # --- Facet
        f4 = ("P3" if intent in P3_INTENTS else
              "P2" if intent in P2_INTENTS else
              "P1" if stage in ("EVALUATE", "DISCOVER", "MONITOR", "SETTLE") else "P0")
        f3 = "account_required" if intent in ACCOUNT_INTENTS else "none"
        f2 = ("future" if intent in FUTURE_INTENTS else
              "past" if intent in PAST_INTENTS else "present")
        f6 = "followup" if (turn > 0 and (repeat_mode or rng.random() < 0.35)) else "new"
        f5 = ("navigation" if intent in NAV_RESPONSE else
              "procedure" if intent in PROC_RESPONSE else "fact")

        # --- 대상 종목
        if repeat_mode and prev_ticker:
            ticker = prev_ticker
        elif intent in TICKER_INTENTS and rng.random() < 0.88:
            ticker = rng.choice(CODES)
        else:
            ticker = None
        f1 = "stock" if ticker else "market"

        # --- 결과 결정 (★ DiD·정책 효과를 여기서 심는다)
        fail_p = BASE_FAIL.get(intent, STAGE_FAIL[stage])
        if f6 == "followup":
            fail_p += 0.07                                   # 컨텍스트 상속 실패
        if intent in FNGUIDE_INTENTS and day >= OUTAGE_DATE:
            fail_p += TRUE_DID_EFFECT                        # ★ 소스 중단
        auth_gate = f3 == "account_required" and rng.random() < 0.12
        if auth_gate:
            fail_p += 0.30

        if f4 == "P3" and day >= PROTECTOR_DATE:
            outcome, answerable = "blocked", "blocked"
        elif rng.random() < min(fail_p, 0.95):
            outcome = "fail"
            if intent in FNGUIDE_INTENTS and day >= OUTAGE_DATE:
                answerable = "no_source"
            elif intent == "EXEC.nav":
                answerable = "no_tool"
            elif intent in ("DISC.theme", "DISC.related", "DISC.screen"):
                answerable = "no_tool"
            elif auth_gate:
                answerable = "no_auth"
            elif f6 == "followup":
                answerable = "no_slot"
            else:
                answerable = "yes"                           # → M1 모델 잔차
        else:
            outcome, answerable = "success", "yes"

        # --- ★ OTH 폴백: 분류 실패 → 무관한 응답이 렌더되어 '성공'으로 집계
        is_oth = outcome != "blocked" and rng.random() < TRUE_OTH_RATE
        if is_oth:
            outcome, answerable = "success", "yes"

        # --- 툴 호출
        if is_oth:
            calls = [{"step": 1, "name": "get_news_and_work",
                      "inputs": {"query": "증시 동향"}, "brief": "폴백 응답 생성"}]
        elif stage in ("LEARN", "RECOVER") or outcome == "blocked":
            calls = []                                        # 툴 없이 응답
        elif outcome == "fail" and answerable in ("no_tool", "no_slot"):
            calls = []                                        # 툴 자체가 안 불림
        elif outcome == "success" and rng.random() < (
                0.18 if stage in ("DISCOVER", "EVALUATE") else 0.07):
            calls = []                                        # ★ 툴 없이 내재 지식 응답 = 환각 위험군
        else:
            calls = [{"step": 1, "name": tool_for(intent),
                      "inputs": tool_args(intent, ticker, rng),
                      "brief": f"{intent} 처리"}]
            if intent in ("EVAL.compare", "EVAL.interpret") and rng.random() < 0.55:
                calls.append({"step": 2, "name": "get_price",
                              "inputs": {"ticker": ticker or rng.choice(CODES),
                                         "base_d": "2026-06-30"},
                              "brief": "시세 보강"})

        # --- 원문
        qtext = (rng.choice(FORMAT_REQ) if repeat_mode == "FORMAT"
                 else question_text(intent, ticker, rng))
        ans, atext = answer_json(outcome, intent, ticker, is_oth, rng)

        # --- 운영 분류 (기존 체계)
        if is_oth:
            g, s = "OTH", "OTHER"
        else:
            g, s = legacy_code(intent, rng)

        elapsed = int(np.clip(nrng.gamma(3.0, 480), 180, 60000))
        req = t
        res = req + timedelta(milliseconds=elapsed)

        needs_review = rng.random() < 0.055
        conf = float(np.clip(nrng.normal(0.62 if needs_review else 0.86, 0.11), .3, .99))
        sec = ([rng.choice(L2_INTENTS[stage])] if rng.random() < 0.22 else [])

        rows.append({
            # ---- 원본 로그 스키마
            "ID": len(rows) + 1,
            "APP_NAME": "MTS",
            "APP_ID": "mts-chatbot",
            "VERSION": rng.choice(["5.1.0", "5.2.1", "5.3.0"]),
            "ELAPSED_TIME": elapsed,
            "CHAT_REQ_DATE": kdate(req),
            "CHAT_RES_DATE": kdate(res),
            "CHAT_USER_ID": u["uid"],
            "INTENT_CATEGORY1": g,
            "INTENT_CATEGORY2": s,
            "QUESTION": qtext,
            "ANSWER": ans,
            "FUNCTIONS": json.dumps(calls, ensure_ascii=False) if calls else None,
            "MSG_ID": f"MSG{len(rows) + 1:08d}",
            "CREATED_AT": kdate(req),
            "PROTECTOR": "P" if outcome == "blocked" else "",
            # ---- 재어노테이션 컬럼 (prep_data.ANNOTATION_ALIASES 로 매핑)
            "stage": stage,
            "primary": intent,
            "target": f1,
            "tense": f2,
            "personalization": f3,
            "compliance": f4,
            "response_type": f5,
            "turn_type": f6,
            "answerable": answerable,
            "outcome": outcome,
            "source_expected": ("fnguide" if intent in FNGUIDE_INTENTS else
                                "quote" if intent in CONTROL_INTENTS else "internal"),
            "tool_expected": tool_for(intent),
            "needs_review": needs_review,
            "confidence": round(conf, 3),
            "secondary": json.dumps(sec, ensure_ascii=False),
            # ---- 사후 검수 컬럼 (표본에만 존재)
            "csat": (round(float(np.clip(nrng.normal(
                4.2 if outcome == "success" else 2.4, 0.9), 1, 5)), 1)
                if rng.random() < 0.08 else ""),
            "overblock": ("TRUE" if rng.random() < 0.13 else "FALSE")
                if outcome == "blocked" and rng.random() < 0.30 else "",
            "halluc_audit": ("TRUE" if (is_oth or not calls) and rng.random() < 0.45
                             else "FALSE") if outcome == "success"
                            and rng.random() < 0.10 else "",
        })

        # --- 사용자 상태 갱신
        if outcome == "blocked":
            u["blocked_ever"] = True
            u["recent_fail"] = min(1.0, u["recent_fail"] + 0.35)
        elif outcome == "fail":
            u["recent_fail"] = min(1.0, u["recent_fail"] + 0.25)

        # --- ★ 챗봇 조회 → 주문 (심어놓은 증분 효과)
        if ticker and intent in EVALUATE_LOOKUP_INTENTS:
            p = _base_order_p(u["type"]) + (TRUE_ORDER_LIFT if outcome == "success" else 0)
            if rng.random() < max(p, 0.01):
                order_rows.append({
                    "user_id": u["uid"], "ticker": ticker,
                    "ts": day + timedelta(days=rng.randint(0, 3),
                                          hours=rng.randint(9, 15)),
                    "order_amt": float(nrng.lognormal(14, 1)), "filled": True})

        prev_intent, prev_ticker, prev_ok = intent, ticker, (outcome == "success")
        t = res + timedelta(seconds=rng.randint(15, 240))


RAW_COLUMNS = [
    "ID", "APP_NAME", "APP_ID", "VERSION", "ELAPSED_TIME", "CHAT_REQ_DATE",
    "CHAT_RES_DATE", "CHAT_USER_ID", "INTENT_CATEGORY1", "INTENT_CATEGORY2",
    "QUESTION", "ANSWER", "FUNCTIONS", "MSG_ID", "CREATED_AT", "PROTECTOR",
    "stage", "primary", "target", "tense", "personalization", "compliance",
    "response_type", "turn_type", "answerable", "outcome", "source_expected",
    "tool_expected", "needs_review", "confidence", "secondary",
    "csat", "overblock", "halluc_audit",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="labeled_all.csv", help="원본 로그 CSV 경로")
    ap.add_argument("--data-dir", default="./data", help="C트랙 보조 테이블 저장 위치")
    ap.add_argument("--users", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    q, orders, views, sessions = generate(args.users, args.seed)
    q = q[RAW_COLUMNS]
    q.to_csv(args.out, index=False, encoding="utf-8-sig")

    d = Path(args.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in [("orders", orders), ("app_views", views),
                     ("app_sessions", sessions)]:
        # prep_data.py 는 이미 존재하는 파일을 빈 스텁으로 덮지 않는다.
        df.to_pickle(d / f"{name}.pkl")
        df.to_csv(d / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(f"✅ {args.out}  {len(q):,}행 · {len(q.columns)}컬럼")
    print(f"   기간 {q['CHAT_REQ_DATE'].iloc[0][:10]} ~ {END_DATE:%Y-%m-%d}"
          f" · 사용자 {q['CHAT_USER_ID'].nunique():,}명")
    print(f"   outcome: " + ", ".join(
        f"{k} {v:.1%}" for k, v in q["outcome"].value_counts(normalize=True).items()))
    fn = q[q["primary"].isin(FNGUIDE_INTENTS)].copy()
    fn["d"] = pd.to_datetime(fn["CHAT_REQ_DATE"].str[:10])
    pre = (fn[fn["d"] < OUTAGE_DATE]["outcome"] == "fail").mean()
    post = (fn[fn["d"] >= OUTAGE_DATE]["outcome"] == "fail").mean()
    print(f"   [심어놓은 DiD] FnGuide 실패율 {pre:.1%} → {post:.1%} (Δ{post-pre:+.1%})")
    print(f"✅ {d}/orders.pkl {len(orders):,} · app_views.pkl {len(views):,} "
          f"· app_sessions.pkl {len(sessions):,}")
    print(f"\n다음: python prep_data.py --src {args.out} --out {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
