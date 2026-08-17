"""
입력 스키마 정의 및 검증.

실데이터 컬럼명이 다르면 COLUMN_ALIASES만 수정하면 됩니다.
분석 코드는 전부 표준 컬럼명(아래 SCHEMA)에만 의존합니다.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------- 택소노미

L1_STAGES = [
    "LEARN", "DISCOVER", "EVALUATE", "EXECUTE",
    "MONITOR", "SETTLE", "SERVICE", "RECOVER",
]

L2_INTENTS = {
    "LEARN": ["LEARN.term", "LEARN.mechanism", "LEARN.product", "LEARN.process"],
    "DISCOVER": ["DISC.screen", "DISC.rank_event", "DISC.rank_metric", "DISC.theme",
                 "DISC.related", "DISC.trending", "DISC.ipo_pipeline",
                 "DISC.recommend_open"],
    "EVALUATE": ["EVAL.profile", "EVAL.financials", "EVAL.valuation", "EVAL.consensus",
                 "EVAL.price", "EVAL.supply_demand", "EVAL.news", "EVAL.disclosure",
                 "EVAL.score", "EVAL.compare", "EVAL.causal", "EVAL.interpret",
                 "EVAL.outlook", "EVAL.verdict"],
    "EXECUTE": ["EXEC.order_howto", "EXEC.order_status", "EXEC.eligibility",
                "EXEC.margin", "EXEC.cost", "EXEC.ipo_subscribe", "EXEC.nav"],
    "MONITOR": ["MON.holdings", "MON.performance", "MON.alert", "MON.watchlist",
                "MON.rebalance", "MON.loss_reaction"],
    "SETTLE": ["SETL.dividend", "SETL.rights", "SETL.tax", "SETL.settlement",
               "SETL.record"],
    "SERVICE": ["SVC.account", "SVC.auth", "SVC.transfer", "SVC.loan",
                "SVC.policy", "SVC.app_setting", "SVC.channel"],
    "RECOVER": ["REC.followup", "REC.error", "REC.escalate", "REC.complaint",
                "OOS.out_of_domain", "OOS.chitchat", "RISK.distress"],
}
ALL_INTENTS = [i for v in L2_INTENTS.values() for i in v]

# 보고서·발표용 자연어 라벨. 코드는 내부 식별자일 뿐이므로 외부 문서에는 이쪽을 쓴다.
STAGE_LABEL_KO = {
    "LEARN": "개념·용어 이해", "DISCOVER": "종목 찾기",
    "EVALUATE": "종목 살펴보기", "EXECUTE": "주문·거래",
    "MONITOR": "내 계좌 확인", "SETTLE": "배당·세금·정산",
    "SERVICE": "계좌·앱 업무", "RECOVER": "오류·상담·기타",
}

INTENT_LABEL_KO = {
    # 개념·용어
    "LEARN.term": "용어 뜻 묻기",
    "LEARN.mechanism": "제도·원리 이해",
    "LEARN.product": "금융상품 구조 비교",
    "LEARN.process": "투자 절차 개념",
    # 종목 찾기
    "DISC.screen": "조건 걸어 종목 찾기",
    "DISC.rank_event": "급등·상한가 종목 보기",
    "DISC.rank_metric": "거래량·수익률 상위 보기",
    "DISC.theme": "테마·업종 살펴보기",
    "DISC.related": "관련주 찾기",
    "DISC.trending": "지금 많이 보는 종목",
    "DISC.ipo_pipeline": "상장 예정 종목 보기",
    "DISC.recommend_open": "종목 추천 요청",
    # 종목 살펴보기
    "EVAL.profile": "회사가 뭐 하는 곳인지",
    "EVAL.financials": "실적·재무 확인",
    "EVAL.valuation": "주가가 비싼지 싼지",
    "EVAL.consensus": "증권가 목표주가",
    "EVAL.price": "주가·시세 확인",
    "EVAL.supply_demand": "외국인·기관 매매동향",
    "EVAL.news": "관련 뉴스 보기",
    "EVAL.disclosure": "공시 내용 확인",
    "EVAL.score": "종목 평가 점수",
    "EVAL.compare": "종목끼리 비교",
    "EVAL.causal": "왜 오르내렸는지",
    "EVAL.interpret": "호재인지 악재인지",
    "EVAL.outlook": "앞으로 어떨지",
    "EVAL.verdict": "지금 사도 되는지",
    # 주문·거래
    "EXEC.order_howto": "주문하는 방법",
    "EXEC.order_status": "체결·정정·취소 확인",
    "EXEC.eligibility": "지금 거래 가능한지",
    "EXEC.margin": "증거금·신용거래",
    "EXEC.cost": "수수료·거래세",
    "EXEC.ipo_subscribe": "공모주 청약 방법",
    "EXEC.nav": "화면·메뉴 위치 찾기",
    # 내 계좌
    "MON.holdings": "내 보유 종목·평가금액",
    "MON.performance": "내 수익률 확인",
    "MON.alert": "알림 설정",
    "MON.watchlist": "관심종목 관리",
    "MON.rebalance": "내 포트폴리오 상담",
    "MON.loss_reaction": "손실 상황 대응 문의",
    # 배당·세금·정산
    "SETL.dividend": "배당금·지급 시기",
    "SETL.rights": "배당락·증자 일정",
    "SETL.tax": "세금 문의",
    "SETL.settlement": "결제일·출금 가능일",
    "SETL.record": "거래내역·증명서 발급",
    # 계좌·앱 업무
    "SVC.account": "계좌 개설·변경",
    "SVC.auth": "인증서·로그인",
    "SVC.transfer": "입출금·이체",
    "SVC.loan": "담보대출·신용융자",
    "SVC.policy": "약관·제도 안내",
    "SVC.app_setting": "앱 설정 변경",
    "SVC.channel": "지점·ATM·상담시간",
    # 오류·상담·기타
    "REC.followup": "앞 질문 이어가기",
    "REC.error": "오류·장애 신고",
    "REC.escalate": "상담원 연결 요청",
    "REC.complaint": "불만 제기",
    "OOS.out_of_domain": "증권 업무와 무관한 질문",
    "OOS.chitchat": "인사·잡담",
    "RISK.distress": "심각한 정서 위기 신호",
}


def label_ko(code) -> str:
    """의도·단계 코드를 자연어 라벨로. 미등록 코드는 원문 유지."""
    c = str(code).strip()
    return INTENT_LABEL_KO.get(c) or STAGE_LABEL_KO.get(c) or c
INTENT_TO_STAGE = {i: s for s, v in L2_INTENTS.items() for i in v}

# FnGuide 의존 의도 (DiD 처치군) — 실제 툴 인벤토리 확인 후 조정
FNGUIDE_INTENTS = [
    "EVAL.financials", "EVAL.valuation", "EVAL.consensus", "EVAL.score",
]
# 시세 기반 비의존 의도 (DiD 대조군)
CONTROL_INTENTS = [
    "EVAL.price", "DISC.rank_event", "DISC.rank_metric",
]

# 정보→주문 전환 분석에서 '조회'로 간주할 의도
EVALUATE_LOOKUP_INTENTS = [
    "EVAL.profile", "EVAL.financials", "EVAL.valuation", "EVAL.consensus",
    "EVAL.price", "EVAL.supply_demand", "EVAL.news", "EVAL.disclosure",
    "EVAL.score", "EVAL.compare", "EVAL.causal", "EVAL.interpret",
]

# ---------------------------------------------------------------- Facet 값

F4_LEVELS = ["P0", "P1", "P2", "P3"]
ANSWERABLE = ["yes", "no_source", "no_tool", "no_slot", "no_auth", "blocked", "unknown"]
OUTCOMES = ["success", "fail", "blocked"]

# 실패 귀속 코드
FAIL_CODES = {
    "D1": ("소스 미보유", "프로덕트", "조달·제휴"),
    "D2": ("소스 중단", "프로덕트", "복구·이중화"),
    "D3": ("소스 커버리지 결손", "데이터", "커버리지 확장"),
    "T1": ("툴 미구현", "엔지니어링", "함수 개발"),
    "T2": ("툴 오라우팅", "AI", "라우팅 개선"),
    "S1": ("슬롯 미해결", "대화설계", "되묻기·후보 제시"),
    "A1": ("인증 미충족", "UX", "인증 게이트 설계"),
    "X1": ("컨텍스트 상속 실패", "엔지니어링", "세션 상태 관리"),
    "C1": ("컴플라이언스 정당 차단", "-", "실패 아님"),
    "C2": ("과차단", "준법·AI", "임계값 조정"),
    "C3": ("차단 후 대체 제공 실패", "복합", "데이터 복구 연동"),
    "M1": ("모델 오류(잔차)", "AI", "학습·프롬프트"),
}
# 해결 난이도 가중 (우선순위 스코어용, 낮을수록 쉬움)
FAIL_COST = {"D2": 1.0, "T1": 1.5, "S1": 1.5, "X1": 1.5, "C2": 2.0,
             "C3": 2.0, "T2": 2.0, "A1": 2.0, "M1": 3.0, "D3": 3.0,
             "D1": 4.0, "C1": 0.0}

# ---------------------------------------------------------------- 테이블 스키마

SCHEMA = {
    "queries": {
        "query_id": "string", "session_id": "string", "user_id": "string",
        "ts": "datetime64[ns]",
        "l1_stage": "string", "l2_intent": "string",
        "f1_target_type": "string", "f2_tense": "string", "f3_personal": "string",
        "f4_compliance": "string", "f5_response": "string", "f6_turn": "string",
        "slot_target": "object",       # list[str] 종목코드
        "tool_expected": "string", "source_expected": "object",
        "answerable": "string", "outcome": "string",
        "latency_ms": "float64",
        "sample_stratum": "string", "sample_weight": "float64",
    },
    "orders": {
        "user_id": "string", "ticker": "string", "ts": "datetime64[ns]",
        "order_amt": "float64", "filled": "bool",
    },
    "app_views": {   # 챗봇이 아닌 경로의 종목 조회 (대조군 생성용)
        "user_id": "string", "ticker": "string", "ts": "datetime64[ns]",
        "channel": "string",
    },
    "app_sessions": {  # 앱 전체 세션 (챗봇 이탈 ≠ 고객 이탈 판별용)
        "user_id": "string", "date": "datetime64[ns]",
    },
}

# ------------------------------------------------- 기존 운영 분류 ↔ 신규 택소노미
# INTENT_CATEGORY2 는 신규 택소노미가 아니라 기존 운영 분류 코드다.
# 두 체계는 1:N 관계이므로 문자열 직접 비교가 아니라 크로스워크로 대조해야 한다.
# 실제 코드값(영문 enum)이 확인되면 키를 추가하십시오. 미매핑 값은 실행 시 출력된다.
LEGACY_CROSSWALK: dict[str, list[str]] = {
    # ── 키는 "그룹.서브코드" (INTENT_CATEGORY1.INTENT_CATEGORY2) ──
    # BASIC 처럼 그룹에 따라 의미가 완전히 다른 서브코드가 있으므로
    # 반드시 2단 키로 대조해야 한다. 그룹이 없으면 서브코드 단독 키로 폴백.

    # USM — 사용 안내
    "USM.INTRO": ["OOS.chitchat", "LEARN.process"],          # 챗봇 인사·소개
    "USM.PLATFORM_USE": ["EXEC.nav", "SVC.app_setting", "EXEC.order_howto",
                         "SVC.account", "SVC.auth", "SVC.transfer",
                         "SVC.loan", "SVC.channel"],
    "USM.INVITE_GUIDE": ["LEARN.process", "LEARN.mechanism",
                         "SVC.policy"],                       # 투자 유의사항
    "USM.INVITE_PROCESS": ["LEARN.process", "EXEC.order_howto",
                           "EXEC.eligibility", "EXEC.margin",
                           "EXEC.cost", "EXEC.order_status"],  # 주식거래 절차 전반

    # BFK — 기초 지식
    "BFK.FIN_TERMS": ["LEARN.term"],
    "BFK.FIN_PRODUCTS": ["LEARN.product"],

    # MKT — 시장
    "MKT.FORECAST": ["EVAL.outlook", "EVAL.consensus"],
    "MKT.TRENDS_ANALYSIS": ["DISC.theme", "DISC.related", "DISC.trending",
                            "EVAL.causal", "EVAL.interpret", "EVAL.outlook"],

    # RGP — 규정·세제
    "RGP.TAX_INFO": ["SETL.tax"],
    "RGP.LEGAL_ISSUES": ["SVC.policy", "LEARN.mechanism"],

    # IPO — 상장·공모
    "IPO.BASIC": ["DISC.ipo_pipeline", "LEARN.mechanism",
                  "LEARN.process"],                            # 상장 절차 개념
    "IPO.SCHEDULE": ["DISC.ipo_pipeline", "EXEC.ipo_subscribe"],

    # CMP — 기업
    "CMP.BASIC": ["EVAL.profile"],                             # 설립일·경영진
    "CMP.OVERVIEW": ["EVAL.profile"],
    "CMP.NEWS": ["EVAL.news", "EVAL.disclosure", "SETL.rights"],
    "CMP.DIVIDEND": ["SETL.dividend", "SETL.rights"],
    "CMP.FINANCIAL": ["EVAL.financials"],
    "CMP.ANAYLYSIS": ["EVAL.score", "EVAL.compare", "EVAL.interpret",
                      "EVAL.profile"],
    "CMP.ANALYSIS": ["EVAL.score", "EVAL.compare", "EVAL.interpret",
                     "EVAL.profile"],

    # PTD — 시세
    "PTD.PRICE_TRADE": ["EVAL.price", "EVAL.supply_demand",
                        "DISC.rank_event", "DISC.rank_metric"],
    "PTD.PAST_PRICE": ["EVAL.price"],

    # VAL — 밸류에이션
    "VAL.MULTIPLE_INFO": ["EVAL.valuation"],
    "VAL.CONSENSUS": ["EVAL.consensus"],

    # OTH — 분류 실패. 대응 없음이 정상이며, 정합률 0%가 곧 폴백 규모다.
    "OTH.OTHER": [],
    "OTHER": [], "OTH": [], "기타": [],
}

# 그룹 없이 서브코드만 있을 때의 폴백. BASIC 은 의미 충돌이라 의도적으로 제외한다.
_AMBIGUOUS_SUBCODES = {"BASIC"}
for _k, _v in list(LEGACY_CROSSWALK.items()):
    if "." in _k:
        _sub = _k.split(".", 1)[1]
        if _sub not in _AMBIGUOUS_SUBCODES and _sub not in LEGACY_CROSSWALK:
            LEGACY_CROSSWALK[_sub] = _v


def legacy_key(group, sub) -> str:
    """INTENT_CATEGORY1/2 → 크로스워크 키. 그룹이 없으면 서브코드 단독."""
    g = "" if group is None or str(group) in ("nan", "<NA>") else str(group).strip()
    s_ = "" if sub is None or str(sub) in ("nan", "<NA>") else str(sub).strip()
    if g and f"{g}.{s_}" in LEGACY_CROSSWALK:
        return f"{g}.{s_}"
    return s_


# 신규 택소노미에만 있고 기존 분류에 대응이 없는 의도 (설계서상 '신설')
LEGACY_UNCOVERED = [
    "MON.holdings", "MON.performance", "MON.alert", "MON.watchlist",
    "MON.rebalance", "MON.loss_reaction",
    "EVAL.compare", "EVAL.causal", "EVAL.verdict",
    "EXEC.nav", "EXEC.order_status", "EXEC.eligibility", "EXEC.margin",
    "DISC.screen", "DISC.recommend_open",
    "SETL.settlement", "SETL.record",
    "REC.followup", "REC.error", "REC.escalate", "REC.complaint",
    "OOS.out_of_domain", "OOS.chitchat", "RISK.distress",
]

# ---------------------------------------------------------------- 선택 컬럼
# 있으면 추가 분석이 활성화되고, 없으면 해당 분석만 건너뛴다.
OPTIONAL_COLUMNS = {
    "intent_pred":   ("string",  pd.NA),   # 운영 의도분류 결과 (gold=l2_intent 와 대조)
    "tool_called":   ("string",  pd.NA),   # 실제 호출된 툴 (없으면 무툴 응답)
    "response_len":  ("float64", float("nan")),  # 응답 길이(자)
    "cited":         ("boolean", pd.NA),   # 근거·출처 포함 여부
    "csat":          ("float64", float("nan")),  # 태스크 직후 만족도
    "intent_pred_group": ("string", pd.NA),  # INTENT_CATEGORY1 (그룹 코드)
    "query_text":    ("string",  pd.NA),   # 재질문 판정용 원문 (없으면 의도+슬롯으로 대체)
    "answer_text":   ("string",  pd.NA),   # 응답 본문 (HTML 포함) — 관련성·구조 분석용
    "overblock":     ("boolean", pd.NA),   # 사후 검수: 과차단 여부
    "halluc_audit":  ("boolean", pd.NA),   # 사후 검수: 환각 여부 (샘플링 검수분만)
    "needs_review":  ("boolean", pd.NA),   # 어노테이터가 확신 못한 건 → 본 집계에서 분리
    "confidence":    ("float64", float("nan")),  # 라벨 확신도
    "secondary":     ("object",  None),    # 부차 의도 리스트 → 공기 분석
    "tool_args":     ("string",  pd.NA),   # 툴 호출 인자 원본(JSON)
    "tool_query":    ("string",  pd.NA),   # 툴에 전달된 재작성 쿼리 → 질의 왜곡 분석
    "slot_period":   ("string",  pd.NA),
    "slot_metric":   ("string",  pd.NA),
    "slot_sort":     ("string",  pd.NA),
    "slot_count":    ("string",  pd.NA),
    "tool_steps":    ("float64", float("nan")),  # 한 응답의 툴 연쇄 호출 수
    "tool_brief":    ("string",  pd.NA),         # 계획 단계 설명
    "protector_flag": ("boolean", pd.NA),        # 원본에 'P'로 표기된 차단 여부
}

# 형식 재요청 판정 키워드 (원문이 있을 때만 사용)
FORMAT_REQUEST_PATTERNS = (
    "표로", "테이블", "차트", "그래프", "요약해", "짧게", "간단히", "자세히",
    "더 길게", "정리해", "리스트로", "번호로", "그림으로", "한 줄로",
)


def ensure_optional(df: pd.DataFrame) -> pd.DataFrame:
    """선택 컬럼을 없으면 채워 넣어 하위 분석이 안전하게 동작하게 한다."""
    df = df.copy()
    for col, (dt, default) in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.Series([default] * len(df), index=df.index, dtype=dt)
        else:
            try:
                df[col] = df[col].astype(dt)
            except (TypeError, ValueError):
                pass
    return df


def available_optional(df: pd.DataFrame) -> dict[str, bool]:
    """선택 컬럼별 실제 사용 가능 여부(전부 결측이면 False)."""
    return {c: (c in df.columns and df[c].notna().any()) for c in OPTIONAL_COLUMNS}


# 실데이터 컬럼명 → 표준명. 필요 시 여기만 수정.
COLUMN_ALIASES: dict[str, dict[str, str]] = {
    "queries": {},
    "orders": {},
    "app_views": {},
    "app_sessions": {},
}


class SchemaError(Exception):
    pass


def normalize(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """별칭 적용 + 필수 컬럼 확인 + dtype 정리."""
    if table not in SCHEMA:
        raise SchemaError(f"알 수 없는 테이블: {table}")
    df = df.rename(columns=COLUMN_ALIASES.get(table, {}))
    spec = SCHEMA[table]
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise SchemaError(f"[{table}] 누락 컬럼: {missing}")
    extra = [c for c in df.columns
             if c not in spec and c in OPTIONAL_COLUMNS]
    out = df[list(spec) + extra].copy()
    for col, dt in spec.items():
        if dt.startswith("datetime"):
            out[col] = pd.to_datetime(out[col])
        elif dt == "object":
            pass
        else:
            out[col] = out[col].astype(dt)
    return out


def validate_queries(df: pd.DataFrame) -> list[str]:
    """치명적이지 않은 품질 경고를 반환. 분석 전 반드시 확인."""
    warn: list[str] = []

    n_na = int(df["l2_intent"].isna().sum())
    if n_na:
        warn.append(f"l2_intent 결측 {n_na:,}행 ({n_na/len(df):.1%}) — "
                    "어노테이션 실패분. 의도별 집계에서 자동 제외됨")
    for c in ("l1_stage", "f4_compliance", "outcome", "answerable"):
        m = int(df[c].isna().sum()) if c in df.columns else 0
        if m:
            warn.append(f"{c} 결측 {m:,}행 ({m/len(df):.1%})")

    bad = set(df["l2_intent"].dropna().unique()) - set(ALL_INTENTS)
    if bad:
        warn.append(f"택소노미 외 의도 {len(bad)}종: {sorted(bad)[:5]}")

    mism = df[df["l2_intent"].map(INTENT_TO_STAGE) != df["l1_stage"]]
    if len(mism):
        warn.append(f"l1_stage와 l2_intent 불일치 {len(mism)}건 — 라벨 정합성 확인 필요")

    cov = df["user_id"].notna().mean()
    warn.append(f"user_id 커버리지 {cov:.1%}")
    if cov < 0.99:
        by_intent = (df.assign(has=df["user_id"].notna())
                       .groupby("l2_intent")["has"].mean().sort_values())
        low = by_intent[by_intent < cov - 0.05]
        if len(low):
            warn.append(
                "  ↳ 의도별 ID 커버리지 편차 있음. ID 기반 분석 시 역가중 필요: "
                + ", ".join(f"{k}={v:.0%}" for k, v in low.head(5).items()))

    if (df["sample_weight"] <= 0).any():
        warn.append("sample_weight ≤ 0 존재 — 층화 역가중 확인")

    n_user_multi = (df.groupby("user_id")["session_id"].nunique() > 1).mean()
    warn.append(f"2세션 이상 사용자 비율 {n_user_multi:.1%} (패널 분석 가용성)")

    return warn
