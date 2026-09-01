"""goal parse -> function 매핑 (capability graph) 과 커버리지/정답 리포트.

run_csv.py 가 낸 *_out.csv 의 json 컬럼을 읽어, 각 goal 을 함수 호출 집합으로
결정적으로 매핑한다. LLM 은 여기 관여하지 않는다.

이전 판은 (domain, facet) -> 함수 **하나** 였다. golden_labels.csv 를 맞춰보면
그 가정이 틀렸다는 게 드러난다. 질의 하나가 요구하는 것은 함수 하나가 아니라
서로 보완하는 함수 **집합**이고, 무엇이 더 붙는지는 facet 이 아니라 goal 에
달린 엔티티/제약/타입이 정한다. 그래서 테이블을 3층 그래프로 나눴다.

    1층 BASE       (domain, facet) -> 그 goal 을 답하는 주 함수
    2층 EXPANSION  goal 의 엔티티·타입·기간이 조건인 팬아웃 간선
    3층 LEXICON    구조 신호가 없어 값 문자열에 기대는 자리 (별도 계상)

3층은 원리가 아니라 휴리스틱이다. 격리해 두고 리포트에서 따로 세는 이유는,
파서가 신호를 못 내주고 있다는 사실을 테이블 뒤에 숨기지 않기 위해서다.

실행:
    .venv/Scripts/python.exe route.py                       # 골든의 source 전부
    .venv/Scripts/python.exe route.py invest_out.csv        # 지정한 것만
    .venv/Scripts/python.exe route.py --no-golden           # 채점 없이 커버리지만
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────
# 매핑 결과 종류
# ─────────────────────────────────────────────────────────────

RESOLVED = "RESOLVED"  # 함수가 붙음
AMBIGUOUS = "AMBIGUOUS"  # 후보는 있으나 파스만으로 못 고름 → 판별자 부족
SUBSUMED = "SUBSUMED"  # 호출하지 않음. 다른 goal 결과에 흡수됨
FALLBACK = "FALLBACK"  # facet 이 도메인과 안 맞음. 도메인 기본 함수로 흘림
UNMAPPED = "UNMAPPED"  # 대응 함수 자체가 없음

BASE, EXPANSION, LEXICON = "base", "expansion", "lexicon"


class PeriodSplit:
    """period 제약 유무로 갈리는 자리. 함수 시그니처가 판별자를 준다."""

    def __init__(self, with_period: str, without: str, why: str):
        self.with_period = with_period
        self.without = without
        self.why = why


# 목표주가는 두 함수가 다른 것을 준다. get_company_evaluation 은 기간 인자가
# 없는 최근 2일자 스냅샷(목표가격 + 8개 점수), get_financial_data 는
# base_year/base_q 로 색인된 시계열 컨센서스. period 제약이 곧 판별자다.
PERIOD_SPLIT = PeriodSplit(
    with_period="get_financial_data",
    without="get_company_evaluation",
    why="기간이 지목되면 시계열 컨센서스, 아니면 최근 스냅샷 평가",
)

class PeriodGate:
    """기간이 지목돼야 호출이 성립하는 자리.

    issuer/price 인데 period 제약도 대상 엔티티도 없는 goal 은 "주가 좀 봐라"
    라는 빈 지시다(예: invest/2 g2). 기간 기본값을 지어내 시세 API 를 때리는
    대신 호출을 만들지 않는다. 주가 수준 판단은 get_company_evaluation 의
    주가모멘텀 점수가 이미 담고 있다.

    종목이 지목돼 있으면(예: "SK하이닉스 지금 주가") 기간이 없어도 현재가
    질의로 성립하므로 그대로 부른다. 빈 것은 대상이 없는 goal 뿐이다.
    """

    def __init__(self, with_period: str, why: str):
        self.with_period = with_period
        self.why = why

    def resolves(self, g) -> bool:
        return g.has_period or bool(g.ents)


PRICE_GATE = PeriodGate(
    "get_stock_price",
    "기간 제약 없는 시세 목표 — 독립 호출 없이 평가점수로 흡수",
)


# ─────────────────────────────────────────────────────────────
# 1층 BASE: (domain, facet) -> 주 함수
#
#   str          확정 함수
#   PeriodSplit  기간 제약이 판별자
#   PERIOD_REQUIRED  기간 없으면 호출 안 함
#   (A, B, …)    후보 다수 = 판별자 부족
#   None         대응 함수 없음
# ─────────────────────────────────────────────────────────────

TABLE: dict[tuple[str, str], object] = {
    # ── issuer: 대상 발행사가 지정됨 ──────────────────────────
    ("issuer", "profile"): "get_basis_data",
    ("issuer", "ipo"): "get_initial_listing",
    ("issuer", "price"): PRICE_GATE,
    ("issuer", "flow"): "get_stock_investor_trading",
    ("issuer", "short"): "get_stock_shorting_period",
    ("issuer", "fundamentals"): "get_financial_data",
    ("issuer", "valuation"): "get_stock_multiple_period",
    ("issuer", "estimate"): "get_financial_data",
    ("issuer", "target_price"): PERIOD_SPLIT,
    ("issuer", "scoring"): "get_company_evaluation",
    ("issuer", "news"): "get_stock_news",
    ("issuer", "disclosure"): "get_announcement",
    ("issuer", "sector_map"): "get_sector",
    ("issuer", "screening"): "search_top_stock",
    ("issuer", "fx"): "get_exchange",
    ("issuer", "knowledge"): "get_basic_financial_knowledge",
    ("issuer", "regulation"): "get_guide_and_policy",
    ("issuer", "howto"): "get_work_manual",
    # ── market: 대상 미지정 또는 시장 단위 ───────────────────
    ("market", "price"): "get_index_price",  # EXPANSION 이 업종지표로 뺄 수 있다
    ("market", "flow"): "get_index_investor_trading",
    ("market", "short"): "get_index_shorting_period",
    ("market", "valuation"): "get_index_multiple_period",
    # 기본 결과 단위는 종목. 업종·테마를 골라내는 질의면 3층이 대체한다.
    ("market", "screening"): "search_top_stock",
    ("market", "sector_map"): "get_sector",
    ("market", "ipo"): "search_top_stocks_by_event",  # metric='IPO' 일정 목록
    ("market", "fx"): "get_exchange",
    ("market", "knowledge"): "get_basic_financial_knowledge",
    ("market", "regulation"): "get_guide_and_policy",
    ("market", "howto"): "get_work_manual",
    # 뉴스는 두 코퍼스다. get_news 는 시장·거시·업황, get_stock_news 는 종목.
    # 대상이 시장이면 전자, 발행사면 후자가 주 함수다.
    ("market", "news"): "get_news",
    ("market", "fundamentals"): None,  # 시장·업종 합산 실적
    ("market", "estimate"): None,  # 시장·업종 실적 전망
    ("market", "target_price"): None,  # 지수 목표치
    ("market", "disclosure"): None,  # 시장 전체 공시 피드
    ("market", "scoring"): None,
    ("market", "profile"): None,  # 업종·테마 자체의 개요
    # ── internal: 자사 절차·앱 ───────────────────────────────
    ("internal", "howto"): "get_work_manual",
    ("internal", "ipo"): "get_ipo_subscription_allocation",
    # 자사 절차 문맥의 "규정" 질의는 매뉴얼이냐 규정이냐를 고르는 자리가
    # 아니다. 둘 다 필요하다 — 절차는 매뉴얼에, 그 절차를 강제하는 근거는
    # 규정집에 있다. 예전에 후보 튜플(=판별자 부족)로 두었던 것이 오독이었다.
    ("internal", "regulation"): "get_work_manual",  # + EXPANSION 이 정책을 붙임
    ("internal", "knowledge"): "get_basic_financial_knowledge",
    ("internal", "fx"): "get_exchange",
    # 자사 절차에서 "서류/요건"을 묻는 자리. 공시 피드가 아니라 규정집이다.
    ("internal", "disclosure"): "get_guide_and_policy",
    # ── finance_legal: 제도 자체 ─────────────────────────────
    ("finance_legal", "knowledge"): "get_basic_financial_knowledge",
    ("finance_legal", "regulation"): "get_guide_and_policy",
    ("finance_legal", "ipo"): ("get_guide_and_policy", "get_ipo_subscription_allocation"),
    # 제도 문맥의 "방법"은 자사 매뉴얼이 아니라 일반 투자 가이드다. 그리고
    # "고배당주 투자 유의점"처럼 개념 설명이 붙는 게 보통이라 둘 다 부른다.
    ("finance_legal", "howto"): "get_guide",
}

# facet 이 도메인과 안 맞게 나온 경우의 안전망. 답은 내되 파스 오류로 계상한다.
DOMAIN_FALLBACK = {
    "internal": "get_work_manual",
    "finance_legal": "get_basic_financial_knowledge",
}

# 파스만으로는 못 고르는 자리에서 무엇이 더 필요한지
DISCRIMINATOR = {
    ("finance_legal", "ipo"): "청약 제도 설명인지 청약 방법 안내인지",
}


# ─────────────────────────────────────────────────────────────
# 2층 EXPANSION: goal 의 모양이 조건인 팬아웃 간선
#
# 각 규칙은 (이름, 조건, 붙일 함수들, 사유). 조건은 goal 뷰 g 하나만 본다.
# replace=True 면 BASE 를 대체한다.
# ─────────────────────────────────────────────────────────────

# 지수 엔티티가 이것들이면 진짜 주가지수. 아니면 업황 지표(운임지수 등)다.
EQUITY_INDEXES = ("코스피", "코스닥", "kospi", "kosdaq", "s&p", "나스닥", "다우")

# 데이터가 아니라 설명을 요구하는 facet. 여기에는 업황·시세 채널이 붙지 않는다.
CONCEPTUAL_FACETS = ("knowledge", "regulation", "howto")

# 판단 목표(type=assessment)가 요구하는 증거 묶음.
#
# domain 이 먼저고 horizon 이 그다음이다. 이 순서가 중요하다 — 처음엔 horizon
# 만 봤는데, 그러면 "코스피 전망"과 "금융소득종합과세 영향"에 똑같이
# get_company_evaluation 을 붙인다. 평가점수는 발행사에만 있는 것이고, 지수에는
# 지수 함수가, 제도에는 가이드가 따로 있다. 증거의 종류는 판단의 대상이 정한다.
#
#   issuer  forward/current — 평가점수·목표가 + 컨센서스
#           past            — 이미 벌어진 일의 배경. 원인은 뉴스에 있다.
#   market  — 지수 수준·수급·시장 뉴스. market/index 엔티티가 있을 때만 (아래 게이트).
#   finance_legal — 제도의 결과를 묻는 것이므로 데이터가 아니라 가이드.
JUDGMENT_BUNDLE = {
    ("issuer", "forward"): ("get_company_evaluation", "get_financial_data"),
    ("issuer", "current"): ("get_company_evaluation", "get_financial_data"),
    ("issuer", "past"): ("get_stock_news",),
    ("market", "forward"): (
        "get_news",
        "get_index_price",
        "get_index_investor_trading",
    ),
    ("market", "current"): (
        "get_news",
        "get_index_price",
        "get_index_investor_trading",
    ),
    ("market", "past"): ("get_news",),
    ("finance_legal", "forward"): ("get_guide",),
    ("finance_legal", "current"): ("get_guide",),
    ("finance_legal", "past"): ("get_guide",),
}

# 업종 평균·타 종목과 견주는 판단은 전망 묶음이 아니라 상대가치가 근거다.
COMPARATIVE_BUNDLE = ("get_stock_multiple_period", "get_index_multiple_period")

# 지표를 골라달라는 목표(type=recommendation)는 바닥/과열 판단용 지표 세트를
# 통째로 요구한다. 어느 하나로 좁힐 수 없는 게 정상이다.
INDICATOR_BUNDLE = (
    "get_stock_price",
    "get_stock_multiple_period",
    "get_company_evaluation",
    "get_stock_news",
)


class Rule:
    def __init__(self, name, when, add=(), why="", replace=False, layer=EXPANSION):
        self.name = name
        self.when = when
        self.add = tuple(add)
        self.why = why
        self.replace = replace
        self.layer = layer

    def fire(self, g):
        return self.when(g)


def _is_equity_index(g):
    vals = g.ents.get("index", [])
    return any(any(k in v.lower() for k in EQUITY_INDEXES) for v in vals)


def _is_market_unit(g):
    """판단의 대상이 시장·지수 자체인가. market 판단 묶음의 게이트.

    "코스피 지수 향후 전망"은 지수 함수를 부르지만, "환율이 수출주에 미치는
    영향"은 같은 market/none/assessment 여도 지수를 부를 일이 아니다. 시장·주가
    지수 엔티티가 실제로 지목됐는지가 그 차이다.
    """
    if g.ents.get("metric"):
        # 지목된 지표가 있으면 그 지표 goal 이 따로 있다. 판단은 지수 수준만
        # 옆에 있으면 되고(그건 facet 규칙이 준다), 수급·뉴스까지 끌 일이 아니다.
        return False
    return bool(g.ents.get("market")) or _is_equity_index(g)


def _is_comparative(g):
    """업종 평균·타 대상과 견주는 목표인가."""
    return bool(g.ents.get("sector")) or bool(g.cons.get("scope"))


EXPANSIONS: list[Rule] = [
    # ── 업황 문맥 ────────────────────────────────────────────
    # theme / market_event 엔티티는 "이 질문은 개별 기업이 아니라 업황을
    # 경유한다"는 표시다. 다만 업황이 성립하는 것은 시세·기업 도메인에서다.
    # "해외주식 양도소득세"의 해외주식은 세제의 적용 범위지 업황이 아니고,
    # "미국 금리 인하의 영향 메커니즘"은 개념 설명이지 업종 조회가 아니다.
    Rule(
        "theme→sector",
        lambda g: g.domain in ("issuer", "market")
        and g.facet not in CONCEPTUAL_FACETS
        and g.has_any_ent("theme", "market_event"),
        add=["get_sector"],
        why="테마·업황 엔티티 — 업종 데이터",
    ),
    # 업황이 특정 종목에 닿는 질의일 때만 뉴스 두 갈래가 따라온다. 종목이
    # 지목됐거나(company) 질의 자체가 뉴스면 그렇다. "환율이 수출주에 미치는
    # 영향"처럼 종목이 없는 업황 질의는 섹터까지가 답이다.
    Rule(
        "themed issuer→news channel",
        lambda g: g.domain in ("issuer", "market")
        and g.has_any_ent("theme", "market_event")
        and (g.has_any_ent("company") or g.facet == "news"),
        add=["get_news", "get_stock_news"],
        why="업황이 종목에 닿는 질의 — 시장 뉴스 + 종목 뉴스",
    ),
    # market 도메인에서 대상이 업종·섹터·테마 단위로 지목되면 섹터 데이터가 필요하다.
    Rule(
        "sector-unit target→sector",
        lambda g: g.domain == "market" and g.text_has(SECTOR_UNIT_CUE),
        add=["get_sector"],
        why="대상 단위가 업종·섹터 — 섹터 구성/흐름 데이터",
        layer=LEXICON,
    ),
    # 시장 시세인데 지목된 지수가 주가지수가 아니면(운임지수·가격지수 등)
    # get_index_price 가 아니라 업황 지표다. 이때는 업황 채널이되 종목 뉴스는
    # 붙지 않는다 — 지수는 업종 자체를 가리키지 특정 종목을 가리키지 않는다.
    Rule(
        "non-equity index→industry channel",
        lambda g: g.key == ("market", "price")
        and g.has_any_ent("index")
        and not _is_equity_index(g),
        add=["get_sector", "get_news"],
        replace=True,
        why="주가지수가 아닌 업황 지표 — 지수 시세가 아니라 섹터·시장 뉴스",
    ),
    # ── 이벤트 스터디 ────────────────────────────────────────
    # 시세 goal 에 corporate_event 가 붙으면 "그 이벤트 전후의 주가"를 묻는
    # 것이다. 이벤트 시점은 공시에, 해석은 뉴스에 있다.
    Rule(
        "corporate_event on price→announcement+news",
        lambda g: g.facet == "price" and g.has_any_ent("corporate_event"),
        add=["get_announcement", "get_stock_news"],
        why="기업 이벤트 기준 시세 분석 — 공시로 시점 확정, 뉴스로 해석",
    ),
    # ── 판단 목표 ────────────────────────────────────────────
    # 이벤트 효과 판단은 밸류에이션 예측이 아니다. 평가점수·컨센서스가 아니라
    # 이벤트 자체의 사례/반응을 본다. 아래 assessment bundle 과는 조건이
    # 배타적이라(corporate_event 유무) 둘 중 하나만 발화한다.
    Rule(
        "event assessment",
        lambda g: g.type == "assessment" and g.has_any_ent("corporate_event"),
        add=["get_stock_news"],
        why="이벤트 효과 판단 — 전망 묶음이 아니라 이벤트 반응 사례",
    ),
    # 업종 평균 대비 같은 상대 비교 판단은 전망이 아니라 멀티플 비교다.
    Rule(
        "comparative assessment",
        lambda g: g.type == "assessment" and _is_comparative(g),
        add=COMPARATIVE_BUNDLE,
        why="상대 비교 판단 — 종목 멀티플 대 업종 멀티플",
    ),
    # 붙일 함수가 (domain, horizon) 에 따라 달라지는 유일한 규칙. add 를 비워
    # 두고 route() 가 judgment_bundle(g) 로 채운다. 위 두 규칙(이벤트·상대비교)
    # 과는 조건이 배타적이라 셋 중 하나만 발화한다.
    Rule(
        "assessment bundle",
        lambda g: g.type == "assessment"
        and not g.has_any_ent("corporate_event")
        and not _is_comparative(g)
        # market 판단은 대상이 시장·지수 자체일 때만 지수 함수를 부른다.
        and (g.domain != "market" or _is_market_unit(g)),
        add=(),
        why="판단 목표 — (domain, horizon) 별 증거 묶음",
    ),
    # 외부 지수 대비 판단(운임지수 반영 주가 전망 등)은 종목 주가 자체가
    # 비교 대상이라 시세가 필요하다. 발행사 판단일 때만이다.
    Rule(
        "index-relative assessment→price",
        lambda g: g.type == "assessment"
        and g.domain == "issuer"
        and g.has_any_ent("index"),
        add=["get_stock_price"],
        why="외부 지수 대비 판단 — 종목 주가 추이가 비교축",
    ),
    # ── 시장 데이터 목표의 지수 동반 ─────────────────────────
    # 수급·공매도·멀티플은 지수 수준을 옆에 놓아야 읽힌다. "공매도 비중이
    # 높은 편인가"는 지수가 어디 있는지 모르면 답이 안 된다.
    Rule(
        "market data→index price",
        lambda g: g.domain == "market"
        and g.facet in ("flow", "short", "valuation"),
        add=["get_index_price"],
        why="시장 데이터 해석에는 지수 수준이 기준선",
    ),
    Rule(
        "recommendation bundle",
        lambda g: g.type == "recommendation",
        add=INDICATOR_BUNDLE,
        why="지표 선정 목표 — 시세·멀티플·평가점수·뉴스 세트가 통째로 필요",
    ),
    # ── 자사 절차의 동반 코퍼스 ──────────────────────────────
    # 제출서류를 묻는 순간 그것은 앱 사용법이 아니라 법정 요건이다.
    Rule(
        "document→policy",
        lambda g: g.domain == "internal" and g.has_any_ent("document"),
        add=["get_guide_and_policy"],
        why="서류 요건은 실명확인·법정 요건 — 규정집 동반",
    ),
    # 상품 엔티티(연금저축·IRP·ISA…)가 붙으면 절차나 세제만으로는 답이 안 되고
    # 상품 자체의 구조 설명이 따라붙는다. 자사 절차든 제도든 마찬가지다.
    Rule(
        "product→knowledge",
        lambda g: g.domain in ("internal", "finance_legal")
        and g.has_any_ent("product"),
        add=["get_basic_financial_knowledge"],
        why="상품 엔티티 — 절차·세제 + 상품 구조 설명",
    ),
    Rule(
        "internal regulation→policy",
        lambda g: g.key == ("internal", "regulation"),
        add=["get_guide_and_policy"],
        why="자사 절차 + 그 절차를 강제하는 규정 근거",
    ),
    # 제도상의 "방법"은 가이드만으로 끝나지 않는다. 왜 그렇게 해야 하는지의
    # 개념 설명이 늘 함께 필요하다.
    Rule(
        "finance_legal howto→knowledge",
        lambda g: g.key == ("finance_legal", "howto"),
        add=["get_basic_financial_knowledge"],
        why="제도상 방법 안내 + 그 근거 개념 설명",
    ),
]


# ─────────────────────────────────────────────────────────────
# 3층 LEXICON: 구조 신호가 없어 값 문자열에 기대는 자리
#
# 여기 걸리는 goal 은 파서가 판별자를 안 내줬다는 뜻이다. 리포트에서 따로
# 세고, 이 층에만 의존해 붙은 함수는 별도 표시한다.
# ─────────────────────────────────────────────────────────────

# 규제가 절차를 규정하는 계좌·행위. 매뉴얼만으로는 답이 반쪽이다.
REGULATED_SUBJECT = (
    "휴면", "한도제한", "해지", "미성년", "실명", "자금세탁", "예금자보호",
)
# 절차 완료 "이후"를 묻는 자리 — 운용 가이드 코퍼스.
NEXT_STEP_CUE = ("다음 단계", "이후 절차", "후속 절차")
# 방법이 아니라 필요 여부를 묻는 자리 — 규정 판단.
ELIGIBILITY_CUE = ("필요성", "필요 여부", "가능 여부", "해야 하나")
# 결과 단위가 종목이 아니라 업종·섹터·테마인 자리. screening 의 판별자다.
SECTOR_UNIT_CUE = ("업종", "섹터", "테마")
# 상장 이벤트로 거르는 스크리닝. 상장 제원 자체도 함께 필요하다.
EVENT_SCREEN_CUE = ("상장", "공모가")
# 공모주 청약·배정 제도. 도메인과 무관하게 이 코퍼스가 필요하다.
# "공모가"(상장 제원)와 "공모주식수"(발행 제원)는 청약 제도가 아니다. 맨
# "공모주"를 큐에 넣으면 "공모주식수"에 걸리므로 넣지 않는다 — 실제 청약
# 질의에는 청약·배정 어느 쪽이든 반드시 등장한다.
IPO_SUBSCRIPTION_CUE = ("청약", "균등배정", "비례배정")
# 환율은 facet=fx 가 아닌 자리에도 엔티티로만 등장한다.
FX_CUE = ("환율",)
# 배당. 배당금·배당수익률 수치는 재무 데이터에, 배당 발표는 공시에 있다.
DIVIDEND_CUE = ("배당",)
ANNOUNCEMENT_CUE = ("발표", "공시")

LEXICON_RULES: list[Rule] = [
    Rule(
        "regulated subject→policy",
        lambda g: g.domain == "internal" and g.text_has(REGULATED_SUBJECT),
        add=["get_guide_and_policy"],
        why="규제가 절차를 규정하는 대상 — 근거 규정 동반",
        layer=LEXICON,
    ),
    Rule(
        "next-step→guide",
        lambda g: g.domain == "internal" and g.text_has(NEXT_STEP_CUE),
        add=["get_guide"],
        why="절차 완료 이후 운용 안내 — 투자 가이드 코퍼스",
        layer=LEXICON,
    ),
    Rule(
        "eligibility→policy",
        lambda g: g.domain == "internal" and g.text_has(ELIGIBILITY_CUE),
        add=["get_guide_and_policy"],
        why="방법이 아니라 필요 여부 질의 — 규정 판단",
        layer=LEXICON,
    ),
    # ── 스크리닝의 결과 단위 ─────────────────────────────────
    # market/screening 은 오래 AMBIGUOUS 로 두었던 자리다. 판별자는 있었다 —
    # 골라낼 것이 종목인지 업종인지. 그건 대상 문구에 적혀 있다.
    Rule(
        "sector-unit screening",
        lambda g: g.facet == "screening" and g.text_has(SECTOR_UNIT_CUE),
        add=["search_top_sector_theme", "get_sector", "get_news"],
        replace=True,
        why="결과 단위가 업종·테마 — 섹터 랭킹 + 구성 + 배경 뉴스",
        layer=LEXICON,
    ),
    # 상장 이벤트로 거르는 스크리닝은 이벤트 랭킹과 상장 제원이 함께 필요하다.
    Rule(
        "event screening→listing",
        lambda g: g.facet == "screening" and g.text_has(EVENT_SCREEN_CUE),
        add=["search_top_stocks_by_event", "get_initial_listing"],
        why="상장 이벤트 기준 스크리닝 — 이벤트 랭킹 + 상장 제원",
        layer=LEXICON,
    ),
    # ── 청약 제도 ────────────────────────────────────────────
    # 청약·배정은 별도 코퍼스다. 앱에서 하는 법을 묻든(internal), 제도를
    # 묻든(finance_legal), 일정을 묻든(market) 같은 곳을 봐야 한다.
    Rule(
        "ipo subscription",
        lambda g: g.text_has(IPO_SUBSCRIPTION_CUE),
        add=["get_ipo_subscription_allocation"],
        why="공모주 청약·배정 — 전용 코퍼스",
        layer=LEXICON,
    ),
    # ── 환율 ────────────────────────────────────────────────
    # facet=fx 가 아니어도 환율이 엔티티로 등장하면 환율 데이터가 필요하다.
    Rule(
        "fx entity→exchange",
        lambda g: g.facet != "fx" and g.text_has(FX_CUE),
        add=["get_exchange"],
        why="환율이 판단의 축으로 등장 — 환율 데이터",
        layer=LEXICON,
    ),
    # ── 배당 ────────────────────────────────────────────────
    # 배당금·배당수익률 수치는 재무 데이터에 있다. 발행사 질의일 때만이다.
    Rule(
        "dividend→financial data",
        lambda g: g.domain == "issuer" and g.text_has(DIVIDEND_CUE),
        add=["get_financial_data"],
        why="배당금·배당수익률 수치는 재무 데이터",
        layer=LEXICON,
    ),
    # 발표·공시로 걸러진 질의는 공시 피드가 출처다.
    Rule(
        "announcement cue",
        lambda g: g.facet != "disclosure" and g.text_has(ANNOUNCEMENT_CUE),
        add=["get_announcement"],
        why="발표·공시 기준 질의 — 공시 피드",
        layer=LEXICON,
    ),
]


# 2·3층 전체. 순서가 의미를 갖는다 — replace=True 규칙이 앞선 결과를 지운다.
ALL_RULES: list[Rule] = EXPANSIONS + LEXICON_RULES


# 기간 파라미터를 받는 함수: period 제약이 없으면 기본값 결정이 필요하다
NEEDS_PERIOD = {
    "get_stock_price",
    "get_index_price",
    "get_stock_investor_trading",
    "get_index_investor_trading",
    "get_stock_shorting_period",
    "get_index_shorting_period",
    "get_stock_multiple_period",
    "get_index_multiple_period",
    "get_stock_news",
    "get_news",
    "get_announcement",
    "search_top_stock",
    "search_top_stocks_by_event",
    "search_top_sector_theme",
}

CATALOG = {
    "get_basis_data", "get_initial_listing", "get_financial_data", "get_stock_price",
    "get_stock_investor_trading", "get_index_investor_trading",
    "get_index_shorting_period", "get_stock_multiple_period",
    "get_index_multiple_period", "get_guide_and_policy",
    "get_basic_financial_knowledge", "get_guide", "search_top_stocks_by_event",
    "search_top_stock", "search_top_sector_theme", "get_index_price",
    "get_stock_shorting_period", "get_ipo_subscription_allocation",
    "get_company_evaluation", "get_stock_news", "get_news", "get_announcement",
    "get_sector", "get_exchange", "get_work_manual",
}


# ─────────────────────────────────────────────────────────────
# goal 뷰: 규칙이 보는 것 전부
# ─────────────────────────────────────────────────────────────

class Goal:
    def __init__(self, raw: dict, entities: list[dict], constraints: list[dict]):
        self.raw = raw
        self.id = raw["id"]
        self.domain = raw["domain"]
        self.facet = raw["facet"]
        self.type = raw["type"]
        self.horizon = raw.get("horizon", "current")
        self.target = raw.get("target", "")
        self.key = (self.domain, self.facet)

        self.ents = defaultdict(list)
        for e in entities:
            if e["goal_id"] == self.id:
                self.ents[e["type"]].append(e["value"])
        self.cons = defaultdict(list)
        for c in constraints:
            if c["goal_id"] == self.id:
                self.cons[c["type"]].append(c["value"])
        self.has_period = bool(self.cons.get("period"))

    def has_any_ent(self, *types) -> bool:
        return any(self.ents.get(t) for t in types)

    def text_has(self, cues) -> bool:
        """target + 엔티티/제약 값 전체를 한 덩어리로 보고 키워드 검사."""
        blob = " ".join(
            [self.target]
            + [v for vs in self.ents.values() for v in vs]
            + [v for vs in self.cons.values() for v in vs]
        )
        return any(c in blob for c in cues)

    def __str__(self):
        return f"{self.id}[{self.domain}/{self.facet}/{self.type}/{self.horizon}]"


def judgment_bundle(g: Goal) -> tuple[str, ...]:
    return JUDGMENT_BUNDLE.get((g.domain, g.horizon), ())


def route(g: Goal, rules: list | None = None):
    """goal -> (상태, {함수: (층, 사유)}, 사유, 후보들)

    후보들은 AMBIGUOUS 일 때만 채워진다.

    rules 를 주면 2·3층을 그 부분집합으로만 돈다. ab_fit.py 가 규칙 선택을
    기계적으로 하기 위해 쓴다 — 규칙을 손으로 켜고 끄면 그게 곧 골든 전체에
    맞추는 것이고, 그러면 holdout 이 holdout 이 아니게 된다.
    """
    picked: dict[str, tuple[str, str]] = {}
    status, note, candidates = RESOLVED, "", []

    # ── 1층 ──
    if g.key in TABLE:
        target = TABLE[g.key]
        if target is None:
            status, note = UNMAPPED, "대응 함수가 카탈로그에 없음"
        elif isinstance(target, PeriodGate):
            if target.resolves(g):
                picked[target.with_period] = (BASE, "")
            else:
                status, note = SUBSUMED, target.why
        elif isinstance(target, PeriodSplit):
            fn = target.with_period if g.has_period else target.without
            picked[fn] = (BASE, target.why)
        elif isinstance(target, tuple):
            status, candidates = AMBIGUOUS, list(target)
            note = DISCRIMINATOR.get(g.key, "판별자 미정의")
        else:
            picked[target] = (BASE, "")
    elif g.facet == "none":
        # facet=none 은 순수 판단 목표의 자리다. 1층에 자리가 없는 게 정상이고
        # 2층 판단 묶음이 채운다.
        if g.type not in ("assessment", "recommendation"):
            status = UNMAPPED
            note = f"type={g.type} 인데 facet=none — 분류 실패"
    else:
        fb = DOMAIN_FALLBACK.get(g.domain)
        if fb:
            status = FALLBACK
            picked[fb] = (BASE, f"{g.key} 는 테이블에 없음 — 도메인 기본 함수로")
            note = f"{g.key} 는 테이블에 없음 — 도메인 기본 함수로"
        else:
            status, note = UNMAPPED, f"테이블에 {g.key} 항목 없음"

    if status in (UNMAPPED, AMBIGUOUS):
        return status, picked, note, candidates

    # ── 2층 + 3층 ──
    for rule in ALL_RULES if rules is None else rules:
        if not rule.fire(g):
            continue
        add = judgment_bundle(g) if rule.name == "assessment bundle" else rule.add
        if not add:
            continue
        if rule.replace:
            picked = {}
        for fn in add:
            picked.setdefault(fn, (rule.layer, rule.why or rule.name))

    if picked:
        status = FALLBACK if status == FALLBACK else RESOLVED
    elif status == RESOLVED:
        status = SUBSUMED
        note = note or "호출 없이 상류 goal 결과로 합성"

    return status, picked, note, candidates


def goals_of(rec: dict) -> list[Goal]:
    return [Goal(raw, rec["entities"], rec["constraints"]) for raw in rec["goals"]]


def predict(rec: dict, rules: list | None = None) -> set[str]:
    """파스 레코드 하나 -> 이 발화가 호출할 함수 집합.

    goal 단위가 아니라 발화 단위인 게 요점이다. 여러 goal 이 같은 함수를
    가리키면 한 번만 부르고, 판단 goal 이 요구한 증거가 이미 다른 goal 로
    확보돼 있으면 추가 호출이 생기지 않는다.
    """
    fns = set()
    for g in goals_of(rec):
        fns |= set(route(g, rules)[1])
    return fns


# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────

def load(path: str) -> list[tuple[str, dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [
            (r.get("idx", ""), json.loads(r["json"]))
            for r in csv.DictReader(f)
            if r.get("json")
        ]


def load_golden(path: str) -> dict[tuple[str, int], tuple[str, set[str]]]:
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fns = {x for x in r["functions"].split(";") if x}
            out[(r["source"], int(r["idx"]))] = (r["query"], fns)
    return out


def source_of(path: str) -> str:
    return path.split("/")[-1].split("\\")[-1].split("_out")[0]


def discover(golden_path: str) -> list[str]:
    """골든의 source 목록에서 산출물이 실제로 있는 *_out.csv 만 모은다."""
    try:
        with open(golden_path, encoding="utf-8-sig", newline="") as f:
            sources = list(dict.fromkeys(r["source"] for r in csv.DictReader(f)))
    except FileNotFoundError:
        return []
    return [s + "_out.csv" for s in sources if os.path.exists(s + "_out.csv")]


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    golden_path = "golden_labels.csv"
    paths = []
    it = iter(argv)
    for a in it:
        if a == "--golden":
            golden_path = next(it, golden_path)
        elif a == "--no-golden":
            golden_path = None
        else:
            paths.append(a)
    paths = paths or discover(golden_path or "golden_labels.csv")
    if not paths:
        print("[FAIL] 채점할 *_out.csv 를 찾지 못했습니다.")
        return 1

    try:
        golden = load_golden(golden_path) if golden_path else {}
    except FileNotFoundError:
        golden = {}

    status_n = Counter()
    fn_n = Counter()
    layer_n = Counter()
    holes = Counter()
    ambig = Counter()
    period_gap = Counter()
    lexicon_hits = []
    queries = []  # (source, idx, [(goal, status, picked, note)])

    for path in paths:
        src = source_of(path)
        for idx, rec in load(path):
            per_goal = []
            for g in goals_of(rec):
                status, picked, note, cands = route(g)
                status_n[status] += 1
                per_goal.append((g, status, picked, note, cands))
                for fn, (layer, _) in picked.items():
                    fn_n[fn] += 1
                    layer_n[layer] += 1
                    if layer == LEXICON:
                        lexicon_hits.append((src, idx, str(g), fn))
                    if fn in NEEDS_PERIOD and not g.has_period:
                        period_gap[fn] += 1
                if status == UNMAPPED:
                    holes[g.key] += 1
                elif status == AMBIGUOUS:
                    ambig[g.key] += 1
            queries.append((src, idx, per_goal))

    total = sum(status_n.values())
    called = total - status_n[SUBSUMED]

    print(f"goal {total}개 (호출 대상 {called}, 흡수 {status_n[SUBSUMED]})\n")

    print("── 상태별 ─────────────────────────────")
    for s in (RESOLVED, FALLBACK, AMBIGUOUS, UNMAPPED, SUBSUMED):
        n = status_n[s]
        share = f"{n / called:6.1%}" if called and s != SUBSUMED else "     -"
        print(f"  {s:<10} {n:>4}   호출대상 대비 {share}")

    print("\n── 층별 호출 근거 ─────────────────────")
    tot_calls = sum(layer_n.values()) or 1
    for layer in (BASE, EXPANSION, LEXICON):
        print(f"  {layer:<10} {layer_n[layer]:>4}   {layer_n[layer] / tot_calls:6.1%}")

    print("\n── 확정된 함수 ────────────────────────")
    for fn, n in fn_n.most_common():
        print(f"  {n:>3}  {fn}")

    print("\n── 구멍: 대응 함수 없음 ───────────────")
    print("  (없음)" if not holes else "", end="")
    for (d, f), n in holes.most_common():
        print(f"  {n:>3}  {d}/{f}")

    print("\n── 판별자 부족 ────────────────────────")
    print("  (없음)" if not ambig else "", end="")
    for (d, f), n in ambig.most_common():
        print(f"  {n:>3}  {d}/{f}")
        print(f"       후보: {', '.join(TABLE[(d, f)])}")
        print(f"       필요: {DISCRIMINATOR.get((d, f), '?')}")

    print("\n── 기간 파라미터 미지정 (기본값 규칙 필요) ──")
    print("  (없음)" if not period_gap else "", end="")
    for fn, n in period_gap.most_common():
        print(f"  {n:>3}  {fn}")

    print("\n── 3층(문자열 휴리스틱)에 기댄 호출 ───")
    print("  (없음)" if not lexicon_hits else "", end="")
    for src, idx, gs, fn in lexicon_hits:
        print(f"  {src}/{idx} {gs} → {fn}")

    print("\n── 스모크에서 한 번도 안 쓰인 함수 ────")
    print("  " + ", ".join(sorted(CATALOG - set(fn_n))))

    if golden:
        score(queries, golden)
    return 0


def score(queries, golden):
    print("\n── golden_labels 대조 ─────────────────")
    exact = tp = fp = fn_ = 0
    diffs = []
    scored = 0
    for src, idx, per_goal in queries:
        gkey = (src, int(idx))
        if gkey not in golden:
            continue
        scored += 1
        query, want = golden[gkey]
        got = set()
        for g, status, picked, note, cands in per_goal:
            got |= set(picked)
        miss, extra = want - got, got - want
        tp += len(want & got)
        fn_ += len(miss)
        fp += len(extra)
        if not miss and not extra:
            exact += 1
        else:
            diffs.append((src, idx, query, per_goal, miss, extra))

    # 채점 대상 밖인 두 방향을 다 밝혀 둔다. 골든만 있고 파스가 없는 행과,
    # 파스만 있고 골든이 없는 행. 후자는 커버리지 리포트에는 잡히지만 P/R/F1
    # 에는 안 잡히므로, 점수가 전체를 대표하는 것처럼 읽히면 안 된다.
    unscored = Counter(src for (src, _) in golden) - Counter(
        src for src, idx, _ in queries if (src, int(idx)) in golden
    )
    unlabeled = Counter(
        src for src, idx, _ in queries if (src, int(idx)) not in golden
    )

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"  질의 완전일치  {exact}/{scored}  ({exact / max(scored, 1):.1%})")
    print(f"  함수 P/R/F1    {prec:.3f} / {rec:.3f} / {f1:.3f}")
    if unscored:
        tot = sum(unscored.values())
        detail = ", ".join(f"{s}×{n}" for s, n in sorted(unscored.items()))
        print(f"  채점 제외      {tot}행 — 파스 산출물(*_out.csv) 없음: {detail}")
    if unlabeled:
        tot = sum(unlabeled.values())
        detail = ", ".join(f"{s}×{n}" for s, n in sorted(unlabeled.items()))
        print(f"  채점 제외      {tot}행 — 골든 라벨 없음: {detail}")

    print("\n── 불일치 상세 ────────────────────────")
    print("  (없음)" if not diffs else "", end="")
    for src, idx, query, per_goal, miss, extra in diffs:
        print(f"  {src}/{idx} {query}")
        for g, status, picked, note, cands in per_goal:
            fns = ", ".join(f"{f}({l[0]})" for f, (l, _) in picked.items()) or "-"
            print(f"      {g} {status:<9} {fns}")
        if miss:
            print(f"      MISS  {sorted(miss)}")
        if extra:
            print(f"      EXTRA {sorted(extra)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
