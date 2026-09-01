"""arm C 용 라우터: 의도(intent) + 엔티티 -> 함수 집합.

A/B 는 goal 을 (domain, type, facet, horizon) 으로 분해해 capability graph 에
넣는다. C 는 그 분해를 LLM 에게 시키지 않는다. 대신 24개 intent 택소노미 중
해당하는 것을 고르게 하고, 엔티티만 뽑게 한다.

여기서 하는 일은 intent 를 capability graph 가 알아듣는 (domain, facet, type)
으로 번역하는 것뿐이다. 그러면 route.py 의 2·3층 규칙이 그대로 돌아간다 —
C 를 위해 그래프를 새로 만들지 않는다. 그래야 A 대 C 비교가 프롬프트 차이만
재는 비교가 된다.

domain 이 intent 만으로 안 정해지는 자리가 있다. "주가 및 거래내역"은 삼성전자
면 issuer, 코스피면 market 이다. 그건 엔티티가 정하므로 AUTO 로 두고 런타임에
푼다 — LLM 에게 물을 일이 아니다.
"""

import csv

from route import Goal, route

AUTO = "auto"  # 대상 엔티티가 domain 을 정한다

# intent_id -> (domain, facet, type)
#
# facet 은 택소노미의 소분류가 곧 답이다. type 은 소분류 이름에 이미 들어 있는
# 것만 넣었다 — "전망", "분석", "평가"는 판단 목표고, 나머지는 조회다. 이걸
# 넣는 이유는 assessment 묶음이 F1 에 0.08 을 쥐고 있어서다(절삭 실험).
INTENT_MAP: dict[str, tuple[str, str, str]] = {
    # 기본적인 금융 지식
    "I1-1": ("finance_legal", "knowledge", "explanation"),
    "I1-2": ("finance_legal", "knowledge", "explanation"),
    # 규정 및 지침
    "I2-1": ("finance_legal", "regulation", "query"),
    "I2-2": ("finance_legal", "regulation", "query"),
    "I2-3": ("finance_legal", "howto", "explanation"),
    # 사용 매뉴얼
    "I3-1": ("internal", "howto", "explanation"),
    "I3-2": ("internal", "howto", "explanation"),
    # IPO
    "I4-1": ("issuer", "ipo", "query"),
    "I4-2": ("market", "ipo", "query"),
    # 시장 동향 및 분석
    "I5-1": ("market", "price", "assessment"),  # 시장 전망
    "I5-2": ("market", "news", "analysis"),  # 산업 동향 분석
    # 기업정보
    "I6-1": ("issuer", "profile", "query"),
    "I6-2": ("issuer", "profile", "query"),
    "I6-3": ("issuer", "news", "query"),
    "I6-4": ("issuer", "disclosure", "query"),
    "I6-5": ("issuer", "fundamentals", "query"),  # 배당 — 3층이 재무로 보낸다
    "I6-6": ("issuer", "fundamentals", "query"),
    "I6-7": ("issuer", "fundamentals", "analysis"),  # 기업 분석
    "I6-8": ("issuer", "scoring", "assessment"),  # 기업 평가 및 전망
    # 시세 정보
    "I7-1": (AUTO, "price", "query"),
    "I7-2": (AUTO, "price", "analysis"),
    # 평가 및 밸류에이션
    "I8-1": (AUTO, "valuation", "query"),
    # 컨센서스는 "증권가 추정치" 조회지 판단이 아니다. assessment 로 두면
    # 판단 묶음이 붙어 get_company_evaluation 을 과호출한다.
    "I8-2": ("issuer", "estimate", "query"),
    # 기타
    "I9-1": ("market", "fx", "query"),
}

# 수급을 묻는 intent 는 facet=price 로 뭉뚱그려 있다. investor_group 엔티티가
# 있으면 flow 가, 공매도면 short 가 맞다. 택소노미가 셋을 한 칸에 넣어둔 탓이라
# 엔티티로 되돌린다.
FLOW_HINT = ("외국인", "기관", "개인", "수급")
SHORT_HINT = ("공매도", "대차", "숏")


def _domain_of(ents: list[dict]) -> str:
    """대상 엔티티가 발행사면 issuer, 시장·지수면 market."""
    types = {e["type"] for e in ents}
    if "company" in types:
        return "issuer"
    if types & {"market", "index"}:
        return "market"
    return "market"  # 대상 미지정 = 시장 단위


def _refine_facet(facet: str, ents: list[dict]) -> str:
    if facet != "price":
        return facet
    blob = " ".join(e["value"] for e in ents)
    if any(k in blob for k in SHORT_HINT):
        return "short"
    if any(k in blob for k in FLOW_HINT):
        return "flow"
    return "price"


def to_goals(rec: dict) -> list[Goal]:
    """{intents, entities, constraints} -> capability graph 가 먹는 Goal 목록."""
    out = []
    for n, iid in enumerate(rec["intents"], 1):
        if iid not in INTENT_MAP:
            continue
        gid = f"g{n}"
        ents = [e for e in rec["entities"] if e.get("intent_id", iid) == iid]
        cons = [c for c in rec["constraints"] if c.get("intent_id", iid) == iid]
        domain, facet, gtype = INTENT_MAP[iid]
        if domain == AUTO:
            domain = _domain_of(ents)
        facet = _refine_facet(facet, ents)
        raw = {
            "id": gid,
            "target": rec.get("subject") or rec.get("target_blob", ""),
            "domain": domain,
            "facet": facet,
            "type": gtype,
            "horizon": "current",
        }
        out.append(
            Goal(
                raw,
                [{**e, "goal_id": gid} for e in ents],
                [{**c, "goal_id": gid} for c in cons],
            )
        )
    return out


def predict(rec: dict, rules: list | None = None) -> set[str]:
    fns = set()
    for g in to_goals(rec):
        fns |= set(route(g, rules)[1])
    return fns


def load_taxonomy(path: str = "intent_taxonomy.csv") -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["intent_id"]] = f"{r['대분류']}>{r['소분류']} — {r['설명']}"
    return out
