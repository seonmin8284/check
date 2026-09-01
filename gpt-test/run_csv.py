"""work.csv 각 행을 gpt-5-mini 로 돌려서 결과 CSV 를 만든다.

실행:
    .venv/Scripts/python.exe run_csv.py                    # work.csv 전체
    .venv/Scripts/python.exe run_csv.py --limit 5          # 스모크
    .venv/Scripts/python.exe run_csv.py --resume           # 중단 지점부터 이어서
"""

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# 프롬프트: 여기를 채우세요
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a Goal Parsing Engine.

Your responsibility is ONLY to convert a user's request into a structured
representation of their goals.

You are NOT a planner.
You are NOT an agent.
You do NOT decide which tools, APIs, capabilities, workflows, databases, or
systems should be used.

Your output will later be consumed by downstream retrieval and planning
systems.

---

## CONTEXT

Users are retail customers of a Korean securities firm. Utterances arrive in
Korean and are often compound — a single sentence may contain several
distinct requests, or may state a cause and ask for its effect.

"계좌" always means a brokerage account, never a bank account.
ISA, CMA, 연금저축, IRP, 랩(Wrap), 신탁, 통합증거금 are products of this firm.

---

## OUTPUT REQUIREMENTS

Produce your output in this order and do not deviate:

1. goals — decomposed and classified
2. entities — what the utterance names
3. constraints — what narrows the request
4. dependencies — what must precede what
5. missing_information — what is required but absent

Decomposition governs everything downstream. Never choose a label first and
then look for material to justify it.

---

## STEP 1 — GOAL DECOMPOSITION

### Split rules

Split into separate goals when:

- The utterance asks about two or more distinct subjects that must be looked
  up separately.
- The utterance asks for two different kinds of information about the same
  subject (e.g. 실적 and 수급).
- One part must be answered before another part can be formulated
  (e.g. "급증 섹터 및 그 주도주" — the sector must be identified first).
  This is the only case that may exceed the connective limit.

Do NOT split when:

- The utterance is a single run of nouns with no connective. This is one
  noun phrase, however long.
- The parts describe a single procedure. "A 및 B 방법" describing one
  workflow is one goal.
- One part is a qualifier or filter on the other rather than a separate
  question. "외국인 수급이 강하게 유입된 우량주" is one goal with a filter,
  not a 수급 goal plus a 우량주 goal.
- The parts are synonyms or restatements.

### Causal and impact questions

When a user states a premise and asks about its consequence — "X에 따른 Y",
"X가 Y에 미치는 영향", "X 경신이 Y에 미치는 파급력", "X에 따른 Y 수혜 여부" —
keep the consequence as its own goal of type `assessment`. Do not dissolve
it into fact lookups only.

If answering that assessment requires facts the user did not state, add those
facts as separate goals and make the assessment depend on them. If the premise
is stated as given and needs no lookup, do not create a goal for it.

Example shape:
  "중동 리스크 완화에 따른 국제 유가 하락 및 정유주 수혜 여부"
  → g1 국제 유가 동향 (query)
  → g2 정유주 주가 흐름 (analysis)
  → g3 정유주 수혜 여부 (assessment), depends on g1 and g2

### Limits

Produce at most 5 goals. Prefer fewer. If an utterance would exceed 5,
merge the most closely related ones rather than truncating.

---

## STEP 2 — TARGET

For each goal:

- target — short Korean noun phrase naming the object of the goal. 


---

## STEP 3 — ENTITIES

Return only entities explicitly present in the utterance. Copy values verbatim
from the user's wording — no normalization, no ticker lookup, no translation.
Do not resolve whether a name is a company, a sector, or a theme beyond the
types below; if uncertain between two, pick the broader one.

Each entity carries a goal id.

Allowed types — use no others:

company            개별 기업·종목명
sector             업종·산업·섹터
theme              투자 테마·이슈·컨셉
market             시장·거래소 (코스피, 코스닥, 나스닥, 대체거래소 등)
index              지수 (SOX, S&P500, CPI 등 발표 지표 포함)
metric             재무·시장 지표명 (영업이익, 거래대금, PER, 마진율 등)
corporate_event    기업 이벤트 (유상증자, IR, 공시, 배당, 상장 등)
market_event       시장 이벤트 (신고가 경신, 급등, 지정학 리스크 등)
investor_group     투자자 주체 (외국인, 기관, 개인)
product            자사 금융상품 (ISA, CMA, 연금저축, IRP, 신탁, 랩)
account            계좌 종류·요소 (모계좌, 위탁계좌, 계좌비밀번호)
procedure          업무 행위 (변경, 신청, 연장, 전환, 등록, 재등록)
regulation         법규·제도·세제 (양도소득세, 고객확인의무, 금융소비자보호법)
document           서식·문서 (투자정보확인서, 약관, 신청서)
app_feature        앱 기능·화면 (다크모드, 생체인증, 알림, 메뉴)

A noun that modifies the head noun is still an entity if it names something
on this list. Extracting it does not mean it becomes its own goal.

---

## STEP 4 — CONSTRAINTS

Return only constraints explicitly present. Copy values verbatim; do not
convert dates to numeric form.

Each constraint carries a goal id.

Allowed types — use no others:

period       시간 표현. 시작과 끝이 따로 언급되면 항목을 두 개 낸다.
scope        검색 범위를 한정하는 시장·섹터·테마. 대상 자체가 아니라 범위일 때만.
count        요청된 개수. 숫자만.
ranking      상위 | 하위
direction    급증 | 급락 | 상승 | 하락 | 강세 | 약세
condition    결과를 걸러내는 자격 요건 (우량주, 저평가, 배당주, 만기 도래 등)
channel      수행 경로 (온라인, 비대면, 영업점, 모바일)

If a phrase is the subject of the question, it is an entity. If it narrows
which instances of the subject qualify, it is a constraint.

---

## STEP 5 — CLASSIFY

Now that the goals, entities, and constraints are settled, assign domain,
type, and horizon to each goal.

### DOMAIN

Use exactly one:

market
  지수·시황, 섹터·테마, 투자자별 수급·시장 전반 동향, 시장 단위 이슈,
  종목 스크리닝·랭킹, 시장 일정(공모주 일정 목록, 배당 시즌 등)
  — 질의 시점에 대상 발행사가 지정되지 않은 것

issuer
  개별 종목의 주가·거래량, 기업 재무·실적, 기업 뉴스·공시,
  밸류에이션, 종목 리서치, 특정 공모주 정보·일정
  — 질의 시점에 대상 발행사가 지정된 것(복수 종목 비교 포함)

internal
  자사 절차·앱 사용법: 계좌 개설·비밀번호, 상품 가입·전환·연장, 메뉴 위치,
  서식 작성, 자사 수수료, 고객센터, 자사 기준 거래시간·운영규칙
  — 발화에 자사 언급이 없어도, 고객이 이 앱에서 실제로 거래·확인하려는
    맥락이면 internal

finance_legal
  일반 금융 개념·상품, 법규·제도, 세제 기준, 컴플라이언스, 약관·계약 —
  제도 자체이지 자사 구현이 아님

unknown
  판정 불가

Boundaries:
- "코스피 정규장 시간" → internal (거래 가능 시간)
- "거래 정지 종목 상태 확인법" → internal (확인 방법)
- market/issuer는 종목명 등장 여부가 아니라 요구된 답의 대상 단위로 판정.
  "삼성전자 때문에 코스피 오르나" → market. "삼성전자가 반도체 섹터에서
  어느 위치야" → issuer.
- 대상을 지정하지 않고 조건으로 종목을 찾게 하는 질의(대장주, 배당 상위,
  급등주)는 스크리닝이므로 market.
- 세금·법규를 제도로 물으면 finance_legal. 내 내역을 어디서 보는지, 자사
  서비스가 어떻게 처리하는지 물으면 internal.
- 청약 제도 자체는 finance_legal. 이번 주 청약 일정 목록은 market.
  특정 공모주의 공모가·일정은 issuer. 앱에서 청약하는 방법은 internal.

### TYPE

Use exactly one. Decide by what the user wants produced, not by how the
answer would be obtained.

query
  A stated fact or set of facts. No computation across periods, no judgment.

explanation
  A concept, term, procedure, or reason explained. How something works, how
  to do something, why something is so.

comparison
  Two or more subjects set side by side on the same measure.

analysis
  Computation, aggregation, or summarization across multiple data points or
  a time range. Trends, changes, rankings, market wrap-ups. States what the
  data shows, not what it means for the future.

assessment
  Judgment, outlook, causal interpretation, or evaluation. Whether something
  is favorable, what the impact will be, whether a valuation is high or low.

recommendation
  Suggestions or rankings the user is meant to act on. Which to pick, what to
  buy.

Decision order:
1. Asking what to pick or act on → recommendation
2. Asking for judgment, outlook, impact, or 여부 → assessment
3. Setting two or more subjects against each other → comparison
4. Requiring computation over a range or multiple points → analysis
5. Asking how, why, or what something means → explanation
6. Otherwise → query

Ties: explanation vs query → explanation. analysis vs assessment → assessment.

### FACET

Use exactly one. The facet names *which kind of information about the target*
the goal asks for. Decide it from what the user wants to know, not from where
such information might come from.

profile        기업 개요·기본 정보. 사업 내용, 대표, 업종, 결산월, 발행주식수,
               상장 사실 등 잘 변하지 않는 서술.
ipo            신규상장·공모. 공모가, 공모주식수, 청약, 상장일정, 상장 조건.
price          주가·지수의 수치. 시가·종가·고저가, 거래량, 거래대금, 시가총액,
               등락률, 52주 최고·최저·신고가.
flow           투자자 주체별 매매동향·수급. 외국인·기관·개인의 매수·매도·순매수.
short          공매도. 거래량, 비율, 잔고.
fundamentals   재무제표·실적. 매출, 영업이익, 순이익, 자산·부채, 재무비율.
valuation      멀티플·상대가치. PER, PBR, ROE, 배당수익률.
estimate       아직 실현되지 않은 실적 *수치*의 예상치. 예상 매출, 영업이익
               추정치, EPS·PER 전망, 컨센서스 실적.
target_price   목표주가. 적정주가·목표가 상향·하향을 포함한다.
scoring        정량 평가 점수. 실적점수, 펀더멘탈점수, 수급점수 등 등급화된 지표.
news           뉴스·언론 보도, 그리고 특정 종목에 매이지 않는 테마·업종·시장
               단위의 동향과 이슈 현황.
disclosure     기업이 공개하는 공시와 그에 딸린 주요 일정. 고객이 제출하거나
               작성하는 서식·서류는 여기가 아니라 howto 다.
screening      조건에 맞는 대상을 찾아내는 것. 질의 시점에 대상이 지정되지 않고
               순위·조건으로 종목이나 섹터·테마를 골라내야 하는 경우.
sector_map     특정 테마·이슈에 해당하는 관련주·관련 섹터의 구성.
fx             환율.
knowledge      금융 개념·용어의 설명.
regulation     법규·제도·세제·약관.
howto          절차·방법·메뉴 위치·서식 작성.
none           위 어느 데이터 면도 직접 요구하지 않는 목표. 다른 목표들의 결과를
               근거로 판단·해석·전망하는 목표가 여기 해당한다.

Rules:
- A goal whose type is `assessment` or `recommendation` is normally `none`,
  because it consumes other goals rather than requesting a facet of its own.
  Give it a real facet only when the user explicitly asks for that facet.
- If the target is named and the question is a ranking *within* that named
  target, the facet is the measure being ranked, not `screening`. `screening`
  is only for when the qualifying instances themselves must be found.
- fundamentals vs estimate: 이미 확정된 수치는 fundamentals, 아직 실현되지
  않은 추정치는 estimate.
- estimate vs target_price: 사용자가 알고 싶은 것이 실적 수치면 estimate,
  주가 수준이면 target_price. 둘 다 명시적으로 물으면 목표를 나눈다.

### HORIZON

Use exactly one, describing the time the requested information refers to:

past      Already realized and confirmed.
current   As of now, or the most recent available.
forward   Not yet realized — forecasts, estimates, expectations, outlooks,
          target prices, consensus figures, anything about what will happen.

A goal is `forward` whenever the requested figure has not yet occurred, even
if that figure already exists as a published estimate.

---

## STEP 6 — DEPENDENCIES

Create a dependency only when one goal's output is genuinely required as
input to another. Independent goals get no dependency.

Each dependency has: from, to, binds.

`binds` names what passes between them — the entity type or constraint type
the downstream goal receives. Use "context" when the downstream goal needs
the upstream answer as background rather than as a specific value.

Example:
  "코스닥 당일 거래대금 급증 섹터 및 주요 주도주"
  → g1 급증 섹터 식별, g2 해당 섹터 주도주
  → {"from":"g1","to":"g2","binds":"sector"}

---

## STEP 7 — MISSING INFORMATION

Include only information required to fulfill the request that the user did
not provide, and that cannot be resolved by ordinary defaults. A relative
time expression, an unstated market, or an unstated count is not missing
information. An unnamed subject where the goal cannot proceed without one is.

Most utterances have none. An empty list is the correct answer.

---

## FORBIDDEN

Do not mention tools, APIs, workflows, capabilities, databases, retrieval
systems, or execution plans.
Do not infer internal system behavior.
Do not answer the user's question.
Do not invent entities or constraints not present in the utterance.
Do not add a goal for information the user did not ask for, even when that
information commonly accompanies the topic.
Only represent what the user asked for.
"""

# CSV 한 행이 {query} 자리에 들어간다. 컬럼이 여러 개면 {컬럼명} 으로 더 쓸 수 있다.
USER_TEMPLATE = """{query}"""

# ─────────────────────────────────────────────────────────────

MODEL = "gpt-5-mini"

# ─────────────────────────────────────────────────────────────
# SCHEMA — 프롬프트가 정의한 출력 구조를 strict 로 강제
#   property 순서 = 프롬프트의 "Produce your output in this order"
#   goal id 를 g1..g5 enum 으로 묶어 "최대 5개" 제한도 스키마로 강제
# ─────────────────────────────────────────────────────────────

GOAL_IDS = ["g1", "g2", "g3", "g4", "g5"]

DOMAINS = ["market", "issuer", "internal", "finance_legal", "unknown"]
TYPES = [
    "query",
    "explanation",
    "comparison",
    "analysis",
    "assessment",
    "recommendation",
]
HORIZONS = ["past", "current", "forward"]

FACETS = [
    "profile",
    "ipo",
    "price",
    "flow",
    "short",
    "fundamentals",
    "valuation",
    "estimate",
    "target_price",
    "scoring",
    "news",
    "disclosure",
    "screening",
    "sector_map",
    "fx",
    "knowledge",
    "regulation",
    "howto",
    "none",
]

ENTITY_TYPES = [
    "company",
    "sector",
    "theme",
    "market",
    "index",
    "metric",
    "corporate_event",
    "market_event",
    "investor_group",
    "product",
    "account",
    "procedure",
    "regulation",
    "document",
    "app_feature",
]
CONSTRAINT_TYPES = [
    "period",
    "scope",
    "count",
    "ranking",
    "direction",
    "condition",
    "channel",
]


def _obj(props: dict) -> dict:
    """strict 모드는 모든 property 가 required 이고 additionalProperties 가 false 여야 한다."""
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


SCHEMA = _obj(
    {
        "goals": {
            "type": "array",
            "items": _obj(
                {
                    "id": {"type": "string", "enum": GOAL_IDS},
                    "target": {
                        "type": "string",
                        "description": "목표 대상을 가리키는 짧은 한국어 명사구",
                    },
                    "domain": {"type": "string", "enum": DOMAINS},
                    "type": {"type": "string", "enum": TYPES},
                    "facet": {
                        "type": "string",
                        "enum": FACETS,
                        "description": "요구된 정보의 종류. 다른 목표의 결과를 소비하는 판단 목표는 none",
                    },
                    "horizon": {"type": "string", "enum": HORIZONS},
                }
            ),
        },
        "entities": {
            "type": "array",
            "items": _obj(
                {
                    "goal_id": {"type": "string", "enum": GOAL_IDS},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "value": {
                        "type": "string",
                        "description": "발화에 등장한 표현 그대로. 정규화·번역 금지",
                    },
                }
            ),
        },
        "constraints": {
            "type": "array",
            "items": _obj(
                {
                    "goal_id": {"type": "string", "enum": GOAL_IDS},
                    "type": {"type": "string", "enum": CONSTRAINT_TYPES},
                    "value": {
                        "type": "string",
                        "description": "발화에 등장한 표현 그대로. 날짜 변환 금지",
                    },
                }
            ),
        },
        "dependencies": {
            "type": "array",
            "items": _obj(
                {
                    "from": {"type": "string", "enum": GOAL_IDS},
                    "to": {"type": "string", "enum": GOAL_IDS},
                    "binds": {
                        "type": "string",
                        "description": (
                            "하류 목표가 전달받는 entity/constraint 타입. "
                            "특정 값이 아니라 배경으로 필요하면 context"
                        ),
                    },
                }
            ),
        },
        "missing_information": {
            "type": "array",
            "description": "요청 수행에 반드시 필요한데 발화에 없는 정보. 보통 빈 배열",
            "items": {"type": "string"},
        },
    }
)

TEXT_FORMAT = {
    "type": "json_schema",
    "name": "goal_parse",
    "strict": True,
    "schema": SCHEMA,
}

FIELDS = [
    "idx",
    "input",
    "n_goals",
    "goals",
    "entities",
    "constraints",
    "dependencies",
    "missing_information",
    "json",
    "in_tok",
    "out_tok",
    "error",
]


def read_rows(path: str, column: str) -> list[tuple[int, dict]]:
    """(원본 행번호, 행dict) 목록. 빈 행은 버린다.

    원본 CSV 에 따옴표 없는 쉼표가 있으면 DictReader 가 뒷부분을 restkey 로
    흘려버린다. 조용히 잘린 입력으로 돌지 않도록 다시 이어 붙인다.
    """
    rest = "__rest__"
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, restkey=rest))
    out = []
    for i, row in enumerate(rows):
        extra = row.pop(rest, None)
        if extra:
            last = list(row)[-1]  # 넘친 값은 마지막 컬럼에 붙는다
            row[last] = ",".join([row.get(last) or "", *extra])
        value = (row.get(column) or "").strip()
        if value:
            out.append((i, row))
    return out


def done_indices(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {
            int(r["idx"])
            for r in csv.DictReader(f)
            if r.get("idx", "").isdigit() and not r.get("error")
        }


def flatten(data: dict) -> dict:
    """엑셀에서 눈으로 보기 위한 요약 컬럼. 원본은 json 컬럼에 그대로 남는다."""
    goals = data["goals"]
    return {
        "n_goals": len(goals),
        "goals": "\n".join(
            f"{g['id']} [{g['domain']}/{g['type']}/{g['facet']}/{g['horizon']}] {g['target']}"
            for g in goals
        ),
        "entities": "\n".join(
            f"{e['goal_id']} {e['type']}={e['value']}" for e in data["entities"]
        ),
        "constraints": "\n".join(
            f"{c['goal_id']} {c['type']}={c['value']}" for c in data["constraints"]
        ),
        "dependencies": "\n".join(
            f"{d['from']}->{d['to']} ({d['binds']})" for d in data["dependencies"]
        ),
        "missing_information": "\n".join(data["missing_information"]),
    }


def ask(client: OpenAI, row: dict, args) -> dict:
    prompt = USER_TEMPLATE.format(**row)
    resp = client.responses.create(
        model=args.model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": args.effort},
        text={"format": TEXT_FORMAT, "verbosity": args.verbosity},
    )

    raw = resp.output_text
    if not raw:
        # 보통 reasoning 이 예산을 다 써서 status=incomplete 인 경우
        detail = getattr(resp, "incomplete_details", None)
        raise RuntimeError(f"빈 응답 (status={resp.status}, detail={detail})")

    data = json.loads(raw)  # strict 스키마라 파싱은 실패하지 않아야 정상
    usage = resp.usage
    return {
        "input": prompt,
        **flatten(data),
        "json": json.dumps(data, ensure_ascii=False),
        "in_tok": usage.input_tokens if usage else "",
        "out_tok": usage.output_tokens if usage else "",
        "error": "",
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="work.csv")
    p.add_argument("--output", default="work_out.csv")
    p.add_argument("--column", default="query", help="프롬프트에 넣을 입력 컬럼명")
    p.add_argument("--limit", type=int, help="앞에서 N행만")
    p.add_argument("--workers", type=int, default=4, help="동시 호출 수")
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--effort", default="low", choices=["minimal", "low", "medium", "high"]
    )
    p.add_argument("--verbosity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--resume", action="store_true", help="출력 CSV 에 있는 행은 건너뜀")
    args = p.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY 가 없습니다. .env 를 확인하세요.")
        return 1

    rows = read_rows(args.input, args.column)
    skip = done_indices(args.output) if args.resume else set()
    rows = [r for r in rows if r[0] not in skip]
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("[INFO] 처리할 행이 없습니다.")
        return 0

    print(f"[MODEL] {args.model} (effort={args.effort}, workers={args.workers})")
    print(f"[RUN] {len(rows)}행" + (f" (건너뜀 {len(skip)})" if skip else ""))

    client = OpenAI(api_key=api_key)
    append = args.resume and os.path.exists(args.output)
    lock = threading.Lock()
    failures = 0

    with open(
        args.output, "a" if append else "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not append:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ask, client, row, args): idx for idx, row in rows}
            for n, fut in enumerate(as_completed(futures), 1):
                idx = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    failures += 1
                    rec = {"error": f"{type(e).__name__}: {e}"}
                with lock:
                    writer.writerow({"idx": idx, **rec})
                    f.flush()
                    tag = "ERR" if rec["error"] else "OK "
                    print(f"[{n}/{len(rows)}] {tag} row={idx}")

    sort_by_index(args.output)
    print(f"\n[DONE] {args.output} (실패 {failures}건)")
    return 1 if failures else 0


def sort_by_index(path: str) -> None:
    """완료 순으로 append 된 결과를 원본 행 순서대로 정렬해 다시 쓴다."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["idx"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
