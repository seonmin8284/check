"""A/B 테스트 D안: B안에서 intent 를 걷어낸 것. 근거 추론 기반 목적 분할 + 경계 few-shot.

A안(run_csv.py) 대비 변경점만:
  1. 목적 분할을 표면 연결어가 아니라 "답하는 데 필요한 근거"에서 유도한다.
    3. 경계 케이스 few-shot 8건을 프롬프트에 넣는다.
스키마의 나머지·모델·파라미터·출력 컬럼은 A안과 동일하게 두어 비교 가능성을 유지한다.

실행:
    .venv/Scripts/python.exe run_csv_b.py --input work.csv   --output work_out_s4c.csv
    .venv/Scripts/python.exe run_csv_b.py --input invest.csv --output invest_out_b.csv
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
# 프롬프트 (B안)
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

## STEP 1 — CLASSIFY

Now that the goals, entities, and constraints are settled, assign domain,
type, facet, and horizon to each goal.

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

query          A stated fact or set of facts. No computation, no judgment.
explanation    A concept, term, procedure, or reason explained.
comparison     Two or more subjects set side by side on the same measure.
analysis       Computation, aggregation, or summarization across multiple
               data points or a time range. States what the data shows, not
               what it means for the future.
assessment     Judgment, outlook, causal interpretation, or evaluation.
recommendation Suggestions or rankings the user is meant to act on.

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

profile        기업 개요·기본 정보. 사업 내용, 대표, 업종, 결산월, 발행주식수.
ipo            신규상장·공모. 공모가, 공모주식수, 청약, 상장일정, 상장 조건.
price          주가·지수의 수치. 시가·종가·고저가, 거래량, 거래대금, 시가총액,
               등락률, 52주 최고·최저·신고가, 주가 변동성.
flow           투자자 주체별 매매동향·수급. 외국인·기관·개인의 매수·매도·순매수.
short          공매도. 거래량, 비율, 잔고.
fundamentals   재무제표·실적. 매출, 영업이익, 순이익, 자산·부채, 재무비율.
valuation      멀티플·상대가치. PER, PBR, ROE, 배당수익률.
estimate       아직 실현되지 않은 실적 *수치*의 예상치. 예상 매출, 영업이익
               추정치, EPS·PER 전망, 컨센서스 실적, 예상 성장률.
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
  target, the facet is the measure being ranked, not `screening`.
- fundamentals vs estimate: 이미 확정된 수치는 fundamentals, 아직 실현되지
  않은 추정치는 estimate.
- estimate vs target_price: 사용자가 알고 싶은 것이 실적 수치면 estimate,
  주가 수준이면 target_price. 둘 다 명시적으로 물으면 목표를 나눈다.
- 테마·업황의 수요·업황 동향 자체를 묻는 것은 news. 그 테마에 속한 종목·
  섹터 구성을 묻는 것은 sector_map.

### HORIZON

past      Already realized and confirmed.
current   As of now, or the most recent available.
forward   Not yet realized — forecasts, estimates, expectations, outlooks,
          target prices, consensus figures.

A goal is `forward` whenever the requested figure has not yet occurred, even
if that figure already exists as a published estimate.

---

## STEP 2 — EVIDENCE INVENTORY (think before you split)

Before writing any goal, answer this to yourself:

  "If I had to actually answer this utterance, what separate pieces of
   evidence would I need to have in front of me?"

Enumerate them. Each piece of evidence is a distinct observable fact about a
distinct subject — a figure, a record, a document, a procedure, a status.

Rules for the inventory:

- Two facts are the *same* piece of evidence if one lookup of one subject
  yields both. 시가 and 종가 of one stock is one piece. 매출 and 영업이익
  of one company for one period is one piece.
- Two facts are *different* pieces if they concern different subjects, or
  different kinds of information about the same subject (실적 vs 수급 vs
  공시 vs 주가).
- Realized figures and not-yet-realized figures about the same subject are
  different pieces (2024 확정 매출 vs 2026 예상 매출).
- A premise the user states as given still needs a piece of evidence if the
  answer depends on its magnitude, direction, or current status. A premise
  that merely frames the question needs none.
- The user's own judgment request — 전망, 영향도, 여부, 가능성, 효과, 배경 —
  is never a piece of evidence. It consumes them.

Then: **one piece of evidence → one goal**, plus one goal for the judgment
if the user asked for one. This inventory, not the sentence's connectives,
decides the split.

### What this changes

Do not split just because you see 및, 와, 그리고, 후, 에 따른.
Do not refuse to split just because there is no connective. A single run of
nouns can still require two pieces of evidence.

  "삼성전자 HBM 납품 실적 영향도 전망"
    evidence: (a) 삼성전자 HBM 납품 실적 → goal
              (b) 판단: 그것이 실적에 미치는 영향 → goal, depends on (a)

  "비대면계좌 개설 후 한도제한계좌 해제 방법"
    evidence: (a) 한도제한계좌 해제 절차 → goal
    개설은 해제 절차가 적용되는 상황을 설명하는 전제일 뿐 별도 조회가 아니다.
    → 1 goal.

### Causal and impact questions

When a user states a premise and asks about its consequence — "X에 따른 Y",
"X가 Y에 미치는 영향", "X에 따른 Y 수혜 여부" — keep the consequence as its
own goal of type `assessment`. Do not dissolve it into fact lookups only.
Make it depend on the evidence goals it consumes.

### Do NOT split when

- The parts describe a single procedure. "A 및 B 방법" describing one
  workflow is one goal.
- One part is a qualifier or filter on the other. "외국인 수급이 강하게
  유입된 우량주" is one goal with a filter.
- The parts are synonyms or restatements.

### Limits

Produce at most 5 goals. Prefer fewer. If an utterance would exceed 5, merge
the most closely related evidence pieces rather than truncating.

---

## STEP 3 — TARGET

For each goal:

- target — short Korean noun phrase naming the object of the goal.

---

## STEP 4 — ENTITIES

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

## STEP 5 — CONSTRAINTS

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

## STEP 6 — DEPENDENCIES

Create a dependency only when one goal's output is genuinely required as
input to another. Independent goals get no dependency.

Each dependency has: from, to, binds. `binds` names what passes between them
— the entity type or constraint type the downstream goal receives. Use
"context" when the downstream goal needs the upstream answer as background
rather than as a specific value.

Every `assessment` goal with facet=none must have at least one incoming
dependency. If it has none, your evidence inventory was incomplete — go back
to STEP 0.

---

## STEP 7 — MISSING INFORMATION

Include only information required to fulfill the request that the user did
not provide, and that cannot be resolved by ordinary defaults. A relative
time expression, an unstated market, or an unstated count is not missing
information. An unnamed subject where the goal cannot proceed without one is.

Most utterances have none. An empty list is the correct answer.

---

## BOUNDARY EXAMPLES

These are chosen because each sits on a line that is easy to get wrong.
Entities and constraints are abbreviated for readability; produce them in
full in your actual output.

---
IN: 삼성전자 HBM 납품 실적 영향도 전망

Evidence: 삼성전자의 HBM 납품 실적 수치 하나. "영향도 전망"은 판단이므로
근거가 아니다. HBM 업황은 사용자가 묻지 않았으므로 goal 로 만들지 않는다.

  g1 [issuer/query/fundamentals/current] (I6-6) 삼성전자 HBM 납품 실적
  g2 [issuer/assessment/none/forward]    (I6-8) HBM 납품의 실적 영향도 전망
  dep: g1->g2 (context)

왜 2개인가: 확정 실적 조회와 그 해석은 서로 다른 산출물이다.
왜 3개가 아닌가: "HBM 업황"은 발화에 없다. 흔히 같이 따라오는 정보라도
사용자가 묻지 않았으면 goal 이 아니다.
---
IN: SK하이닉스 최근분기 HBM 수요 증가에 따른 예상 영업이익 추정치

Evidence: (a) 최근분기 HBM 수요 증가 — 전제이지만 크기를 알아야 추정치를
해석할 수 있으므로 근거다. (b) SK하이닉스 예상 영업이익 — 아직 실현되지
않은 수치.

  g1 [market/query/news/past]          (I5-2) 최근분기 HBM 수요 증가 동향
  g2 [issuer/query/estimate/forward]   (I8-2) SK하이닉스 예상 영업이익 추정치
  dep: g1->g2 (context)
  constraint: g1 period=최근분기

왜 g1 이 market 인가: HBM 수요는 테마 단위 동향이고 발행사가 지정되지 않았다.
왜 estimate 이고 fundamentals 가 아닌가: 아직 실현되지 않은 추정치다.
---
IN: LG에너지솔루션 전기차 수요 둔화 관련 목표 주가 하향 가능성

Evidence: (a) 전기차 수요 둔화 동향, (b) LG에너지솔루션 현재 목표주가.
"하향 가능성"은 판단.

  g1 [market/query/news/current]           (I5-2) 전기차 수요 둔화 동향
  g2 [issuer/query/target_price/forward]   (I8-2) LG에너지솔루션 목표주가
  g3 [issuer/assessment/none/forward]      (I6-8) 목표주가 하향 가능성
  dep: g1->g3 (context), g2->g3 (context)

왜 target_price 이고 estimate 이 아닌가: 사용자가 알고 싶은 것은 실적 수치가
아니라 주가 수준이다.
---
IN: 네이버(NAVER) 최근 외국인 대량 매도 및 주가 하락 배경 분석

Evidence: (a) 외국인 수급, (b) 주가 추이. 둘은 같은 종목이지만 다른 종류의
정보이므로 별개의 근거다. "배경 분석"은 판단.

  g1 [issuer/query/flow/current]      (I7-1) 네이버 외국인 매도 동향
  g2 [issuer/analysis/price/past]     (I7-2) 네이버 주가 하락 추이
  g3 [issuer/assessment/none/current] (I6-8) 매도·하락 배경
  dep: g1->g3 (context), g2->g3 (context)
  constraint: g1 period=최근, g1 direction=대량 매도 / g2 direction=하락

"및" 이 있어서 나눈 것이 아니라 수급과 주가가 서로 다른 근거라서 나눴다.
---
IN: 한화솔루션 유상증자 결정 공시 후 단기 주가 변동성 분석

Evidence: (a) 유상증자 결정 공시, (b) 공시 이후 주가 변동성.

  g1 [issuer/query/disclosure/past]  (I6-4) 한화솔루션 유상증자 결정 공시
  g2 [issuer/analysis/price/current] (I7-2) 공시 이후 단기 주가 변동성
  dep: g1->g2 (context)

왜 assessment goal 이 없는가: "변동성 분석"은 데이터가 무엇을 보여주는지를
묻는 것이지 앞으로 어떨지를 묻는 것이 아니다. analysis 로 끝난다.
변동성은 price facet 이다. scoring 의 변동성점수와 혼동하지 말 것.
---
IN: 비대면계좌 개설 후 한도제한계좌 해제 방법

Evidence: 한도제한계좌 해제 절차 하나.

  g1 [internal/explanation/howto/current] (I3-2) 한도제한계좌 해제 방법
  constraint: g1 channel=비대면

왜 1개인가: "비대면계좌 개설"은 해제가 필요해진 상황을 설명하는 전제이고,
사용자가 개설 방법을 묻고 있지 않다. 근거가 하나면 goal 도 하나다.
---
IN: 미성년자 계좌개설 시 필수 제출 서류 목록

Evidence: (a) 개설 시 제출 서류 목록(업무 안내), (b) 미성년자에게 그 서류를
요구하는 근거 규정. 목록과 요건은 서로 다른 코퍼스에 있다.

  g1 [internal/explanation/howto/current]        (I3-2) 미성년자 계좌개설 제출 서류
  g2 [finance_legal/explanation/regulation/current] (I2-2) 미성년자 계좌개설 서류 요건
  dep: g1->g2 (context)

절차 goal 과 그 절차의 법적 근거 goal 은 나눈다.
---
IN: 계좌 비밀번호 5회 오류 시 재등록 방법

Evidence: 비밀번호 재등록 절차 하나.

  g1 [internal/explanation/howto/current] (I3-2) 계좌 비밀번호 재등록 방법

"5회 오류 시"는 절차가 적용되는 조건일 뿐 별도 조회 대상이 아니다.
재등록을 앱 어디서 하는지가 아니라 무엇을 해야 하는지를 묻고 있으므로 I3-2.
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
# SCHEMA — A안과 동일 (B안의 intent 필드를 뺐다)
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
    p.add_argument("--output", default="work_out_s4c.csv")
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
    print(f"[RUN] B안 {len(rows)}행" + (f" (건너뜀 {len(skip)})" if skip else ""))

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
