"""arm C: 의도 분류 + 엔티티 추출만 시킨다. 함수 매핑은 route_intent.py 가 한다.

A/B 는 goal 분해와 (domain, type, facet, horizon) 4축 분류를 LLM 에게 시킨다.
그 설명에만 프롬프트의 절반이 든다. C 는 그걸 24개 intent 택소노미 하나로
갈음하고, 팬아웃에 필요한 엔티티만 남긴다.

절삭 실험이 근거다 — 라우팅 F1 에 대한 기여가 facet −0.535, 엔티티 −0.118,
type −0.080, horizon −0.009 였다. C 는 facet 을 intent 로 대신하고 엔티티는
그대로 두며 horizon 은 버린다.

실행:
    .venv/Scripts/python.exe run_csv_c.py --input "work copy.csv" --output work_out_c.csv
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

from run_csv import ENTITY_TYPES, CONSTRAINT_TYPES, _obj, read_rows

MODEL = "gpt-5-mini"
TAXONOMY = "intent_taxonomy.csv"


def load_taxonomy(path: str = TAXONOMY):
    """(intent_id 목록, 프롬프트에 넣을 표) — 택소노미 파일이 곧 단일 출처다."""
    ids, lines = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ids.append(r["intent_id"])
            lines.append(
                f"{r['intent_id']}  {r['대분류']} > {r['소분류']}\n"
                f"      {r['설명']}"
            )
    return ids, "\n".join(lines)


INTENT_IDS, TAXONOMY_TABLE = load_taxonomy()

SYSTEM_PROMPT = f"""
You are an Intent Classifier.

Your only job is to label a user's request with the intents it contains, and to
extract the entities and constraints the request names.

You do NOT decide which tools, APIs, or functions should be used.
You do NOT decompose the request into goals.
You do NOT answer the question.

---

## CONTEXT

Users are retail customers of a Korean securities firm. Utterances arrive in
Korean and are often compound — a single sentence may contain several distinct
requests, or may state a cause and ask for its effect.

"계좌" always means a brokerage account, never a bank account.
ISA, CMA, 연금저축, IRP, 랩(Wrap), 신탁, 통합증거금 are products of this firm.

---

## STEP 1 — INTENTS

Choose every intent the utterance actually asks for, from this list and no
other. Order them by how central they are to the request.

{TAXONOMY_TABLE}

Rules:

- Multi-label. A compound utterance carries several intents. "실적 전망" asks
  for 재무 정보 and 기업 평가 및 전망 both.
- Label what the user asked for, not what would be useful to know. Do not add
  an intent for information that merely accompanies the topic.
- When a request states a premise and asks its consequence ("X에 따른 Y",
  "X가 Y에 미치는 영향"), label both the premise's intent and the
  consequence's intent. The consequence is usually 기업 평가 및 전망 or
  시장 전망.
- 시장 전망(I5-1) vs 산업 동향 분석(I5-2): 지수·시장 전반이면 I5-1,
  특정 업종·테마면 I5-2.
- 재무 정보(I6-6) vs 컨센서스(I8-2): 확정된 실적이면 I6-6, 아직 실현되지
  않은 추정치·목표주가면 I8-2.
- 티레이더M 사용법(I3-1) vs 투자 절차 안내(I3-2): 앱 화면·메뉴 경로면 I3-1,
  업무 처리 절차면 I3-2.
- At most 5 intents. Prefer fewer.

---

## STEP 2 — ENTITIES

Return only entities explicitly present in the utterance. Copy values verbatim —
no normalization, no ticker lookup, no translation.

Each entity carries the intent_id it belongs to.

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

A noun that modifies the head noun is still an entity if it names something on
this list.

---

## STEP 3 — CONSTRAINTS

Return only constraints explicitly present. Copy values verbatim; do not convert
dates to numeric form. Each constraint carries the intent_id it belongs to.

Allowed types — use no others:

period       시간 표현. 시작과 끝이 따로 언급되면 항목을 두 개 낸다.
scope        검색 범위를 한정하는 시장·섹터·테마. 대상 자체가 아니라 범위일 때만.
count        요청된 개수. 숫자만.
ranking      상위 | 하위
direction    급증 | 급락 | 상승 | 하락 | 강세 | 약세
condition    결과를 걸러내는 자격 요건 (우량주, 저평가, 배당주, 만기 도래 등)
channel      수행 경로 (온라인, 비대면, 영업점, 모바일)

If a phrase is the subject of the question, it is an entity. If it narrows which
instances of the subject qualify, it is a constraint.

---

## STEP 4 — SUBJECT

Return `subject`: a short Korean noun phrase naming what the request is about.
One phrase for the whole utterance.

---

## FORBIDDEN

Do not mention tools, APIs, functions, workflows, or execution plans.
Do not invent entities or constraints not present in the utterance.
Do not add intents for information the user did not ask for.
"""

USER_TEMPLATE = """{query}"""

SCHEMA = _obj(
    {
        "intents": {
            "type": "array",
            "description": "발화가 담은 의도. 중심적인 것부터. 최대 5개",
            "items": {"type": "string", "enum": INTENT_IDS},
        },
        "entities": {
            "type": "array",
            "items": _obj(
                {
                    "intent_id": {"type": "string", "enum": INTENT_IDS},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "value": {"type": "string", "description": "발화 표현 그대로"},
                }
            ),
        },
        "constraints": {
            "type": "array",
            "items": _obj(
                {
                    "intent_id": {"type": "string", "enum": INTENT_IDS},
                    "type": {"type": "string", "enum": CONSTRAINT_TYPES},
                    "value": {"type": "string", "description": "발화 표현 그대로"},
                }
            ),
        },
        "subject": {
            "type": "string",
            "description": "요청 대상을 가리키는 짧은 한국어 명사구",
        },
    }
)

TEXT_FORMAT = {
    "type": "json_schema",
    "name": "intent_parse",
    "strict": True,
    "schema": SCHEMA,
}

FIELDS = [
    "idx",
    "input",
    "n_intents",
    "intents",
    "entities",
    "constraints",
    "subject",
    "json",
    "in_tok",
    "out_tok",
    "error",
]


def done_indices(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {
            int(r["idx"])
            for r in csv.DictReader(f)
            if r.get("idx", "").isdigit() and not r.get("error")
        }


def flatten(d: dict) -> dict:
    return {
        "n_intents": len(d["intents"]),
        "intents": ";".join(d["intents"]),
        "entities": "\n".join(
            f"{e['intent_id']} {e['type']}={e['value']}" for e in d["entities"]
        ),
        "constraints": "\n".join(
            f"{c['intent_id']} {c['type']}={c['value']}" for c in d["constraints"]
        ),
        "subject": d["subject"],
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
        detail = getattr(resp, "incomplete_details", None)
        raise RuntimeError(f"빈 응답 (status={resp.status}, detail={detail})")
    data = json.loads(raw)
    usage = resp.usage
    return {
        "input": prompt,
        **flatten(data),
        "json": json.dumps(data, ensure_ascii=False),
        "in_tok": usage.input_tokens if usage else "",
        "out_tok": usage.output_tokens if usage else "",
        "error": "",
    }


def sort_by_index(path: str) -> None:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["idx"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="work.csv")
    p.add_argument("--output", default="work_out_c.csv")
    p.add_argument("--column", default="query")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--effort", default="low", choices=["minimal", "low", "medium", "high"]
    )
    p.add_argument("--verbosity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--resume", action="store_true")
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


if __name__ == "__main__":
    raise SystemExit(main())
