"""두 개의 비교 arm.

Arm A: LLM 은 의도분류 + 슬롯추출만 → 결정적 capability graph 가 실제 계산
Arm B: 모든 결과 테이블을 시스템 프롬프트에 박은 단건 호출 → LLM 이 분류+조회+답변
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

import capabilities as caps
import graph

MODEL = "gpt-5-mini"
REASONING_EFFORT = "low"   # benchmark.py 가 런타임에 덮어쓴다


@dataclass
class ArmResult:
    arm: str
    utterance: str
    intent: str | None
    result: Any
    latency_s: float
    api_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None
    detail: dict = field(default_factory=dict)
    answer: str | None = None
    compose_s: float = 0.0


def _reasoning_tokens(resp) -> int:
    d = getattr(resp.usage, "output_tokens_details", None)
    return getattr(d, "reasoning_tokens", 0) if d else 0


# ===========================================================================
# 공통 — 결과값을 자연어로 옮겨 적는 단계
# ===========================================================================

COMPOSE_SYSTEM = """너는 이미 계산된 결과를 사용자에게 전달하는 역할이다.

[절대 규칙]
- 주어진 결과값만 사용한다. 새로 계산하거나, 값을 고치거나, 항목을 더하거나 빼지 마라.
- 결과값이 null 이면 처리할 수 없는 요청이라는 뜻이다. 짧게 안내하고 끝내라.
- 한국어로 2문장 이내. 사족 금지."""


def compose_answer(client: OpenAI, utterance: str, intent: str, result: Any):
    payload = json.dumps({"의도": intent, "결과값": result}, ensure_ascii=False)
    t = time.perf_counter()
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": COMPOSE_SYSTEM},
            {"role": "user", "content": f"사용자 질문: {utterance}\n계산 결과: {payload}"},
        ],
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": "low"},
    )
    return (resp.output_text, time.perf_counter() - t,
            resp.usage.input_tokens, resp.usage.output_tokens)


# ===========================================================================
# 출력 규약 — 두 arm 의 채점 기준을 동일하게 맞추기 위한 공통 명세
# ===========================================================================

ARRAY_INTENTS = [
    "fibonacci", "primes", "collatz", "divisors", "prime_factors", "squares",
    "triangular", "sort", "unique", "evens", "odds", "cumsum", "last_k",
    "fib_primes", "fib_sorted", "fib_evens", "fib_cumsum", "fib_last_k",
    "primes_last_k",
]
SCALAR_INTENTS = [
    "factorial", "gcd", "lcm", "mean", "sum", "median", "max", "min", "count",
    "fib_mean", "fib_sum", "fib_median", "fib_max", "fib_evens_sum",
    "primes_mean", "primes_sum", "primes_count", "collatz_len", "collatz_max",
    "divisors_sum", "divisors_count",
]
OBJECT_INTENTS = ["spread", "primes_spread"]

OUTPUT_CONTRACT = f"""[결과값 형식]
- 배열을 내는 의도: {", ".join(ARRAY_INTENTS)}
- 숫자 하나를 내는 의도: {", ".join(SCALAR_INTENTS)}
- 객체를 내는 의도: {", ".join(OBJECT_INTENTS)}
  -> {{"range": 최대-최소, "variance": 표본분산, "stdev": 표본표준편차}} (둘 다 소수점 4자리 반올림)
- unsupported: null

[계산 규약]
- 피보나치는 F[0]=0, F[1]=1 이며 "n항까지" 는 F[0]..F[n] 즉 n+1 개다.
- 소수 판정에서 0 과 1 은 소수가 아니다.
- 콜라츠 궤적은 시작값과 마지막 1 을 모두 포함한다.
- 약수는 1 과 자기 자신을 포함하고 오름차순이다.
- 소인수분해는 중복을 포함한 오름차순 목록이다.
- 제곱수는 1^2..n^2, 삼각수는 앞에서부터 n 개다.
- 누적합은 원소 개수가 원본과 같다.
- 평균(mean)은 소수점 4자리 반올림한다.
- last_k 는 뒤에서 k 개를 원래 순서대로 낸다."""


# ===========================================================================
# Arm A — 의도분류 + 결정적 capability graph
# ===========================================================================

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": caps.INTENTS},
        "n": {"type": ["integer", "null"], "description": "항 개수 / 상한 / 대상 정수"},
        "a": {"type": ["integer", "null"], "description": "gcd·lcm 첫 인자"},
        "b": {"type": ["integer", "null"], "description": "gcd·lcm 둘째 인자"},
        "k": {"type": ["integer", "null"], "description": "마지막 k 개의 k"},
        "numbers": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "description": "사용자가 발화에 직접 나열한 수열",
        },
        "desc": {"type": ["boolean", "null"], "description": "내림차순이면 true"},
    },
    "required": ["intent", "n", "a", "b", "k", "numbers", "desc"],
    "additionalProperties": False,
}

ARM_A_SYSTEM = f"""너는 의도 분류기다. 사용자 발화를 아래 의도 중 하나로 분류하고 슬롯을 채워라.
계산은 절대 하지 마라 — 분류와 슬롯 추출만 한다.
단, 슬롯 값이 발화에 간접적으로만 있으면(예: "2의 6제곱 이하") 그 값은 정수로 환산해 채워라.

[의도 목록]
{caps.taxonomy_text()}

해당 없는 슬롯은 null 로 둔다."""


def run_arm_a(client: OpenAI, utterance: str, compose: bool = False) -> ArmResult:
    t0 = time.perf_counter()
    try:
        t_api = time.perf_counter()
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": ARM_A_SYSTEM},
                {"role": "user", "content": utterance},
            ],
            reasoning={"effort": REASONING_EFFORT},
            text={"format": {"type": "json_schema", "name": "intent_slots",
                             "schema": CLASSIFY_SCHEMA, "strict": True}},
        )
        api_s = time.perf_counter() - t_api
    except Exception as e:
        return ArmResult("A", utterance, None, None, time.perf_counter() - t0, 0.0,
                         error=f"{type(e).__name__}: {e}")

    parsed = json.loads(resp.output_text)
    intent = parsed.pop("intent")
    slots = {k: v for k, v in parsed.items() if v is not None}

    run = graph.execute(intent, slots)   # 여기엔 LLM 이 없다

    answer, compose_s, c_in, c_out = None, 0.0, 0, 0
    if compose and run.ok:
        answer, compose_s, c_in, c_out = compose_answer(
            client, utterance, intent, run.result
        )

    return ArmResult(
        arm="A", utterance=utterance, intent=intent, result=run.result,
        latency_s=time.perf_counter() - t0, api_s=api_s,
        input_tokens=resp.usage.input_tokens + c_in,
        output_tokens=resp.usage.output_tokens + c_out,
        reasoning_tokens=_reasoning_tokens(resp),
        error=run.error,
        detail={"slots": slots, "trace": run.trace},
        answer=answer, compose_s=compose_s,
    )


# ===========================================================================
# Arm B — 전부 시스템 프롬프트에 넣는 단건 호출
# ===========================================================================


def _lookup_tables() -> str:
    """미리 계산해 프롬프트에 통째로 박아 넣는 정적 표.

    capability 가 늘수록 표로 못 만드는 의도(콜라츠·약수·사용자수열 등)가 늘어난다.
    그 부분은 모델이 직접 계산해야 한다 — 이 접근의 확장 한계 자체가 측정 대상이다.
    """
    fib = caps.fibonacci(40)
    return f"""[피보나치 표] F[0]..F[40]
{chr(10).join(f"  F[{i}] = {v}" for i, v in enumerate(fib))}

[소수 표] 500 이하의 모든 소수
  {", ".join(map(str, caps.primes_upto(500)))}

[팩토리얼 표] 0! .. 25!
{chr(10).join(f"  {i}! = {caps.factorial(i)}" for i in range(26))}

[제곱수 표] 1^2 .. 20^2
  {", ".join(map(str, caps.squares(20)))}

[삼각수 표] 앞에서부터 20개
  {", ".join(map(str, caps.triangular(20)))}

[표에 없는 것] 콜라츠, 약수, 소인수분해, gcd/lcm, 사용자가 직접 준 수열에 대한
모든 연산은 표가 없다. 직접 정확히 계산해라."""


ARM_B_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": caps.INTENTS},
        "result_json": {
            "type": "string",
            "description": "결과값을 JSON 으로 직렬화한 문자열. unsupported 면 \"null\".",
        },
    },
    "required": ["intent", "result_json"],
    "additionalProperties": False,
}

ARM_B_SYSTEM = f"""너는 수치 질의 처리기다. 사용자 발화를 의도로 분류하고, 아래 표를 참고해
최종 결과값까지 한 번에 만들어라.
슬롯 값이 발화에 간접적으로만 있으면(예: "2의 6제곱 이하") 정수로 환산해서 쓴다.

[의도 목록]
{caps.taxonomy_text()}

{_lookup_tables()}

{OUTPUT_CONTRACT}"""


def _arm_b_schema(compose: bool) -> dict:
    """작성 단계는 arm B 의 설계상 같은 호출 안에서 필드 하나로 처리한다."""
    if not compose:
        return ARM_B_SCHEMA
    s = json.loads(json.dumps(ARM_B_SCHEMA))
    s["properties"]["answer"] = {
        "type": "string",
        "description": "사용자에게 보여줄 최종 한국어 문장. 2문장 이내.",
    }
    s["required"].append("answer")
    return s


def run_arm_b(client: OpenAI, utterance: str, compose: bool = False) -> ArmResult:
    t0 = time.perf_counter()
    try:
        t_api = time.perf_counter()
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": ARM_B_SYSTEM},
                {"role": "user", "content": utterance},
            ],
            reasoning={"effort": REASONING_EFFORT},
            text={"format": {"type": "json_schema", "name": "intent_result",
                             "schema": _arm_b_schema(compose), "strict": True}},
        )
        api_s = time.perf_counter() - t_api
    except Exception as e:
        return ArmResult("B", utterance, None, None, time.perf_counter() - t0, 0.0,
                         error=f"{type(e).__name__}: {e}")

    parsed = json.loads(resp.output_text)
    try:
        result = json.loads(parsed["result_json"])
        err = None
    except json.JSONDecodeError as e:
        result, err = None, f"result_json 파싱 실패: {e}"

    return ArmResult(
        arm="B", utterance=utterance, intent=parsed["intent"], result=result,
        latency_s=time.perf_counter() - t0, api_s=api_s,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        reasoning_tokens=_reasoning_tokens(resp),
        error=err,
        detail={"raw": parsed["result_json"][:200]},
        answer=parsed.get("answer"), compose_s=0.0,
    )
