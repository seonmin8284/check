"""gpt-5-mini 툴 콜링(function calling) 테스트.

로컬 파이썬 함수 fibonacci() 를 모델에 노출하고,
"피보나치 10번째 구해줘" 같은 요청에 모델이 실제로 그 함수를 호출하는지 확인한다.

실행: python fibonacci_tool_test.py "피보나치 10번째 구해줘"
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-5-mini"


# --- 모델이 호출할 실제 로컬 함수 -------------------------------------------


def fibonacci(n: int) -> list[int]:
    """0번째부터 n번째까지의 피보나치 수열을 반환."""
    if n < 0:
        raise ValueError("n은 0 이상이어야 합니다.")
    seq = [0, 1]
    for _ in range(n - 1):
        seq.append(seq[-1] + seq[-2])
    return seq[: n + 1]


TOOLS = [
    {
        "type": "function",
        "name": "fibonacci",
        "description": (
            "피보나치 수열을 계산한다. 사용자가 피보나치를 요청하면 "
            "직접 계산하지 말고 반드시 이 함수를 호출할 것."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "몇 번째 항까지 구할지 (0-indexed)",
                }
            },
            "required": ["n"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

DISPATCH = {"fibonacci": fibonacci}


# --- 툴 콜링 루프 -----------------------------------------------------------


def run(client: OpenAI, prompt: str) -> str:
    messages: list = [{"role": "user", "content": prompt}]
    called_any = False

    for turn in range(1, 6):  # 무한루프 방지용 상한
        resp = client.responses.create(model=MODEL, input=messages, tools=TOOLS)

        calls = [item for item in resp.output if item.type == "function_call"]
        if not calls:
            if not called_any:
                print("[WARN] 모델이 함수를 호출하지 않고 그냥 답했습니다.")
            return resp.output_text

        messages += resp.output  # 모델의 function_call 아이템을 대화에 되돌려 넣는다
        for call in calls:
            called_any = True
            args = json.loads(call.arguments)
            print(f"[TOOL CALL #{turn}] {call.name}({args})")

            try:
                result = DISPATCH[call.name](**args)
                output = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                output = f"ERROR: {type(e).__name__}: {e}"

            print(f"[TOOL RESULT] {output}")
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    return "[FAIL] 툴 호출 상한(5턴)에 도달했습니다."


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY 가 없습니다. .env 파일을 만들어 주세요.")
        return 1

    prompt = " ".join(sys.argv[1:]) or "피보나치 10번째까지 구해줘"
    print(f"[MODEL] {MODEL}")
    print(f"[PROMPT] {prompt}\n")

    try:
        answer = run(OpenAI(api_key=api_key), prompt)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return 1

    print(f"\n[ANSWER]\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
