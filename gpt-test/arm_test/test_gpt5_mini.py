"""gpt-5-mini 연결 테스트 스크립트.

실행: python test_gpt5_mini.py "질문 내용"
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-5-mini"


def main() -> int:
    # Windows 콘솔에서 한글이 깨지지 않도록
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY 가 없습니다. .env 파일을 만들어 주세요.")
        return 1

    prompt = " ".join(sys.argv[1:]) or "안녕! 한 문장으로 자기소개 해줘."
    client = OpenAI(api_key=api_key)

    print(f"[MODEL] {MODEL}")
    print(f"[PROMPT] {prompt}\n")

    try:
        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
    except Exception as e:  # 인증/모델 접근 권한/네트워크 오류 구분용
        print(f"[FAIL] {type(e).__name__}: {e}")
        return 1

    print("[OUTPUT]")
    print(resp.output_text)

    usage = resp.usage
    if usage:
        print(
            f"\n[USAGE] input={usage.input_tokens} "
            f"output={usage.output_tokens} total={usage.total_tokens}"
        )

    print("\n[OK] 정상 동작합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
