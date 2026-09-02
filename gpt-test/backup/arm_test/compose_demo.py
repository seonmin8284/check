"""데이터셋 한 건을 두 arm 에 통과시켜 단계별로 들여다보는 스크립트.

최종 문장을 LLM 이 쓰게 했을 때 각 단계가 무엇을 주고받는지 확인하는 용도.

  python compose_demo.py            # 기본 케이스
  python compose_demo.py --case 20  # 20번 케이스
  python compose_demo.py --utterance "피보나치 25까지 중 소수만"   # 임의 발화
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import arms
import graph
from dataset import CASES


def show(title: str, body: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("  " + body.replace("\n", "\n  "))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=22, help="dataset.CASES 인덱스")
    ap.add_argument("--utterance", help="데이터셋 대신 직접 발화 지정")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[FAIL] OPENAI_API_KEY 가 없습니다.")
        return 1

    if args.utterance:
        utt, gold_intent, gold_slots = args.utterance, None, None
    else:
        utt, gold_intent, gold_slots = CASES[args.case]

    client = OpenAI()
    print("=" * 72)
    print(f"발화: {utt}")
    if gold_intent:
        gold = graph.execute(gold_intent, gold_slots)
        print(f"정답: intent={gold_intent} slots={gold_slots}")
        print(f"      result={str(gold.result)[:60]}")
    print("=" * 72)

    # ---- Arm A: 분류 -> 그래프 -> 작성 (API 2회) --------------------------
    print("\n\033[36m┌─ Arm A ─ 분류 → capability graph → 작성 (API 2회)\033[0m")
    a = arms.run_arm_a(client, utt, compose=True)
    show("① [API #1] 분류기 출력 — 계산은 안 함",
         json.dumps({"intent": a.intent, **a.detail["slots"]}, ensure_ascii=False))
    show("② [파이썬] 실행된 노드 — LLM 개입 없음",
         " → ".join(a.detail["trace"]) or "(없음)")
    show("③ [파이썬] 그래프가 확정한 결과값 — 이 값이 근거가 된다",
         str(a.result))
    show("④ [API #2] 이 값만 보고 LLM 이 쓴 문장",
         a.answer or "(작성 안 함)")
    print(f"\n  총 {a.latency_s:.2f}s = 분류 {a.api_s:.2f}s + 작성 {a.compose_s:.2f}s"
          f"  |  토큰 in={a.input_tokens} out={a.output_tokens}")

    # ---- Arm B: 단건 호출이 전부 처리 (API 1회) ---------------------------
    print("\n\033[33m┌─ Arm B ─ fat prompt 단건 호출 (API 1회)\033[0m")
    b = arms.run_arm_b(client, utt, compose=True)
    show("① [API #1] 분류 + 계산 + 작성이 한 응답에 같이 나옴",
         f"intent={b.intent}\nresult={str(b.result)[:120]}")
    show("② 같은 응답의 answer 필드", b.answer or "(작성 안 함)")
    print(f"\n  총 {b.latency_s:.2f}s  |  토큰 in={b.input_tokens} out={b.output_tokens}")

    # ---- 대조 -------------------------------------------------------------
    print("\n" + "=" * 72)
    same = a.result == b.result
    print(f"두 arm 의 결과값 일치: {'예' if same else '아니오'}")
    if not same:
        print(f"  A: {str(a.result)[:60]}")
        print(f"  B: {str(b.result)[:60]}")
    print(f"지연: A {a.latency_s:.2f}s (2회 합) vs B {b.latency_s:.2f}s (1회)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
