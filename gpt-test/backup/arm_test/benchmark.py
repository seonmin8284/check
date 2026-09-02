"""Arm A(분류+capability graph) vs Arm B(단건 fat-prompt) 벤치마크.

실행 예:
  python benchmark.py                  # 전체 평가셋 1회
  python benchmark.py --limit 6        # 앞 6케이스만 (스모크)
  python benchmark.py --repeat 3       # 지연시간 안정화용 3회 반복
  python benchmark.py --arm A          # 한쪽만
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

import arms
import graph
from dataset import CASES
from edge_cases import EDGE_CASES, NOTES


def normalize(value):
    """부동소수/정수 표기 차이를 흡수해 결과값을 비교 가능한 형태로 만든다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return int(f) if f.is_integer() else round(f, 4)
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items())}
    return value


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(int(round((p / 100) * (len(s) - 1))), len(s) - 1)
    return s[idx]


def summarize(records: list[dict], arm: str) -> dict:
    rows = [r for r in records if r["arm"] == arm]
    if not rows:
        return {}
    ok = [r for r in rows if r["api_error"] is None]
    lat = [r["latency_s"] for r in ok]
    return {
        "arm": arm,
        "n": len(rows),
        "api_errors": len(rows) - len(ok),
        "intent_acc": sum(r["intent_ok"] for r in rows) / len(rows),
        "result_acc": sum(r["result_ok"] for r in rows) / len(rows),
        "lat_mean": statistics.mean(lat) if lat else 0.0,
        "lat_p50": percentile(lat, 50),
        "lat_p95": percentile(lat, 95),
        "in_tok": statistics.mean([r["input_tokens"] for r in ok]) if ok else 0,
        "out_tok": statistics.mean([r["output_tokens"] for r in ok]) if ok else 0,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 케이스만")
    ap.add_argument("--repeat", type=int, default=1, help="평가셋 반복 횟수")
    ap.add_argument("--arm", choices=["A", "B", "both"], default="both")
    ap.add_argument("--compose", action="store_true",
                    help="최종 자연어 문장까지 LLM 이 작성 (A는 API 2회가 된다)")
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high"],
                    default="low", help="reasoning effort (저지연 조건 탐색용)")
    ap.add_argument("--edge", action="store_true",
                    help="경계 케이스 셋(edge_cases.py)으로 실행")
    ap.add_argument("--verbose", action="store_true",
                    help="케이스마다 예측 의도/슬롯을 상세 출력")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[FAIL] OPENAI_API_KEY 가 없습니다.")
        return 1

    arms.REASONING_EFFORT = args.effort
    source = EDGE_CASES if args.edge else CASES
    cases = source[: args.limit] if args.limit else source
    client = OpenAI()
    runners = {"A": arms.run_arm_a, "B": arms.run_arm_b}
    active = ["A", "B"] if args.arm == "both" else [args.arm]

    # 정답은 gold 라벨을 결정적 그래프에 통과시켜 만든다
    gold = {}
    for utt, intent, slots in cases:
        run = graph.execute(intent, slots)
        if not run.ok:
            print(f"[FAIL] 평가셋 오류 — {utt!r}: {run.error}")
            return 1
        gold[utt] = (intent, normalize(run.result))

    if not args.no_warmup:
        print("[WARMUP] 커넥션/캐시 예열 중...")
        for a in active:
            runners[a](client, "5 팩토리얼")

    total = len(cases) * args.repeat * len(active)
    print(f"[RUN] {len(cases)}케이스 x {args.repeat}회 x {len(active)}arm = {total}호출\n")

    records: list[dict] = []
    done = 0
    t_start = time.perf_counter()

    for rep in range(args.repeat):
        for utt, _, _ in cases:
            g_intent, g_result = gold[utt]
            for a in active:
                r = runners[a](client, utt, compose=args.compose)
                done += 1

                intent_ok = r.intent == g_intent
                result_ok = intent_ok and normalize(r.result) == g_result
                records.append({
                    "rep": rep,
                    "arm": a,
                    "utterance": utt,
                    "gold_intent": g_intent,
                    "pred_intent": r.intent,
                    "intent_ok": intent_ok,
                    "result_ok": result_ok,
                    "latency_s": r.latency_s,
                    "api_s": r.api_s,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "api_error": r.error,
                    "compose_s": r.compose_s,
                    "answer": r.answer,
                    "reasoning_tokens": r.reasoning_tokens,
                    "effort": args.effort,
                    "gold_result": g_result,
                    "pred_result": normalize(r.result),
                    "detail": r.detail,
                })

                mark = "OK " if result_ok else ("INT" if intent_ok else "XX ")
                print(f"  [{done:>3}/{total}] {a} {mark} {r.latency_s:5.2f}s  {utt[:34]}")
                if args.verbose:
                    slots = r.detail.get("slots", "-")
                    print(f"        gold={g_intent}  pred={r.intent}  slots={slots}")
                    print(f"        result={str(normalize(r.result))[:64]}")

    wall = time.perf_counter() - t_start
    summaries = [s for s in (summarize(records, a) for a in active) if s]

    print(f"\n{'='*74}\n총 소요 {wall:.1f}s / 모델 {arms.MODEL} / effort={arms.REASONING_EFFORT}\n{'='*74}")
    hdr = f"{'arm':<4}{'의도정확도':>11}{'결과정확도':>11}{'평균':>8}{'p50':>7}{'p95':>7}{'in_tok':>9}{'out_tok':>9}"
    print(hdr)
    print("-" * 74)
    for s in summaries:
        print(f"{s['arm']:<4}{s['intent_acc']:>10.1%}{s['result_acc']:>11.1%}"
              f"{s['lat_mean']:>7.2f}s{s['lat_p50']:>6.2f}s{s['lat_p95']:>6.2f}s"
              f"{s['in_tok']:>9.0f}{s['out_tok']:>9.0f}")
        if s["api_errors"]:
            print(f"     ! 실행/출력파싱 오류 {s['api_errors']}건")

    for a in active:
        misses = [r for r in records if r["arm"] == a and not r["intent_ok"]]
        if misses:
            print(f"\n[Arm {a} 의도 오분류 {len(misses)}건]")
            for g, p in Counter((r["gold_intent"], r["pred_intent"]) for r in misses).most_common():
                print(f"  {g} -> {p}")
        wrong = [r for r in records if r["arm"] == a and r["intent_ok"] and not r["result_ok"]]
        if wrong:
            print(f"\n[Arm {a} 의도는 맞았으나 결과값 틀림 {len(wrong)}건]")
            for r in wrong[:5]:
                print(f"  {r['utterance'][:30]!r}")
                print(f"    정답 {str(r['gold_result'])[:70]}")
                print(f"    예측 {str(r['pred_result'])[:70]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "records": records}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n[SAVED] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
