"""배포용 — 세 프롬프트의 파스 산출물을 합쳐 최종 함수 집합을 낸다.

dev 45행에서 고른 구성이고 holdout 22행에서 방향이 확인됐다.

    B 단일 (1콜)          dev 0.8767   holdout 0.8440
    A+B+C 다수결 (3콜)    dev 0.9101   holdout 0.8625

임계값은 **합집합(1표)** 이 기본이다. 이 라우터의 목적은 근거 확보이고,
필요한 데이터를 못 부르는 손실이 하나 더 부르는 손실보다 크다. 재현율
가중(F2) 으로 재면 순위가 뒤집힌다 — D×3 기준 합집합 0.7435 > 단일
0.7139 > 다수결 0.7005 > 만장일치 0.6647. 다수결은 소수 의견을 버리는
장치라 단일 실행보다도 나쁘다.

같은 3콜 예산이면 한 프롬프트를 세 번 돌리는 것보다 서로 다른 프롬프트를
한 번씩 돌리는 게 낫다. 같은 프롬프트의
오류는 서로 상관돼 있어(오류 자카드 0.632) 반복해도 같은 자리에서 같이
틀리지만, 다른 프롬프트는 덜 상관돼 있어(0.501) 투표가 실제 상쇄를 만든다.

사용:
    # 1) 세 프롬프트를 각각 한 번씩 돌린다
    .venv/Scripts/python.exe run_csv.py   --input q.csv --output q_out.csv
    .venv/Scripts/python.exe run_csv_b.py --input q.csv --output q_out_b.csv
    .venv/Scripts/python.exe run_csv_c.py --input q.csv --output q_out_c.csv

    # 2) 합친다
    .venv/Scripts/python.exe ensemble_predict.py q_out.csv q_out_b.csv q_out_c.csv \\
        --output q_functions.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter

from route import predict as route_predict
from route_intent import predict as intent_predict

# 파일명으로 스키마를 고른다. C 는 intent 스키마라 라우터가 다르다.
def predictor_for(path: str):
    return intent_predict if path.endswith("_c.csv") else route_predict


def load(path: str) -> dict[int, set[str]]:
    pred = predictor_for(path)
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("json"):
                continue
            try:
                out[int(r["idx"])] = pred(json.loads(r["json"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                out[int(r["idx"])] = set()  # 파스 실패 = 기권. 투표에서 빠진다
    return out


def queries(path: str) -> dict[int, str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {int(r["idx"]): r.get("input", "") for r in csv.DictReader(f)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("parses", nargs="+", help="*_out.csv 들 (2개 이상)")
    ap.add_argument("--output", default="ensemble_functions.csv")
    ap.add_argument("--threshold", type=int, default=1,
                    help="이 표 이상 받은 함수만. 기본 1 = 합집합")
    args = ap.parse_args()

    if len(args.parses) < 2:
        print("[FAIL] 파스 파일이 2개 이상 필요하다.")
        return 1

    votes = [load(p) for p in args.parses]
    th = args.threshold
    qs = queries(args.parses[0])
    idxs = sorted(set().union(*(set(v) for v in votes)))

    print(f"[VOTE] 파스 {len(votes)}개, 임계 {th}표, {len(idxs)}행")

    n_abstain = 0
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "query", "n_functions", "functions", "votes"])
        for i in idxs:
            c = Counter()
            for v in votes:
                c.update(v.get(i, set()))
            picked = sorted(f for f, n in c.items() if n >= th)
            if not picked:
                n_abstain += 1
            w.writerow([
                i, qs.get(i, ""), len(picked), ";".join(picked),
                ";".join(f"{f}:{n}" for f, n in c.most_common()),
            ])

    print(f"[DONE] {args.output}"
          + (f"  (호출 없음 {n_abstain}행)" if n_abstain else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
