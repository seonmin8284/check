"""자기일관성 투표 — 파스를 n 회 받아 다수결로 함수 집합을 정한다.

결손의 75% 가 "같은 arm 이 실행마다 다른 답을 낸" 행에 몰려 있다. 파스가 안정한
행은 F1 0.91~0.92, 흔들리는 행은 0.68~0.81 이다. 프롬프트를 안 고치고 이
분산을 줄이는 가장 싼 방법이 n 회 추출 후 투표다.

투표 단위는 **파스가 아니라 함수 집합**이다. 파스 JSON 을 다수결하는 건
구조가 달라 불가능하지만, 라우팅을 통과시킨 뒤의 함수 집합은 그냥 집합이라
빈도를 셀 수 있다.

임계값의 의미:
    th=1 (합집합)   재현율 최대, 정밀도 희생
    th=2 (다수결)   n=3 에서 기본값
    th=3 (교집합)   정밀도 최대, 재현율 희생

실행:
    .venv/Scripts/python.exe ensemble.py            # dev 만
    .venv/Scripts/python.exe ensemble.py --holdout  # 봉인 해제 (마지막 1회)
"""

import argparse
import statistics
import sys
from collections import Counter

from freeze_split import dev_keys, holdout_keys
import rep_var as V


def vote(pred_sets: list[set], threshold: int) -> set:
    """여러 실행의 함수 집합 -> 임계 이상 등장한 함수만."""
    c = Counter()
    for s in pred_sets:
        c.update(s)
    return {f for f, n in c.items() if n >= threshold}


def vote_preds(reps, arm, rs, keys, threshold):
    return {k: vote([reps[(arm, r)][k] for r in rs], threshold) for k in keys}


def score(preds, golden, keys):
    """micro P/R/F1 + 완전일치."""
    tp = fp = fn = 0
    for k in keys:
        got, want = preds[k], golden[k]
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    exact = sum(1 for k in keys if preds[k] == golden[k])
    return f, p, r, exact


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="봉인 해제")
    ap.add_argument("--reps", default="2,3", help="쓸 실행 (rep1 은 그래프에 오염됨)")
    args = ap.parse_args()

    golden = V.load_golden()
    reps = {
        (a, r): V.load_rep(a, r, golden)
        for a in V.SUFFIX
        for r in V.REPS
        if V.load_rep(a, r, golden) is not None
    }
    arms = sorted({a for a, _ in reps})

    keys = sorted(holdout_keys() if args.holdout else dev_keys())
    label = "HOLDOUT" if args.holdout else "dev"
    if args.holdout:
        print("!! holdout 을 열었다. 이 뒤로 holdout 은 오염된 것으로 간주하라.\n")

    use = tuple(int(x) for x in args.reps.split(","))
    print(f"[{label}] {len(keys)}행 · 투표에 쓰는 실행 rep{use}\n")

    print("── 단일 실행 vs 투표 ──────────────────────────")
    print(f"  {'arm':<5}{'구성':<14}{'F1':>8}{'P':>8}{'R':>8}{'완전일치':>10}")
    for arm in arms:
        rs = [r for r in use if (arm, r) in reps]
        if not rs:
            continue
        singles = [score({k: reps[(arm, r)][k] for k in keys}, golden, keys)
                   for r in rs]
        f = statistics.mean(s[0] for s in singles)
        p = statistics.mean(s[1] for s in singles)
        r_ = statistics.mean(s[2] for s in singles)
        e = statistics.mean(s[3] for s in singles)
        print(f"  {arm:<5}{'단일 평균':<14}{f:>8.4f}{p:>8.3f}{r_:>8.3f}{e:>9.1f}/{len(keys)}")
        for th, name in ((1, "합집합"), (2, "다수결"), (len(rs), "교집합")):
            if th > len(rs) or (th == 1 and len(rs) == 1):
                continue
            f, p, r_, e = score(vote_preds(reps, arm, rs, keys, th), golden, keys)
            tag = f"{name} (≥{th}/{len(rs)})"
            print(f"  {'':<5}{tag:<14}{f:>8.4f}{p:>8.3f}{r_:>8.3f}{e:>9}/{len(keys)}")
        print()

    # arm 을 가로지르는 투표 — 서로 다른 프롬프트의 오류는 덜 상관돼 있다.
    if len(arms) > 1:
        print("── arm 교차 투표 (같은 rep, arm 간 다수결) ────")
        for r in use:
            avail = [a for a in arms if (a, r) in reps]
            if len(avail) < 2:
                continue
            th = (len(avail) + 1) // 2
            preds = {
                k: vote([reps[(a, r)][k] for a in avail], th) for k in keys
            }
            f, p, r_, e = score(preds, golden, keys)
            print(f"  rep{r}  {'+'.join(avail)} (≥{th})  "
                  f"F1 {f:.4f}  P {p:.3f} R {r_:.3f}  완전일치 {e}/{len(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
