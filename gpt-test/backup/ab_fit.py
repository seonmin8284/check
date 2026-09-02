"""A/B 프롬프트 비교를 편향 두 개를 걷어내고 다시 잰다.

앞선 비교에는 결함이 둘 있었다.

  1. capability graph 를 A 의 파스만 보고 손으로 맞췄다. A 방언에 맞춘 그래프로
     B 를 재면 B 가 불리하다. B 도 제 파스로 맞춘 그래프를 줘야 공평하다.
  2. 그 손맞춤을 골든 67행 전부에 대고 했다. 같은 행으로 채점하면 점수가
     낙관적으로 나온다.

그래서 규칙 선택을 **기계화**한다. route.ALL_RULES 를 후보 풀로 두고, fit 셋
에서만 greedy forward selection 으로 부분집합을 고른 뒤 holdout 에서 잰다.
arm 마다 따로 고르므로 각 arm 은 제 파스에 맞춘 그래프를 갖는다.

남는 편향(정직하게 밝혀 둘 것): **후보 규칙 자체는 내가 골든 67행을 다 보고
썼다.** 선택 절차는 기계적이지만 풀은 그렇지 않다. 따라서 holdout 점수도
진짜 신규 데이터 성능의 상한에 가깝다. 이 절차가 제거하는 것은 "어느 규칙을
켤지"의 과적합이지 "어떤 규칙을 상상할지"의 과적합이 아니다.

실행:
    .venv/Scripts/python.exe ab_fit.py
    .venv/Scripts/python.exe ab_fit.py --splits 200
"""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict

import route as R

GOLDEN = "golden_labels.csv"
SUFFIX = {"A": "_out.csv", "B": "_out_b.csv"}


def load_golden():
    out = {}
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fns = {x.strip() for x in r["functions"].split(";") if x.strip()}
            out[(r["source"], int(r["idx"]))] = fns
    return out


def load_arm(sources, suffix, golden):
    """(source, idx) -> [Goal]. 골든에 있는 행만."""
    out = {}
    for s in sources:
        try:
            f = open(s + suffix, encoding="utf-8-sig", newline="")
        except FileNotFoundError:
            continue
        with f:
            for r in csv.DictReader(f):
                if not r.get("json"):
                    continue
                key = (s, int(r["idx"]))
                if key in golden:
                    out[key] = R.goals_of(json.loads(r["json"]))
    return out


def predict(goals, rules):
    fns = set()
    for g in goals:
        fns |= set(R.route(g, rules)[1])
    return fns


def score(keys, parses, golden, rules):
    """micro P/R/F1 + 완전일치 수."""
    tp = fp = fn = exact = 0
    for k in keys:
        got = predict(parses[k], rules)
        want = golden[k]
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
        exact += got == want
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, exact


def greedy_select(fit_keys, parses, golden, pool):
    """fit 셋 F1 을 최대로 하는 규칙 부분집합. 빈 집합(=BASE만)에서 시작.

    선택된 규칙은 원래 순서를 유지한다 — replace=True 규칙이 앞선 결과를
    지우므로 순서가 결과를 바꾼다.
    """
    order = {id(rule): i for i, rule in enumerate(pool)}
    chosen: list = []
    best = score(fit_keys, parses, golden, chosen)[2]
    while True:
        gain, pick = 0.0, None
        for rule in pool:
            if rule in chosen:
                continue
            trial = sorted(chosen + [rule], key=lambda x: order[id(x)])
            f = score(fit_keys, parses, golden, trial)[2]
            if f - best > gain + 1e-12:
                gain, pick = f - best, rule
        if pick is None:
            return chosen, best
        chosen = sorted(chosen + [pick], key=lambda x: order[id(x)])
        best += gain


def backward_select(fit_keys, parses, golden, pool):
    """전체에서 시작해 빼도 손해 없는 규칙을 떨군다.

    greedy forward 는 국소 최적에 걸린다 — 둘이 같이 있어야 이득인 규칙 쌍을
    못 집는다. "B 의 상한" 같은 주장을 하려면 한 방향만 봐서는 안 되므로
    반대 방향도 재고 둘 중 나은 쪽을 쓴다.
    """
    order = {id(rule): i for i, rule in enumerate(pool)}
    chosen = list(pool)
    best = score(fit_keys, parses, golden, chosen)[2]
    while True:
        gain, drop = 0.0, None
        for rule in chosen:
            trial = [x for x in chosen if x is not rule]
            f = score(fit_keys, parses, golden, trial)[2]
            if f - best > gain - 1e-12:  # 동점이면 규칙 수가 적은 쪽
                gain, drop = f - best, rule
        if drop is None or gain < -1e-12:
            return chosen, best
        chosen = sorted(
            (x for x in chosen if x is not drop), key=lambda x: order[id(x)]
        )
        best += gain


def best_select(fit_keys, parses, golden, pool):
    """forward / backward 중 fit F1 이 높은 쪽."""
    fwd = greedy_select(fit_keys, parses, golden, pool)
    bwd = backward_select(fit_keys, parses, golden, pool)
    return fwd if fwd[1] >= bwd[1] else bwd


def stratified_split(keys, frac, rng):
    by_src = defaultdict(list)
    for k in keys:
        by_src[k[0]].append(k)
    fit, hold = [], []
    for src, ks in sorted(by_src.items()):
        ks = sorted(ks)
        rng.shuffle(ks)
        cut = max(1, round(len(ks) * frac))
        cut = min(cut, len(ks) - 1)  # 양쪽 모두 최소 1행
        fit += ks[:cut]
        hold += ks[cut:]
    return fit, hold


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--frac", type=float, default=0.7, help="fit 비중")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    golden = load_golden()
    sources = list(dict.fromkeys(s for s, _ in golden))
    pool = R.ALL_RULES

    arms = {}
    for arm, suf in SUFFIX.items():
        arms[arm] = load_arm(sources, suf, golden)
    common = sorted(set(arms["A"]) & set(arms["B"]))
    print(f"양쪽 arm 파스가 다 있는 골든 행: {len(common)}/{len(golden)}")
    print(f"후보 규칙 {len(pool)}개, 분할 {args.splits}회 (fit {args.frac:.0%})\n")

    # ── 1. 손으로 맞춘 현행 그래프 (규칙 전부) ──────────────
    print("── 1. 현행 그래프, 규칙 전부, 골든 전체 ─────────────")
    print("   (A 파스를 보고 손으로 맞춘 그래프. 여기가 앞서 보고한 숫자다)")
    for arm in ("A", "B"):
        p, r, f, ex = score(common, arms[arm], golden, pool)
        print(f"   {arm}  P={p:.3f} R={r:.3f} F1={f:.3f}  완전일치 {ex}/{len(common)}")

    # ── 2. arm 마다 제 파스로 규칙을 고른 상한 ──────────────
    print("\n── 2. 상한: arm 별로 골든 전체에 맞춰 규칙 선택 ─────")
    print("   (각 arm 이 제 파스에 최적화된 그래프를 받음. 낙관적 상한)")
    ceiling = {}
    for arm in ("A", "B"):
        chosen, fitf1 = best_select(common, arms[arm], golden, pool)
        p, r, f, ex = score(common, arms[arm], golden, chosen)
        ceiling[arm] = (chosen, f)
        print(
            f"   {arm}  P={p:.3f} R={r:.3f} F1={f:.3f}  "
            f"완전일치 {ex}/{len(common)}  규칙 {len(chosen)}/{len(pool)}개 선택"
        )

    # ── 3. fit/holdout ──────────────────────────────────────
    print(f"\n── 3. fit/holdout {args.splits}회 (source 층화) ──────")
    rng = random.Random(args.seed)
    hold_f1 = {"A": [], "B": []}
    hold_ex = {"A": [], "B": []}
    n_rules = {"A": [], "B": []}
    deltas = []
    for _ in range(args.splits):
        fit_keys, hold_keys = stratified_split(common, args.frac, rng)
        per = {}
        for arm in ("A", "B"):
            chosen, _ = best_select(fit_keys, arms[arm], golden, pool)
            p, r, f, ex = score(hold_keys, arms[arm], golden, chosen)
            hold_f1[arm].append(f)
            hold_ex[arm].append(ex / len(hold_keys))
            n_rules[arm].append(len(chosen))
            per[arm] = f
        deltas.append(per["A"] - per["B"])

    for arm in ("A", "B"):
        f = hold_f1[arm]
        print(
            f"   {arm}  holdout F1 {statistics.mean(f):.3f} "
            f"± {statistics.pstdev(f):.3f}   "
            f"완전일치 {statistics.mean(hold_ex[arm]):.1%}   "
            f"선택 규칙 중앙값 {statistics.median(n_rules[arm]):.0f}개"
        )

    wins = sum(d > 1e-9 for d in deltas)
    ties = sum(abs(d) <= 1e-9 for d in deltas)
    print(
        f"\n   A−B holdout F1 차이: 평균 {statistics.mean(deltas):+.3f} "
        f"(중앙값 {statistics.median(deltas):+.3f})"
    )
    print(
        f"   A 우세 {wins}/{args.splits}회, 동률 {ties}회, "
        f"B 우세 {args.splits - wins - ties}회"
    )

    # ── 3b. 행 부트스트랩 ────────────────────────────────────
    # 위 200회는 같은 67행을 돌려쓰므로 독립 시행이 아니다. p-value 를 붙이면
    # 과장이 된다. 대신 행을 복원추출해 상한 그래프끼리 견준다 — 이건
    # "이 골든 셋의 표집 변동"을 재는 올바른 방향이다.
    print("\n── 3b. 행 부트스트랩 2000회 (상한 그래프끼리) ───────")
    rng2 = random.Random(args.seed + 1)
    gaps = []
    for _ in range(2000):
        sample = [common[rng2.randrange(len(common))] for _ in common]
        fa = score(sample, arms["A"], golden, ceiling["A"][0])[2]
        fb = score(sample, arms["B"], golden, ceiling["B"][0])[2]
        gaps.append(fa - fb)
    gaps.sort()
    lo, hi = gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps))]
    print(
        f"   A−B F1 차이 {statistics.mean(gaps):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
    )
    print(f"   A > B 인 표본 {sum(g > 0 for g in gaps)}/2000")

    # ── 4. 어느 규칙이 arm 별로 살아남았나 ───────────────────
    print("\n── 4. 상한 그래프에서 선택된 규칙 ───────────────────")
    a_set = {r.name for r in ceiling["A"][0]}
    b_set = {r.name for r in ceiling["B"][0]}
    for rule in pool:
        mark = ("A" if rule.name in a_set else "·") + (
            "B" if rule.name in b_set else "·"
        )
        if mark != "··":
            print(f"   [{mark}] {rule.name}")
    only = sorted(a_set ^ b_set)
    if only:
        print("\n   arm 에 따라 갈린 규칙:")
        for name in only:
            who = "A만" if name in a_set else "B만"
            print(f"     {who}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
