"""호출 예산을 고정하고 앙상블 구성을 쓸어본다 — dev 전용.

핵심 질문: 같은 n 회 호출을 쓴다면 **한 프롬프트를 n 번** 돌리는 게 나은가,
**서로 다른 프롬프트를 한 번씩** 돌리는 게 나은가.

같은 프롬프트의 오류는 서로 상관돼 있어 반복해도 같은 자리에서 같이 틀린다.
다른 프롬프트는 다른 자리에서 틀리므로 투표가 실제로 상쇄를 만든다. 그게
사실인지 여기서 잰다.

rep1 은 쓰지 않는다 — capability graph 가 그 실행을 보고 작성됐다.

실행:
    .venv/Scripts/python.exe ensemble_sweep.py
"""

import itertools
import statistics
import sys

from ensemble import score, vote
from freeze_split import dev_keys, holdout_keys
import rep_var as V

REPS = V.CLEAN_REPS  # (2, 3, 4)


def run_sets(reps, arm):
    return [r for r in REPS if (arm, r) in reps]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    use_holdout = "--holdout" in sys.argv
    golden = V.load_golden()
    reps = {}
    for a in V.SUFFIX:
        for r in V.REPS:
            got = V.load_rep(a, r, golden)
            if got is not None:
                reps[(a, r)] = got
    keys = sorted(holdout_keys() if use_holdout else dev_keys())
    label = "HOLDOUT" if use_holdout else "dev"
    if use_holdout:
        print("!! holdout 을 열었다. 이 뒤로 holdout 은 오염된 것으로 간주하라.\n")
    print(f"[{label}] {len(keys)}행 · rep1 제외, 사용 실행 {REPS}\n")

    rows = []  # (호출수, 이름, F1, P, R, 완전일치)

    def add(cost, name, preds):
        f, p, r, e = score(preds, golden, keys)
        rows.append((cost, name, f, p, r, e))

    # ── 예산 1: 단일 실행 ────────────────────────────────
    for arm in sorted({a for a, _ in reps}):
        rs = run_sets(reps, arm)
        fs = [score({k: reps[(arm, r)][k] for k in keys}, golden, keys) for r in rs]
        rows.append((1, f"{arm} 단일 (평균 {len(rs)}회)",
                     statistics.mean(x[0] for x in fs),
                     statistics.mean(x[1] for x in fs),
                     statistics.mean(x[2] for x in fs),
                     statistics.mean(x[3] for x in fs)))

    # ── 예산 2~3: 같은 arm 반복 ──────────────────────────
    for arm in sorted({a for a, _ in reps}):
        rs = run_sets(reps, arm)
        for n in (2, 3):
            if len(rs) < n:
                continue
            # n 개 실행 조합을 모두 평균내 특정 실행 운을 지운다
            for th in range(1, n + 1):
                vals = []
                for combo in itertools.combinations(rs, n):
                    preds = {
                        k: vote([reps[(arm, r)][k] for r in combo], th)
                        for k in keys
                    }
                    vals.append(score(preds, golden, keys))
                name = {1: "합집합", n: "만장일치"}.get(th, f"다수결≥{th}")
                rows.append((n, f"{arm}×{n} {name}",
                             *[statistics.mean(v[i] for v in vals) for i in range(4)]))

    # ── 예산 2~3: 서로 다른 arm 한 번씩 ──────────────────
    arms = sorted({a for a, _ in reps})
    for size in (2, 3):
        for combo_arms in itertools.combinations(arms, size):
            # 각 arm 에서 한 실행씩 뽑는 모든 조합을 평균
            for th in range(1, size + 1):
                vals = []
                grids = [run_sets(reps, a) for a in combo_arms]
                for picks in itertools.product(*grids):
                    preds = {
                        k: vote([reps[(a, r)][k] for a, r in zip(combo_arms, picks)],
                                th)
                        for k in keys
                    }
                    vals.append(score(preds, golden, keys))
                name = {1: "합집합", size: "만장일치"}.get(th, f"다수결≥{th}")
                rows.append((size, f"{'+'.join(combo_arms)} {name}",
                             *[statistics.mean(v[i] for v in vals) for i in range(4)]))

    print(f"  {'호출':>4}  {'구성':<22}{'F1':>8}{'P':>8}{'R':>8}{'완전일치':>10}")
    best_by_cost = {}
    for cost, name, f, p, r, e in sorted(rows, key=lambda x: (x[0], -x[2])):
        star = ""
        if cost not in best_by_cost:
            best_by_cost[cost] = f
            star = "  ★"
        print(f"  {cost:>4}  {name:<22}{f:>8.4f}{p:>8.3f}{r:>8.3f}"
              f"{e:>9.1f}/{len(keys)}{star}")

    print("\n── 예산별 최고 구성 ───────────────────────────")
    base = best_by_cost.get(1, 0)
    for cost in sorted(best_by_cost):
        f = best_by_cost[cost]
        name = next(n for c, n, x, *_ in sorted(rows, key=lambda x: (x[0], -x[2]))
                    if c == cost and abs(x - f) < 1e-12)
        print(f"  {cost}회  {name:<22} F1 {f:.4f}  (단일 대비 {f-base:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
