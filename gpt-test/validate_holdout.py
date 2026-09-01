"""홀드아웃 최종 검증 — 딱 두 구성만, 딱 한 번.

사전 선언(dev 에서 결정, holdout 을 보기 전에 고정):

    베이스라인   B 단일 실행                    1콜
    후보         A+B+C 다수결(>=2/3)           3콜

이 둘만 잰다. 여기서 여러 구성을 훑으면 holdout 이 두 번째 dev 가 될 뿐이다.
규칙 재적합(refit_graph.py)의 결과는 **적용하지 않았다** — 판단 묶음 변경은
dev 1행, 규칙 제거는 7개 중 5개가 dev 발화 0회로, 어느 쪽도 45행으로는
뒷받침되지 않는다.

실행:
    .venv/Scripts/python.exe validate_holdout.py --confirm
"""

import itertools
import statistics
import sys

from ensemble import score, vote
from freeze_split import dev_keys, holdout_keys
import rep_var as V

ARMS = "ABC"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--confirm" not in sys.argv:
        print("holdout 을 여는 것은 되돌릴 수 없다. --confirm 을 붙여라.")
        print("연 뒤로 holdout 은 오염된 것으로 간주하고, 다음 라운드에는")
        print("새로 잘라야 한다.")
        return 1

    golden = V.load_golden()
    reps = {}
    for a in V.SUFFIX:
        for r in V.REPS:
            x = V.load_rep(a, r, golden)
            if x is not None:
                reps[(a, r)] = x
    grids = {a: [r for r in V.CLEAN_REPS if (a, r) in reps] for a in ARMS}

    def evaluate(keys):
        # 베이스라인: B 단일 (실행 평균)
        b = [score({k: reps[("B", r)][k] for k in keys}, golden, keys)
             for r in grids["B"]]
        base = tuple(statistics.mean(x[i] for x in b) for i in range(4))

        # 후보: A+B+C 다수결, 실행 조합 전체 평균
        vals = []
        for picks in itertools.product(*(grids[a] for a in ARMS)):
            preds = {
                k: vote([reps[(a, r)][k] for a, r in zip(ARMS, picks)], 2)
                for k in keys
            }
            vals.append(score(preds, golden, keys))
        cand = tuple(statistics.mean(x[i] for x in vals) for i in range(4))
        return base, cand

    print("!! holdout 개봉. 이 뒤로 holdout 은 오염됐다.\n")
    for name, keys in (("dev", sorted(dev_keys())),
                       ("HOLDOUT", sorted(holdout_keys()))):
        base, cand = evaluate(keys)
        print(f"── {name} ({len(keys)}행) ──────────────────────")
        print(f"  {'구성':<22}{'F1':>8}{'P':>8}{'R':>8}{'완전일치':>10}")
        print(f"  {'B 단일 (1콜)':<22}{base[0]:>8.4f}{base[1]:>8.3f}"
              f"{base[2]:>8.3f}{base[3]:>9.1f}/{len(keys)}")
        print(f"  {'A+B+C 다수결 (3콜)':<22}{cand[0]:>8.4f}{cand[1]:>8.3f}"
              f"{cand[2]:>8.3f}{cand[3]:>9.1f}/{len(keys)}")
        print(f"  {'차이':<22}{cand[0]-base[0]:>+8.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
