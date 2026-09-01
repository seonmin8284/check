"""capability graph 를 **dev 에서만** 재적합한다 — 3단계.

앞선 그래프는 A rep1 을 보고 손으로 맞췄고(rep1 편향 진단 8/8), 그 결과 A rep1
에서만 0.95 가 나왔다. 여기서는 두 가지를 고친다.

  1. 맞추는 대상을 rep1 이 아니라 **배포할 구성**(3콜 교차 앙상블)으로 둔다.
  2. 맞추는 데이터를 dev 45행으로 제한한다. holdout 22행은 열지 않는다.

탐색 공간은 둘이다.

  규칙 부분집합   ALL_RULES 에서 backward elimination. 빼도 안 나빠지면 뺀다.
                  (적은 규칙이 같은 점수면 그쪽이 일반화가 낫다)
  판단 묶음       JUDGMENT_BUNDLE[("issuer","current")] 변형. 앙상블 잔여
                  오류 8행 중 3행이 이 자리의 get_company_evaluation 과호출이다.
                  horizon 을 arm 마다 다르게 붙이는 게 원인인데, 프롬프트로
                  통일시키는 대신 분기 자체를 덜 민감하게 만든다.

실행:
    .venv/Scripts/python.exe refit_graph.py
"""

import itertools
import statistics
import sys

from ensemble import score, vote
from freeze_split import dev_keys
import rep_var as V
import route as R

ARMS = "ABC"


def build(golden):
    reps = {}
    for a in V.SUFFIX:
        for r in V.REPS:
            x = V.load_rep_raw(a, r, golden)
            if x is not None:
                reps[(a, r)] = x
    return reps


def arm_predict(arm, rec, rules):
    """C 는 intent 스키마라 route 규칙이 안 붙는다. 재적합 대상은 A·B 뿐."""
    if arm == "C":
        return V.PREDICT["C"](rec)
    return R.predict(rec, rules)


def ensemble_score(reps, grids, keys, golden, rules):
    """3콜 교차 앙상블(A+B+C 다수결)의 조합 평균 F1."""
    vals = []
    for picks in itertools.product(*(grids[a] for a in ARMS)):
        preds = {}
        for k in keys:
            sets = [
                arm_predict(a, reps[(a, r)][k], rules)
                for a, r in zip(ARMS, picks)
            ]
            preds[k] = vote(sets, 2)
        vals.append(score(preds, golden, keys)[0])
    return statistics.mean(vals)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden = V.load_golden()
    keys = sorted(dev_keys())
    reps = build(golden)
    grids = {a: [r for r in V.CLEAN_REPS if (a, r) in reps] for a in ARMS}
    print(f"[dev] {len(keys)}행 · rep1 제외 "
          f"({', '.join(f'{a}{grids[a]}' for a in ARMS)})\n")

    base_rules = list(R.ALL_RULES)
    base = ensemble_score(reps, grids, keys, golden, base_rules)
    print(f"현행 그래프, 규칙 {len(base_rules)}개 : F1 {base:.4f}\n")

    # ── 1. 판단 묶음 변형 ────────────────────────────────
    print("── 1. JUDGMENT_BUNDLE[('issuer','current')] 변형 ──")
    orig = R.JUDGMENT_BUNDLE[("issuer", "current")]
    cands = {
        "현행 (eval+fin)": ("get_company_evaluation", "get_financial_data"),
        "+news": ("get_company_evaluation", "get_financial_data", "get_stock_news"),
        "past 와 동일 (news)": ("get_stock_news",),
        "eval+news": ("get_company_evaluation", "get_stock_news"),
        "fin+news": ("get_financial_data", "get_stock_news"),
        "빈 묶음": (),
    }
    best_bundle, best_v = orig, base
    for name, val in cands.items():
        R.JUDGMENT_BUNDLE[("issuer", "current")] = val
        v = ensemble_score(reps, grids, keys, golden, base_rules)
        mark = "  ★" if v > best_v + 1e-9 else ""
        if v > best_v + 1e-9:
            best_bundle, best_v = val, v
        print(f"  {name:<22} F1 {v:.4f}  ({v-base:+.4f}){mark}")
    R.JUDGMENT_BUNDLE[("issuer", "current")] = best_bundle
    print(f"\n  채택: {best_bundle}  F1 {best_v:.4f}\n")

    # ── 2. 규칙 backward elimination ─────────────────────
    print("── 2. 규칙 backward elimination ───────────────")
    keep = list(base_rules)
    cur = best_v
    dropped = []
    improved = True
    while improved:
        improved = False
        for rule in list(keep):
            trial = [x for x in keep if x is not rule]
            v = ensemble_score(reps, grids, keys, golden, trial)
            if v >= cur - 1e-9:  # 안 나빠지면 뺀다 (동점이면 단순한 쪽)
                keep, cur = trial, max(cur, v)
                dropped.append((rule.name, v))
                improved = True
                break
    for name, v in dropped:
        print(f"  제거  {name:<38} → F1 {v:.4f}")
    if not dropped:
        print("  (뺄 수 있는 규칙 없음 — 모두 기여하고 있다)")
    print(f"\n  규칙 {len(base_rules)}개 → {len(keep)}개, F1 {base:.4f} → {cur:.4f}")

    print("\n── 3. 남은 규칙 ───────────────────────────────")
    for rule in keep:
        print(f"  [{rule.layer[0]}] {rule.name}")

    print("\n── 4. 적용 방법 ───────────────────────────────")
    print(f"  JUDGMENT_BUNDLE[('issuer','current')] = {best_bundle}")
    names = [r.name for r in base_rules if r not in keep]
    print(f"  제거할 규칙: {names if names else '(없음)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
