"""capability graph 재적합 — B 기준선, 적합셋 215행, 검증 58행.

3차 재적합(refit_graph.py)은 dev 45행에서 돌렸고 아무것도 채택하지 못했다.
판단 묶음 변경은 dev 1행이 뒷받침했고, 규칙 제거 7개 중 5개는 dev 발화 0회라
증거가 아니라 도박이었다. 골든이 273행이 되면서 그 제약이 풀렸다.

바뀐 것 셋:

  기준선   앙상블이 아니라 **B 단일**. 미노출 116행에서 A 0.5887 < B 0.6187 로
           순위가 뒤집혔고, 낙폭도 A -0.371 > B -0.276 이다. A 는 후보에서 뺀다.
  적합셋   dev 45 + burned 170 = 215행.
  검증셋   sealed 58행. 마지막에 한 번만 연다.

목적함수는 **행별 F1 을 그 행에서 쓸 수 있는 B 실행들에 대해 평균낸 뒤,
행 단위로 매크로 평균**한다. 한 실행의 우연에 규칙을 맞추지 않으려는 것이다 —
이 프로젝트에서 반복된 실패가 정확히 그것이었다(A rep1 편향 8/8).

규칙을 뺄지 결정할 때 **발화 횟수를 함께 본다.** "빼도 안 나빠진다"는 무용의
증거가 아니라 증거 없음일 수 있다. 발화 0회인 규칙은 건드리지 않는다.

실행:
    .venv/Scripts/python.exe refit_b.py
    .venv/Scripts/python.exe refit_b.py --validate   # sealed 개봉 (되돌릴 수 없다)
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

import route as R

GOLDEN = "golden_labels.csv"
SPLIT = "split_frozen.csv"
SRCS = ["work", "invest", "ext_ipo", "ext_tax", "ext_div", "ext_fx",
        "ext_index", "ext_basis", "ext_edge"]
B_SUFFIX = {1: "_out_b.csv", 2: "_out_b_r2.csv", 3: "_out_b_r3.csv",
            4: "_out_b_r4.csv"}


def load_golden():
    out = {}
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source"], int(r["idx"]))] = {
                x.strip() for x in r["functions"].split(";") if x.strip()
            }
    return out


def load_split():
    out = {}
    with open(SPLIT, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source"], int(r["idx"]))] = r["split"]
    return out


def load_parses(golden):
    """{key: [파스 레코드, ...]} — 그 행에서 쓸 수 있는 B 실행 전부."""
    out = defaultdict(list)
    for rep in sorted(B_SUFFIX):
        for s in SRCS:
            p = s + B_SUFFIX[rep]
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if not r.get("json"):
                        continue
                    k = (s, int(r["idx"]))
                    if k not in golden:
                        continue
                    try:
                        out[k].append(json.loads(r["json"]))
                    except json.JSONDecodeError:
                        pass
    return out


def f1(pred, gold):
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def objective(keys, parses, golden, rules):
    """행별 (실행 평균 F1) 의 매크로 평균."""
    vals = []
    for k in keys:
        recs = parses[k]
        if not recs:
            continue
        vals.append(statistics.mean(
            f1(R.predict(rec, rules), golden[k]) for rec in recs
        ))
    return statistics.mean(vals) if vals else 0.0


def micro(keys, parses, golden, rules):
    """비교용 micro P/R/F1 + 완전일치 (행마다 첫 실행만)."""
    tp = fp = fn = exact = 0
    for k in keys:
        if not parses[k]:
            continue
        got = R.predict(parses[k][0], rules)
        want = golden[k]
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
        exact += got == want
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, exact


def firing_counts(keys, parses):
    """규칙별 발화 횟수. 증거 없음과 무용을 가르는 데 쓴다."""
    n = defaultdict(int)
    for k in keys:
        for rec in parses[k]:
            for g in R.goals_of(rec):
                for rule in R.ALL_RULES:
                    try:
                        if rule.fire(g):
                            n[rule.name] += 1
                    except Exception:
                        pass
    return n


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="sealed 58행 개봉 (되돌릴 수 없다)")
    args = ap.parse_args()

    golden, split = load_golden(), load_split()
    parses = load_parses(golden)
    fit = sorted(k for k in golden if split.get(k) in ("dev", "burned"))
    seal = sorted(k for k in golden if split.get(k) == "sealed")
    print(f"적합셋 {len(fit)}행 (dev+burned) · 검증셋 {len(seal)}행 (sealed)")
    reps_per_row = statistics.mean(len(parses[k]) for k in fit)
    print(f"적합셋 행당 B 실행 수 평균 {reps_per_row:.2f}\n")

    base_rules = list(R.ALL_RULES)
    fire = firing_counts(fit, parses)
    base = objective(fit, parses, golden, base_rules)
    bm = micro(fit, parses, golden, base_rules)
    print(f"현행 그래프 (규칙 {len(base_rules)}개)")
    print(f"  목적함수(매크로) {base:.4f}   micro F1 {bm[0]:.4f}  "
          f"P {bm[1]:.3f} R {bm[2]:.3f}  완전일치 {bm[3]}/{len(fit)}\n")

    # ── 1. 판단 묶음 변형 ────────────────────────────────
    print("── 1. JUDGMENT_BUNDLE 변형 ────────────────────")
    cands = {
        ("issuer", "current"): [
            ("현행", ("get_company_evaluation", "get_financial_data")),
            ("+news", ("get_company_evaluation", "get_financial_data",
                       "get_stock_news")),
            ("news 만", ("get_stock_news",)),
            ("eval 만", ("get_company_evaluation",)),
            ("빈 묶음", ()),
        ],
        ("issuer", "forward"): [
            ("현행", ("get_company_evaluation", "get_financial_data")),
            ("+news", ("get_company_evaluation", "get_financial_data",
                       "get_stock_news")),
            ("eval 만", ("get_company_evaluation",)),
            ("fin 만", ("get_financial_data",)),
        ],
        ("issuer", "past"): [
            ("현행", ("get_stock_news",)),
            ("+fin", ("get_stock_news", "get_financial_data")),
            ("+price", ("get_stock_news", "get_stock_price")),
        ],
    }
    cur = base
    for slot, variants in cands.items():
        orig = R.JUDGMENT_BUNDLE.get(slot, ())
        best, bestv = orig, cur
        print(f"  {slot}:")
        for name, val in variants:
            R.JUDGMENT_BUNDLE[slot] = val
            v = objective(fit, parses, golden, base_rules)
            star = ""
            if v > bestv + 1e-9:
                best, bestv, star = val, v, "  ★"
            print(f"    {name:<12} {v:.4f}  ({v-cur:+.4f}){star}")
        R.JUDGMENT_BUNDLE[slot] = best
        cur = max(cur, bestv)
    print(f"\n  판단 묶음 조정 후 {cur:.4f}  ({cur-base:+.4f})\n")

    # ── 2. 규칙 backward elimination ─────────────────────
    print("── 2. 규칙 제거 (발화 0회는 건드리지 않음) ────")
    keep = list(base_rules)
    dropped = []
    improved = True
    while improved:
        improved = False
        best_gain, best_rule, best_v = 0.0, None, cur
        for rule in list(keep):
            if fire.get(rule.name, 0) == 0:
                continue  # 증거 없음 — 판단 보류
            trial = [x for x in keep if x is not rule]
            v = objective(fit, parses, golden, trial)
            if v > best_v + 1e-9:
                best_gain, best_rule, best_v = v - cur, rule, v
        if best_rule is not None:
            keep = [x for x in keep if x is not best_rule]
            cur = best_v
            dropped.append((best_rule.name, fire[best_rule.name], cur))
            improved = True
    for name, nf, v in dropped:
        print(f"  제거  {name:<42} (발화 {nf:>4}회) → {v:.4f}")
    if not dropped:
        print("  (제거로 개선되는 규칙 없음)")
    untested = [r.name for r in base_rules if fire.get(r.name, 0) == 0]
    if untested:
        print(f"\n  발화 0회라 판단 보류한 규칙 {len(untested)}개: "
              f"{', '.join(untested)}")

    final = keep
    fm = micro(fit, parses, golden, final)
    print(f"\n── 3. 적합셋 결과 ─────────────────────────────")
    print(f"  규칙 {len(base_rules)} → {len(final)}개")
    print(f"  목적함수 {base:.4f} → {cur:.4f}  ({cur-base:+.4f})")
    print(f"  micro F1 {bm[0]:.4f} → {fm[0]:.4f}  "
          f"완전일치 {bm[3]} → {fm[3]}/{len(fit)}")

    print(f"\n── 4. 채택 내용 ───────────────────────────────")
    for slot in cands:
        print(f"  JUDGMENT_BUNDLE[{slot}] = {R.JUDGMENT_BUNDLE.get(slot)}")
    rm = [r.name for r in base_rules if r not in final]
    print(f"  제거 규칙: {rm if rm else '(없음)'}")

    if args.validate:
        print("\n!! sealed 58행 개봉. 이 뒤로 그 행들은 burned 다.")
        print("── 5. 검증 (sealed 58행) ──────────────────────")
        for label, rules, bundles in (
            ("현행 그래프", base_rules, None),
            ("재적합 그래프", final, True),
        ):
            if bundles is None:
                saved = dict(R.JUDGMENT_BUNDLE)
                for slot, variants in cands.items():
                    R.JUDGMENT_BUNDLE[slot] = dict(variants)["현행"]
            f, p, r, e = micro(seal, parses, golden, rules)
            print(f"  {label:<14} F1 {f:.4f}  P {p:.3f} R {r:.3f}  "
                  f"완전일치 {e}/{len(seal)}")
            if bundles is None:
                R.JUDGMENT_BUNDLE.clear()
                R.JUDGMENT_BUNDLE.update(saved)
    else:
        print("\n  검증하려면 --validate. sealed 는 한 번만 열 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
