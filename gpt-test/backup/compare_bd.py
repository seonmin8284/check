"""B vs D(=B-intent) 비교 — 각 arm 에 제 파스로 맞춘 그래프를 준다.

현행 route.py 는 refit_b.py 가 **B 의 파스만 보고** 고른 그래프다. 그걸로 D 를
재면 D 가 불리하다. 이 프로젝트가 반복해서 걸려 넘어진 함정이고(A rep1 편향
8/8), ab_fit.py 가 세운 대응이 "arm 마다 제 그래프를 준다"였다.

그래서 네 칸을 다 잰다.

               B 그래프    D 그래프
    B 파스        (1)        (3)
    D 파스        (2)        (4)

  (1) vs (4)   각자 최적 그래프에서의 공정 비교  ← 결론은 여기서 낸다
  (1) vs (2)   현행(B 튜닝) 그래프에서 D 가 얼마나 손해 보는가
  (2) vs (4)   그 손해 중 얼마가 그래프 커플링 때문인가

파스는 arm 당 1회 추출이다. B 의 실행 간 SD 가 0.0046 이었으므로(3차 측정)
단일 추출 잡음은 이 비교에서 지배적이지 않다. 그래도 결론은 그 폭 안에서만
읽어야 한다.

실행:
    .venv/Scripts/python.exe compare_bd.py
"""

import csv
import json
import os
import statistics
import sys

import refit_b as F
import route as R

SRCS = F.SRCS
# arm 당 행마다 파스 하나씩. B 는 rep1 이 ext_edge 를 안 덮으므로 거기만 rep2.
ARM_FILES = {
    "B": {s: (s + "_out_b_r2.csv" if s == "ext_edge" else s + "_out_b.csv")
          for s in SRCS},
    "D": {s: s + "_out_d.csv" for s in SRCS},
}

ORIG_BUNDLE = {
    ("issuer", "current"): ("get_company_evaluation", "get_financial_data"),
    ("issuer", "forward"): ("get_company_evaluation", "get_financial_data",
                            "get_stock_news"),
    ("issuer", "past"): ("get_stock_news",),
}


def load_arm(arm, golden):
    out = {}
    for s, path in ARM_FILES[arm].items():
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("json"):
                    continue
                k = (s, int(r["idx"]))
                if k not in golden:
                    continue
                try:
                    out[k] = json.loads(r["json"])
                except json.JSONDecodeError:
                    pass
    return out


def score(keys, parse, golden, rules):
    """micro P/R/F1 + 완전일치. parse 는 {key: 레코드} 하나씩."""
    tp = fp = fn = exact = 0
    for k in keys:
        if k not in parse:
            continue
        got = R.predict(parse[k], rules)
        want = golden[k]
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
        exact += got == want
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, exact


def macro(keys, parse, golden, rules):
    v = [F.f1(R.predict(parse[k], rules), golden[k]) for k in keys if k in parse]
    return statistics.mean(v) if v else 0.0


def fit_graph(keys, parse, golden):
    """그 arm 의 파스로 규칙 부분집합을 고른다 (backward, 발화 0회는 보류)."""
    fire = {}
    for k in keys:
        if k not in parse:
            continue
        for g in R.goals_of(parse[k]):
            for rule in R.ALL_RULES:
                try:
                    if rule.fire(g):
                        fire[rule.name] = fire.get(rule.name, 0) + 1
                except Exception:
                    pass
    keep = list(R.ALL_RULES)
    cur = macro(keys, parse, golden, keep)
    while True:
        best, bestv = None, cur
        for rule in list(keep):
            if fire.get(rule.name, 0) == 0:
                continue
            trial = [x for x in keep if x is not rule]
            v = macro(keys, parse, golden, trial)
            if v > bestv + 1e-9:
                best, bestv = rule, v
        if best is None:
            break
        keep = [x for x in keep if x is not best]
        cur = bestv
    return keep, cur


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden, split = F.load_golden(), F.load_split()
    for s, v in ORIG_BUNDLE.items():
        R.JUDGMENT_BUNDLE[s] = v

    parses = {a: load_arm(a, golden) for a in ("B", "D")}
    common = sorted(set(parses["B"]) & set(parses["D"]))
    print(f"양쪽 파스가 다 있는 행 {len(common)}/{len(golden)}")
    for a in ("B", "D"):
        miss = len(golden) - len(parses[a])
        print(f"  {a}: {len(parses[a])}행" + (f" (누락 {miss})" if miss else ""))
    if len(common) < 100:
        print("\n[대기] D 실행이 아직 안 끝났다.")
        return 1
    print()

    graphs = {"현행(B튜닝)": list(R.ALL_RULES)}
    for a in ("B", "D"):
        rules, v = fit_graph(common, parses[a], golden)
        graphs[f"{a}자체적합"] = rules
        dropped = [r.name for r in R.ALL_RULES if r not in rules]
        print(f"{a} 파스로 재적합: 규칙 {len(R.ALL_RULES)} → {len(rules)}개, "
              f"매크로 {v:.4f}")
        print(f"    제거: {dropped if dropped else '(없음)'}")
    print()

    hdr = "파스 / 그래프"
    print(f"  {hdr:<14}" + "".join(f"{g:>16}" for g in graphs))
    cells = {}
    for a in ("B", "D"):
        row = f"  {a:<14}"
        for gname, rules in graphs.items():
            f1_, p, r, e = score(common, parses[a], golden, rules)
            cells[(a, gname)] = f1_
            row += f"{f1_:>16.4f}"
        print(row)

    print("\n── 판정 ───────────────────────────────────────")
    b_own = cells[("B", "B자체적합")]
    d_own = cells[("D", "D자체적합")]
    b_cur = cells[("B", "현행(B튜닝)")]
    d_cur = cells[("D", "현행(B튜닝)")]
    print(f"  각자 최적 그래프:  B {b_own:.4f}  vs  D {d_own:.4f}   "
          f"Δ {d_own-b_own:+.4f}")
    print(f"  현행 그래프에서 :  B {b_cur:.4f}  vs  D {d_cur:.4f}   "
          f"Δ {d_cur-b_cur:+.4f}")
    print(f"  D 가 자기 그래프로 얻는 이득: {d_own-d_cur:+.4f}  "
          f"← 이게 크면 현행 비교가 D 에 불리했던 것")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
