"""route.py 규칙 최적화 — 후보 자동 생성 + 5-fold 교차검증.

검증 표본이 0 이다(273행 전부 burned). 그래서 "적합셋에서 좋아졌다"가
일반화를 뜻하지 않는다는 걸 이번 세션에 두 번 봤다. 새 데이터가 없는 상태에서
일반화를 추정할 방법은 교차검증뿐이다.

두 가지를 분리해서 잰다.

  절차 성능   바깥 5-fold. 각 fold 마다 train 에서만 규칙을 고르고 test 에서
              잰다. 이 평균이 "이 최적화 절차가 새 데이터에서 낼 성능"의
              추정치다. 최종 규칙 집합의 점수가 아니다.
  최종 산출물 전체 데이터로 한 번 더 골라 실제 채택할 규칙 집합을 만든다.
              이것의 적합셋 점수는 낙관적이므로 보고하지 않는다 — 대신 위
              절차 성능을 그 규칙 집합의 기대 성능으로 읽는다.

후보 규칙은 손으로 쓰지 않고 생성한다. 손으로 쓰면 골든을 본 내 눈이 그대로
편향이 된다(A rep1 8/8 이 그렇게 만들어졌다). 생성 규칙은 단순하다 —
"(domain, facet) 이 X 이고 [엔티티 Y 가 있으면] 함수 F 를 더한다".

실행:
    .venv/Scripts/python.exe optimize_route.py
    .venv/Scripts/python.exe optimize_route.py --folds 5 --seed 11
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

import refit_b as F
import route as R

# _out_e*.csv = STEP4 ② 프롬프트(facet=none 제거) 산출물.
# ② 이전 산출물(_out_d*.csv)은 파스 분포가 달라 backup/ 으로 뺐다 —
# 규칙을 그 위에서 고르면 지금 쓰지 않는 분포에 맞추게 된다.
PARSE_SETS = {
    "e": ["_out_e.csv", "_out_e_r2.csv", "_out_e_r3.csv"],
}
MIN_CONTEXT = 8   # 이만큼은 나와야 후보 문맥으로 삼는다
MIN_GAIN = 1e-6


def load_parses(golden, sufs):
    out = defaultdict(list)
    for suf in sufs:
        for s in F.SRCS:
            p = s + suf
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if not r.get("json"):
                        continue
                    k = (s, int(r["idx"]))
                    if k in golden:
                        try:
                            out[k].append(json.loads(r["json"]))
                        except json.JSONDecodeError:
                            pass
    return out


class Cand:
    """(domain, facet)[+엔티티] -> 함수 하나를 더하는 후보 규칙."""

    __slots__ = ("dom", "fac", "ent", "fn", "name")

    def __init__(self, dom, fac, ent, fn):
        self.dom, self.fac, self.ent, self.fn = dom, fac, ent, fn
        e = f"+{ent}" if ent else ""
        self.name = f"{dom}/{fac}{e} → {fn}"

    def fire(self, g):
        if g.domain != self.dom or g.facet != self.fac:
            return False
        return not self.ent or bool(g.ents.get(self.ent))


def build_candidates(keys, parses, golden):
    ctx = Counter()
    ents_in = defaultdict(Counter)
    fn_in = defaultdict(Counter)
    for k in keys:
        for rec in parses[k]:
            for g in R.goals_of(rec):
                key = (g.domain, g.facet)
                ctx[key] += 1
                for t in g.ents:
                    ents_in[key][t] += 1
                for fn in golden[k]:
                    fn_in[key][fn] += 1
    cands = []
    for (dom, fac), n in ctx.items():
        if n < MIN_CONTEXT:
            continue
        ent_opts = [None] + [
            t for t, m in ents_in[(dom, fac)].items() if m >= MIN_CONTEXT
        ]
        # 후보 함수를 CATALOG 전체가 아니라 **그 문맥이 실제로 등장한 행의
        # 골든에 나오는 함수**로 좁힌다. 전수 탐색은 후보가 수천 개라
        # 시간도 문제지만, 애초에 그 문맥에서 한 번도 정답이 아니었던 함수를
        # 후보로 두는 건 잡음만 늘린다.
        fns = fn_in.get((dom, fac), Counter())
        for fn, m in fns.items():
            if m < 3:
                continue
            for e in ent_opts:
                cands.append(Cand(dom, fac, e, fn))
    return cands


# 기존 CV 규칙은 ② 이전 분포에서 고른 것이다. 다시 뽑을 때는 빼고 시작해야
# 새 분포에 맞는 규칙이 나온다. 재선택되면 그건 그것대로 증거다.
BASE_RULES = [r for r in R.ALL_RULES if r not in R.CV_RULES]


def predict(rec, extra):
    fns = R.predict(rec, BASE_RULES)
    if extra:
        for g in R.goals_of(rec):
            for c in extra:
                if c.fire(g):
                    fns.add(c.fn)
    return fns


BETA = 2.0  # 재현율 가중. 누락 손실 > 오호출 손실.


def fbeta(pred, gold, beta=BETA):
    """이 라우터의 목적은 근거 확보다. 필요한 데이터를 못 부르는 손실이
    쓸데없는 걸 하나 더 부르는 손실보다 크므로 재현율에 가중한다."""
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (p + r) else 0.0


def score(keys, parses, golden, extra):
    """행별 F2 를 그 행의 실행들에 평균낸 뒤 매크로 평균."""
    vals = []
    for k in keys:
        recs = parses[k]
        if not recs:
            continue
        vals.append(statistics.mean(
            fbeta(predict(rec, extra), golden[k]) for rec in recs
        ))
    return statistics.mean(vals) if vals else 0.0


def greedy(keys, parses, golden, cands, max_rules=8):
    """train 에서만 도는 전방 선택."""
    chosen, cur = [], score(keys, parses, golden, [])
    for _ in range(max_rules):
        best, bestv = None, cur
        for c in cands:
            if c in chosen:
                continue
            v = score(keys, parses, golden, chosen + [c])
            if v > bestv + MIN_GAIN:
                best, bestv = c, v
        if best is None:
            break
        chosen.append(best)
        cur = bestv
    return chosen, cur


def folds(keys, k, seed):
    by = defaultdict(list)
    for x in keys:
        by[x[0]].append(x)
    rng = random.Random(seed)
    out = [[] for _ in range(k)]
    for s in sorted(by):
        items = sorted(by[s])
        rng.shuffle(items)
        for i, x in enumerate(items):
            out[i % k].append(x)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-rules", type=int, default=8)
    ap.add_argument("--parses", default="e", choices=list(PARSE_SETS))
    args = ap.parse_args()

    golden = F.load_golden()
    parses = load_parses(golden, PARSE_SETS[args.parses])
    keys = sorted(k for k in golden if parses[k])
    print(f"파스셋 {args.parses} · 행 {len(keys)} · 행당 실행 {statistics.mean(len(parses[k]) for k in keys):.1f}회")

    cands = build_candidates(keys, parses, golden)
    print(f"후보 규칙 {len(cands)}개 (문맥 {MIN_CONTEXT}회 이상)\n")

    base_all = score(keys, parses, golden, [])
    print(f"현행 route.py (전체) F2 {base_all:.4f}\n")

    # ── 바깥 CV: 절차의 일반화 성능 ───────────────────────
    print(f"── {args.folds}-fold 교차검증 (train 에서만 선택) ──")
    fs = folds(keys, args.folds, args.seed)
    base_te, opt_te, picked_all = [], [], Counter()
    for i, te in enumerate(fs):
        tr = [x for j, f in enumerate(fs) if j != i for x in f]
        chosen, _ = greedy(tr, parses, golden, cands, args.max_rules)
        b = score(te, parses, golden, [])
        o = score(te, parses, golden, chosen)
        base_te.append(b)
        opt_te.append(o)
        for c in chosen:
            picked_all[c.name] += 1
        print(f"  fold{i+1}  test {len(te):>3}행   현행 {b:.4f} → 최적화 {o:.4f}"
              f"  ({o-b:+.4f})  규칙 {len(chosen)}개")
    mb, mo = statistics.mean(base_te), statistics.mean(opt_te)
    print(f"\n  평균     현행 {mb:.4f} → 최적화 {mo:.4f}   Δ {mo-mb:+.4f}")
    print(f"  fold 별 Δ SD {statistics.stdev([o-b for o,b in zip(opt_te,base_te)]):.4f}")
    wins = sum(1 for o, b in zip(opt_te, base_te) if o > b)
    print(f"  개선된 fold {wins}/{args.folds}")

    print(f"\n── fold 간 재현된 규칙 (몇 개 fold 에서 뽑혔나) ──")
    for name, n in picked_all.most_common(12):
        mark = "  ← 전 fold" if n == args.folds else ""
        print(f"  {n}/{args.folds}  {name}{mark}")

    # ── 최종 산출물 ───────────────────────────────────────
    final, fitv = greedy(keys, parses, golden, cands, args.max_rules)
    print(f"\n── 전체 데이터로 고른 최종 규칙 {len(final)}개 ──")
    print(f"  (적합셋 점수 {fitv:.4f} 는 낙관적이다. 기대 성능은 위 CV 값 {mo:.4f})")
    for c in final:
        print(f"  {picked_all.get(c.name,0)}/{args.folds} fold 재현   {c.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
