"""D 의 재현성을 무엇이 깎는가 — 분산을 프롬프트 단계에 귀속시킨다.

같은 질의를 3회 돌려 파스가 갈리는 자리를 찾고, 그것을 프롬프트 어느 단계의
산물인지로 분류한다. 단계는 이렇게 대응된다.

    STEP 0  EVIDENCE INVENTORY   -> goal 이 몇 개로 쪼개지는가
    STEP 4  CLASSIFY             -> 각 goal 의 (domain, facet, type, horizon)
    STEP 2  ENTITIES             -> 어떤 엔티티를 뽑는가
    STEP 3  CONSTRAINTS          -> 어떤 제약을 뽑는가

STEP 0 은 출력에 안 나온다(사고 단계다). 그래서 그 효과는 goal 개수와 각
goal 의 target 으로만 관측된다 — 근거를 몇 조각으로 셌는지가 곧 분화 수다.

귀속은 **계층적**이다. goal 개수가 다르면 그 아래는 비교가 성립하지 않으므로
거기서 끊는다. 개수가 같아야 분류를 비교하고, 분류가 같아야 엔티티를 본다.

그리고 갈리는 것과 **답이 바뀌는 것**은 다르다. 라우팅을 통과시킨 함수 집합이
같으면 그 분산은 무해하다. 두 축을 교차해서 본다 — 고칠 값어치가 있는 자리는
"자주 갈리는 곳"이 아니라 "갈렸을 때 답이 바뀌는 곳"이다.

실행:
    .venv/Scripts/python.exe var_source.py
"""

import csv
import itertools
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import refit_b as F
import route as R

REPS = {1: "_out_e.csv", 2: "_out_e_r2.csv", 3: "_out_e_r3.csv"}

# 귀속 라벨 (계층 순서대로)
GOAL_N = "goal 개수 (STEP 0 근거 인벤토리)"
CLASSIFY = "goal 분류 (STEP 4 domain/facet/type)"
HORIZON = "horizon (STEP 4)"
ENTITY = "엔티티 추출 (STEP 2)"
CONSTRAINT = "제약 추출 (STEP 3)"
DEPEND = "의존 관계 (STEP 5)"
VALUE = "값 문자열만 (무해할 수 있음)"
ORDER = [GOAL_N, CLASSIFY, HORIZON, ENTITY, CONSTRAINT, DEPEND, VALUE]


def load(golden):
    out = defaultdict(dict)
    for rep, suf in REPS.items():
        for s in F.SRCS:
            p = s + suf
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
                        out[k][rep] = json.loads(r["json"])
                    except json.JSONDecodeError:
                        pass
    return out


def sig(rec):
    """파스를 단계별 지문으로 쪼갠다."""
    goals = rec.get("goals", [])
    return {
        "n": len(goals),
        "cls": Counter((g["domain"], g["facet"], g["type"]) for g in goals),
        "hz": Counter(g.get("horizon") for g in goals),
        "ent": Counter(e["type"] for e in rec.get("entities", [])),
        "con": Counter(c["type"] for c in rec.get("constraints", [])),
        "dep": len(rec.get("dependencies", [])),
    }


def attribute(a, b):
    """두 파스가 갈리는 첫 자리를 반환. 같으면 None."""
    sa, sb = sig(a), sig(b)
    if sa["n"] != sb["n"]:
        return GOAL_N
    if sa["cls"] != sb["cls"]:
        return CLASSIFY
    if sa["hz"] != sb["hz"]:
        return HORIZON
    if sa["ent"] != sb["ent"]:
        return ENTITY
    if sa["con"] != sb["con"]:
        return CONSTRAINT
    if sa["dep"] != sb["dep"]:
        return DEPEND
    if json.dumps(a, sort_keys=True, ensure_ascii=False) != \
       json.dumps(b, sort_keys=True, ensure_ascii=False):
        return VALUE
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden = F.load_golden()
    parses = load(golden)
    full = {k: v for k, v in parses.items() if len(v) == len(REPS)}
    print(f"3회 파스가 다 있는 행 {len(full)}/{len(golden)}")
    if len(full) < 50:
        print("[대기] D 반복 실행이 아직 안 끝났다.")
        return 1

    keys = sorted(full)
    # ── 안정성 ────────────────────────────────────────────
    same_parse = sum(
        1 for k in keys
        if len({json.dumps(full[k][r], sort_keys=True, ensure_ascii=False)
                for r in REPS}) == 1
    )
    fn = {k: {r: R.predict(full[k][r]) for r in REPS} for k in keys}
    same_fn = sum(1 for k in keys if len({frozenset(fn[k][r]) for r in REPS}) == 1)
    print(f"  파스 3회 동일   {same_parse}/{len(keys)} ({same_parse/len(keys):.0%})")
    print(f"  함수 3회 동일   {same_fn}/{len(keys)} ({same_fn/len(keys):.0%})"
          "   ← 재현성의 실질 지표\n")

    # ── 실행별 F1 ─────────────────────────────────────────
    f1s = []
    for r in REPS:
        v = statistics.mean(F.f1(fn[k][r], golden[k]) for k in keys)
        f1s.append(v)
        print(f"  rep{r} 매크로 F1 {v:.4f}")
    print(f"  평균 {statistics.mean(f1s):.4f}  SD {statistics.stdev(f1s):.4f}\n")

    # ── 분산 귀속 ─────────────────────────────────────────
    # 행마다 3개 쌍(1-2,1-3,2-3)을 보고 가장 상위 단계를 그 행의 원인으로 삼는다
    cause = Counter()
    cause_breaks_fn = Counter()
    for k in keys:
        labels = [attribute(full[k][a], full[k][b])
                  for a, b in itertools.combinations(REPS, 2)]
        labels = [x for x in labels if x]
        if not labels:
            continue
        top = min(labels, key=ORDER.index)
        cause[top] += 1
        if len({frozenset(fn[k][r]) for r in REPS}) > 1:
            cause_breaks_fn[top] += 1

    print("── 분산의 출처 (행 단위, 가장 상위 단계로 귀속) ──")
    print(f"  {'단계':<34}{'갈린 행':>8}{'답이 바뀐 행':>12}{'전파율':>8}")
    for lab in ORDER:
        n = cause[lab]
        if not n:
            continue
        b = cause_breaks_fn[lab]
        print(f"  {lab:<34}{n:>8}{b:>12}{b/n:>8.0%}")
    tot, totb = sum(cause.values()), sum(cause_breaks_fn.values())
    print(f"  {'합계':<34}{tot:>8}{totb:>12}{totb/tot:>8.0%}")

    print("\n── 답이 바뀐 행에 대한 기여도 ─────────────────")
    for lab, b in cause_breaks_fn.most_common():
        print(f"  {lab:<34}{b:>4}건  {b/totb:>6.0%}")

    # ── goal 개수 흔들림의 크기 ───────────────────────────
    print("\n── goal 개수 분포 (STEP 0 이 얼마나 흔들리나) ──")
    spread = Counter()
    for k in keys:
        ns = [len(full[k][r].get("goals", [])) for r in REPS]
        spread[max(ns) - min(ns)] += 1
    for d in sorted(spread):
        print(f"  최대-최소 = {d}: {spread[d]:>3}행 ({spread[d]/len(keys):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
