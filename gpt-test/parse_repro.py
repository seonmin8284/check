"""프롬프트 재현성을 **구조 유사도**로 측정 — route.py 를 통과시키지 않는다.

지금까지의 재현성 수치는 "함수 집합이 같은가"였다. 그건 프롬프트와 라우터를
합친 성능이라, 라우터가 무시하는 차이(target 문구 등)는 안정적으로 보이고
라우터가 증폭하는 차이는 불안정하게 보인다.

그리고 파스를 **완전일치**로 재면 너무 엄격하다. goal 3개 중 2개가 같아도
0점이 된다. 실제로 중요한 것은 "구조가 얼마나 겹치는가"다. 그래서 성분마다
다중집합 자카드를 쓴다 — |교집합| / |합집합| 이라 부분 겹침이 점수로 남는다.

    1.00  세 실행의 구조가 완전히 같다
    0.50  절반만 겹친다
    0.00  전혀 안 겹친다

goal 을 순서로 짝짓지 않는 이유: 같은 질의라도 실행마다 goal 순서가 달라진다.
순서를 무시한 다중집합으로 봐야 "같은 구조를 냈는가"를 잰다.

층위는 위로 갈수록 자유도가 크다(=안 맞기 쉽다).

    target 문구      자유 텍스트. 구조가 아니라 서술.
    엔티티 (타입,값)   무엇을 어떤 문자열로 뽑았나
    엔티티 타입       무엇을 뽑았나 (값 무시) ← route.py 팬아웃의 입력
    goal 분류        (domain, facet, type, horizon)
    goal 골격        (domain, facet) 만 — 가장 거친 구조
    goal 개수        몇 조각으로 쪼갰나

실행:
    .venv/Scripts/python.exe parse_repro.py
"""

import csv
import itertools
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import refit_b as F

# 지난 arm(A/B/D 구판) 산출물은 backup/ 에 있다. 그때 결론은 세 arm 의 파스
# 안정성이 층위별로 1~3%p 차이로 사실상 동률이었다는 것이다 — arm 선택으로
# 재현성이 갈리지 않는다.
ARMS = {
    "E": ["_out_e.csv", "_out_e_r2.csv", "_out_e_r3.csv"],
}


def load(sufs, golden):
    out = defaultdict(dict)
    for i, suf in enumerate(sufs):
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
                            out[k][i] = json.loads(r["json"])
                        except json.JSONDecodeError:
                            pass
    return {k: v for k, v in out.items() if len(v) == len(sufs)}


# ── 성분 지문: 전부 Counter(다중집합) 로 돌려 자카드를 매긴다 ──

def C_target(rec):
    return Counter(g.get("target", "").strip() for g in rec.get("goals", []))


def C_ent_tv(rec):
    return Counter((e["type"], e["value"].strip())
                   for e in rec.get("entities", []))


def C_ent_t(rec):
    return Counter(e["type"] for e in rec.get("entities", []))


def C_cls(rec):
    return Counter((g["domain"], g["facet"], g["type"], g.get("horizon"))
                   for g in rec.get("goals", []))


def C_df(rec):
    return Counter((g["domain"], g["facet"]) for g in rec.get("goals", []))


def C_facet(rec):
    return Counter(g["facet"] for g in rec.get("goals", []))


def C_domain(rec):
    return Counter(g["domain"] for g in rec.get("goals", []))


def C_con(rec):
    return Counter(c["type"] for c in rec.get("constraints", []))


LAYERS = [
    ("target 문구", C_target),
    ("엔티티 (타입,값)", C_ent_tv),
    ("엔티티 타입", C_ent_t),
    ("제약 타입", C_con),
    ("goal 분류 +horizon", C_cls),
    ("goal 골격 (dom,facet)", C_df),
    ("  facet 만", C_facet),
    ("  domain 만", C_domain),
]


def jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 1.0


def count_sim(a: int, b: int) -> float:
    """goal 개수 유사도 — min/max. 2 vs 3 이면 0.67."""
    if a == b:
        return 1.0
    return min(a, b) / max(a, b) if max(a, b) else 1.0


def pairwise(recs, keys, fn, n):
    """행마다 실행 쌍의 자카드를 평균낸 뒤, 행 단위로 매크로 평균."""
    return statistics.mean(
        statistics.mean(
            jaccard(fn(recs[k][a]), fn(recs[k][b]))
            for a, b in itertools.combinations(range(n), 2)
        )
        for k in keys
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden = F.load_golden()
    arms = {a: load(s, golden) for a, s in ARMS.items()}
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    print("실행 수: " + ", ".join(f"{a}×{len(ARMS[a])}" for a in ARMS))
    print(f"3회 파스가 다 있는 행 {len(common)}\n")

    def table(keys, arm_list, label):
        print(f"── {label} ({len(keys)}행) · 구조 유사도 (1.00 = 매번 같은 구조)")
        print(f"  {'층위':<22}" + "".join(f"{a:>9}" for a in arm_list))
        for name, fn in LAYERS:
            row = f"  {name:<22}"
            for a in arm_list:
                n = len(ARMS[a])
                row += f"{pairwise(arms[a], keys, fn, n):>9.2f}"
            print(row)
        # goal 개수는 자카드가 아니라 min/max
        row = f"  {'goal 개수':<22}"
        for a in arm_list:
            n = len(ARMS[a])
            v = statistics.mean(
                statistics.mean(
                    count_sim(len(arms[a][k][x].get("goals", [])),
                              len(arms[a][k][y].get("goals", [])))
                    for x, y in itertools.combinations(range(n), 2))
                for k in keys)
            row += f"{v:>9.2f}"
        print(row + "\n")

    table(common, list(ARMS), "현행 arm(E)")

    # ── source 별로 구조가 어디서 흔들리나 (D) ───────────
    print("── source 별 구조 유사도 (E) ───────────────────")
    by = defaultdict(list)
    for k in arms["E"]:
        by[k[0]].append(k)
    print(f"  {'source':<11}{'행':>4}{'goal골격':>9}{'엔티티타입':>10}{'분류+hz':>9}")
    for s in sorted(by, key=lambda x: -len(by[x])):
        ks = by[s]
        print(f"  {s:<11}{len(ks):>4}"
              f"{pairwise(arms['E'], ks, C_df, 3):>9.2f}"
              f"{pairwise(arms['E'], ks, C_ent_t, 3):>10.2f}"
              f"{pairwise(arms['E'], ks, C_cls, 3):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
