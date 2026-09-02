"""golden_labels.csv 대비 arm 별 함수 커버리지 채점.

route.py 의 매핑 테이블을 그대로 재사용해 각 발화의 goal 들을 함수 집합으로
바꾼 뒤, 골든 함수 집합과 비교한다. 골든은 다중 라벨이므로 set 단위 P/R/F1.

실행:
    .venv/Scripts/python.exe score_ab.py
"""

import csv
import json
import os
import sys
from collections import Counter

from route import predict as route_predict
from route_intent import predict as intent_predict

GOLDEN = "golden_labels.csv"

# 골든의 source 는 곧 파일 접두사다. A 는 <src>_out.csv, B 는 <src>_out_b.csv.
# 양쪽 산출물이 다 있는 source 만 채점한다 — 한쪽만 있으면 비교가 성립하지 않는다.
SUFFIX = {"A": "_out.csv", "B": "_out_b.csv", "C": "_out_c.csv"}

# arm 마다 파스 스키마가 다르다. A/B 는 goal 분해를, C 는 intent 를 낸다.
# 함수로 바꾸는 단계만 갈아끼우고 채점은 똑같이 한다.
PREDICT = {"A": route_predict, "B": route_predict, "C": intent_predict}


def discover_sources() -> list[str]:
    seen, out = set(), []
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            s = r["source"]
            if s in seen:
                continue
            seen.add(s)
            if all(os.path.exists(s + suf) for suf in SUFFIX.values()):
                out.append(s)
            else:
                print(f"[SKIP] {s} — 양쪽 arm 산출물이 갖춰지지 않음", file=sys.stderr)
    return out


SOURCES = discover_sources()
ARMS = {
    arm: {src: src + suf for src in SOURCES} for arm, suf in SUFFIX.items()
}


def load_golden() -> dict[tuple[str, int], set[str]]:
    out = {}
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["source"], int(r["idx"]))
            out[key] = {x.strip() for x in r["functions"].split(";") if x.strip()}
    return out


def predict(path: str, predictor) -> dict[int, set[str]]:
    """*_out.csv -> {idx: 호출 함수 집합}. SUBSUMED/AMBIGUOUS/UNMAPPED 는 호출이 없다."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("json"):
                continue
            out[int(r["idx"])] = predictor(json.loads(r["json"]))
    return out


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden = load_golden()

    results = {}
    for arm, files in ARMS.items():
        per_row = []
        for source, path in files.items():
            for idx, pred in predict(path, PREDICT[arm]).items():
                # 파스는 있는데 골든 라벨이 없는 행(스모크 이후 확장분)은 건너뛴다.
                if (source, idx) not in golden:
                    continue
                gold = golden[(source, idx)]
                per_row.append((source, idx, pred, gold, *prf(pred, gold)))
        results[arm] = per_row

    print("=" * 74)
    print("골든 함수 집합 대비 (route.py 로 확정된 함수만 채점)")
    print("=" * 74)
    print(f"{'':6}{'P':>8}{'R':>8}{'F1':>8}{'완전일치':>12}{'오호출0':>12}")
    for arm, rows in results.items():
        n = len(rows)
        p = sum(r[4] for r in rows) / n
        r_ = sum(r[5] for r in rows) / n
        f = sum(r[6] for r in rows) / n
        exact = sum(1 for x in rows if x[2] == x[3])
        clean = sum(1 for x in rows if x[2] <= x[3])  # 골든 밖 함수를 안 부른 행
        print(f"{arm:6}{p:8.3f}{r_:8.3f}{f:8.3f}{exact:>9}/{n}{clean:>9}/{n}")

    print()
    print("── source 별 ──────────────────────────────────────")
    hdr = "".join(f"{a+'(P/R/F1)':<22}" for a in ARMS)
    print(f"  {'source':<11}{'n':>3}   {hdr}")
    for source in SOURCES:
        cells, f1 = [], {}
        for arm, rows in results.items():
            sub = [x for x in rows if x[0] == source]
            n = len(sub)
            p = sum(x[4] for x in sub) / n
            r_ = sum(x[5] for x in sub) / n
            f = sum(x[6] for x in sub) / n
            f1[arm] = f
            cells.append(f"{p:.3f}/{r_:.3f}/{f:.3f}")
        print(f"  {source:<11}{n:>3}   " + "".join(f"{c:<22}" for c in cells))
    print()

    print("── arm 쌍별 승패 (행 단위 F1) ─────────────────────")
    by_arm = {arm: {(x[0], x[1]): x for x in rows} for arm, rows in results.items()}
    keys = sorted(by_arm["A"], key=lambda k: (k[0], k[1]))
    names = list(ARMS)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            w = Counter()
            for k in keys:
                fx, fy = by_arm[x][k][6], by_arm[y][k][6]
                w["=" if abs(fx - fy) < 1e-9 else (x if fx > fy else y)] += 1
            print(
                f"  {x} vs {y}:  {x} 우세 {w[x]}행 / "
                f"{y} 우세 {w[y]}행 / 동률 {w['=']}행"
            )

    print("\n── 행별 상세 (한 arm 이라도 완전일치가 아닌 행) ───")
    for k in keys:
        fs = {a: by_arm[a][k][6] for a in names}
        if all(abs(f - 1.0) < 1e-9 for f in fs.values()):
            continue
        print(
            f"  {k[0]:<10}{k[1]:>3}  "
            + "  ".join(f"{a}={fs[a]:.2f}" for a in names)
        )
        print(f"        gold: {sorted(by_arm['A'][k][3])}")
        for a in names:
            if abs(fs[a] - 1.0) > 1e-9:
                print(f"        {a}   : {sorted(by_arm[a][k][2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
