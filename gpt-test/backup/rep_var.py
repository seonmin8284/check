"""arm 내 반복 분산 측정 — A/B 격차가 실행 잡음보다 큰지 판정.

지금까지의 A/B 비교는 arm 당 파스 1회 추출에 기대고 있었다. 부트스트랩 CI 는
*행* 재표집 분산만 잡고 *파스* 재추출 분산은 못 잡는다. gpt-5-mini 는
temperature/seed 통제가 없으므로 같은 프롬프트도 매번 다른 파스를 낸다.

여기서는 arm 당 3회 실행을 놓고 두 가지를 잰다.

  1. 실행 간 분산 — 같은 프롬프트를 다시 돌리면 F1 이 얼마나 흔들리는가
  2. 그 잡음 대비 A−B 격차 — 격차가 잡음 안에 묻히는가

행 단위로는 3회 평균 F1 을 쓴다(반복 평균이 행 잡음을 줄인다). 그 위에서
67행 짝지은 부호검정/Wilcoxon 을 돌린다.

실행:
    .venv/Scripts/python.exe rep_var.py
"""

import csv
import itertools
import json
import os
import statistics
import sys
from collections import defaultdict

from route import predict as route_predict
from route_intent import predict as intent_predict

GOLDEN = "golden_labels.csv"
SOURCES = ["work", "invest", "ext_ipo", "ext_tax", "ext_div", "ext_fx",
           "ext_index", "ext_basis", "ext_edge"]
REPS = (1, 2, 3, 4)
# rep1 은 기존 산출물, rep2~4 는 run_reps*.sh 가 낸 것.
# rep1 은 capability graph 가 그 실행을 보고 작성돼 오염돼 있다(A 8/8 진단).
# 깨끗한 3회 다수결을 성립시키려고 rep4 를 A·B 에만 추가로 돌렸다.
CLEAN_REPS = (2, 3, 4)
SUFFIX = {
    "A": {1: "_out.csv", 2: "_out_r2.csv", 3: "_out_r3.csv", 4: "_out_r4.csv"},
    "B": {1: "_out_b.csv", 2: "_out_b_r2.csv", 3: "_out_b_r3.csv",
          4: "_out_b_r4.csv"},
    "C": {1: "_out_c.csv", 2: "_out_c_r2.csv", 3: "_out_c_r3.csv"},
}
# arm 마다 파스 스키마가 다르다. A/B 는 goal 분해, C 는 intent 를 낸다.
PREDICT = {"A": route_predict, "B": route_predict, "C": intent_predict}


def load_golden():
    out = {}
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source"], int(r["idx"]))] = {
                x.strip() for x in r["functions"].split(";") if x.strip()
            }
    return out


def load_rep(arm, rep, golden):
    """(arm, rep) -> {key: 예측 함수 집합}. 없는 파일은 None."""
    if rep not in SUFFIX[arm]:  # arm 마다 돈 횟수가 다르다 (C 는 3회까지)
        return None
    predictor = PREDICT[arm]
    preds = {}
    for src in SOURCES:
        path = src + SUFFIX[arm][rep]
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("json"):
                    continue
                key = (src, int(r["idx"]))
                if key not in golden:
                    continue
                try:
                    preds[key] = predictor(json.loads(r["json"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    preds[key] = set()  # 파스 실패는 빈 호출로 계상
    return preds


def load_rep_raw(arm, rep, golden):
    """(arm, rep) -> {key: 파스 원본 레코드}.

    load_rep 는 라우팅까지 끝낸 함수 집합을 준다. 규칙을 갈아끼우며 다시
    라우팅하려면 원본이 필요하다 (refit_graph.py 가 쓴다).
    """
    if rep not in SUFFIX[arm]:
        return None
    out = {}
    for src in SOURCES:
        path = src + SUFFIX[arm][rep]
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("json"):
                    continue
                key = (src, int(r["idx"]))
                if key in golden:
                    try:
                        out[key] = json.loads(r["json"])
                    except json.JSONDecodeError:
                        out[key] = {"goals": [], "entities": [], "constraints": []}
    return out


def f1(pred, gold):
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def micro(keys, preds, golden):
    tp = fp = fn = 0
    for k in keys:
        got, want = preds[k], golden[k]
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def sign_test(diffs):
    """양측 부호검정 p. n<=25 는 정확 이항, 그 이상은 정규근사."""
    pos = sum(1 for d in diffs if d > 1e-9)
    neg = sum(1 for d in diffs if d < -1e-9)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    tot = 2 ** n
    tail = sum(
        __import__("math").comb(n, i) for i in range(k + 1)
    )
    return min(1.0, 2 * tail / tot), pos, neg


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden = load_golden()

    reps = {}
    for arm in SUFFIX:
        for rep in REPS:
            got = load_rep(arm, rep, golden)
            if got is None:
                print(f"[MISS] {arm} rep{rep} 산출물 없음 — 건너뜀", file=sys.stderr)
                continue
            reps[(arm, rep)] = got

    arms = sorted({a for a, _ in reps})
    keys = sorted(set.intersection(*(set(v) for v in reps.values())))
    print(f"채점 행 {len(keys)}개 · 실행 {len(reps)}개 "
          f"({', '.join(f'{a}×{sum(1 for x,_ in reps if x==a)}' for a in arms)})\n")

    # ── 1. 실행별 전체 F1 ────────────────────────────────
    print("── 1. 실행별 micro F1 ─────────────────────────")
    per_arm = defaultdict(list)
    for (arm, rep), preds in sorted(reps.items()):
        v = micro(keys, preds, golden)
        exact = sum(1 for k in keys if preds[k] == golden[k])
        per_arm[arm].append(v)
        print(f"  {arm} rep{rep}   F1={v:.4f}   완전일치 {exact}/{len(keys)}")

    print("\n── 2. arm 내 실행 간 분산 ─────────────────────")
    for arm in arms:
        vs = per_arm[arm]
        sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
        print(f"  {arm}  평균 {statistics.mean(vs):.4f}  SD {sd:.4f}  "
              f"범위 [{min(vs):.4f}, {max(vs):.4f}]  폭 {max(vs)-min(vs):.4f}")

    print("\n── 2b. arm 쌍별 격차와 실행 잡음 ──────────────")
    for a, b in itertools.combinations(arms, 2):
        gap = statistics.mean(per_arm[a]) - statistics.mean(per_arm[b])
        # 승자의 최악 실행이 패자의 최선 실행보다 높아야 순위가 안 뒤집힌다.
        hi, lo = (a, b) if gap >= 0 else (b, a)
        margin = min(per_arm[hi]) - max(per_arm[lo])
        pairs = [
            micro(keys, reps[(a, ra)], golden) - micro(keys, reps[(b, rb)], golden)
            for ra in REPS if (a, ra) in reps
            for rb in REPS if (b, rb) in reps
        ]
        verdict = "잡음에 견딤" if margin > 0 else "잡음으로 뒤집힘"
        print(f"  {a}−{b}  평균 {gap:+.4f}   "
              f"{hi} 최악 vs {lo} 최선 {margin:+.4f} → {verdict}")
        print(f"          실행쌍 {len(pairs)}개: [{min(pairs):+.4f}, "
              f"{max(pairs):+.4f}]  {a} 우세 {sum(1 for d in pairs if d > 0)}쌍")

    # ── 3. 파스 안정성 ──────────────────────────────────
    print("\n── 3. 파스 안정성 (3회 예측이 모두 같은 행) ───")
    for arm in arms:
        rs = [r for (x, r) in reps if x == arm]
        if len(rs) < 2:
            continue
        same = sum(
            1 for k in keys
            if len({frozenset(reps[(arm, r)][k]) for r in rs}) == 1
        )
        print(f"  {arm}  {same}/{len(keys)}  ({same/len(keys):.1%}) 3회 동일")

    # ── 4. 행 단위 짝지은 비교 (3회 평균) ────────────────
    def row_mean(arm, k, rs=REPS):
        return statistics.mean(
            f1(reps[(arm, r)][k], golden[k]) for r in rs if (arm, r) in reps
        )

    print("\n── 4. 행 단위 짝지은 비교 (반복 평균 F1) ──────")
    for a, b in itertools.combinations(arms, 2):
        diffs = [row_mean(a, k) - row_mean(b, k) for k in keys]
        p, pos, neg = sign_test(diffs)
        mean_d = statistics.mean(diffs)
        se = statistics.stdev(diffs) / len(diffs) ** 0.5
        sig = "유의" if p < 0.05 else "유의하지 않음"
        print(f"  {a}−{b}  Δ{mean_d:+.4f}  95%CI "
              f"[{mean_d-1.96*se:+.4f}, {mean_d+1.96*se:+.4f}]  "
              f"부호검정 {a}{pos}:{b}{neg} (동률 {len(diffs)-pos-neg}) "
              f"p={p:.4g} → {sig}")

    # ── 5. rep1 편향 진단 ───────────────────────────────
    # 그래프가 특정 실행을 보고 작성됐다면 그 실행이 모든 source 에서 최고로
    # 나온다. 8개 source 전부를 이기는 건 우연이라면 (1/3)^8 ≈ 0.015% 다.
    print("\n── 5. rep1 편향 진단 (rep1 ≥ rep2·rep3 인 source 수) ──")
    for arm in arms:
        won = sum(
            1 for src in SOURCES
            if (sk := [k for k in keys if k[0] == src])
            and all(
                micro(sk, reps[(arm, 1)], golden) >= micro(sk, reps[(arm, r)], golden)
                for r in REPS[1:] if (arm, r) in reps
            )
        )
        flag = "  ← 그래프가 이 실행에 맞춰졌을 가능성" if won == len(SOURCES) else ""
        print(f"  {arm}  {won}/{len(SOURCES)}{flag}")

    # ── 6. source 별 ────────────────────────────────────
    print("\n── 6. source 별 (반복 평균 micro F1 / 실행폭) ──")
    print(f"  {'source':<11}{'n':>3}  " + "".join(f"{a:>16}" for a in arms))
    for src in SOURCES:
        sk = [k for k in keys if k[0] == src]
        if not sk:
            continue
        cells = []
        for arm in arms:
            v = [micro(sk, reps[(arm, r)], golden) for r in REPS if (arm, r) in reps]
            cells.append(f"{statistics.mean(v):.3f} ±{max(v)-min(v):.3f}")
        print(f"  {src:<11}{len(sk):>3}  " + "".join(f"{c:>16}" for c in cells))

    # ── 7. 신규 추출만 (rep1 제외) ──────────────────────
    fresh = [r for r in REPS[1:]]
    if all((arm, r) in reps for arm in arms for r in fresh):
        print(f"\n── 7. rep1 제외, 신규 추출 {len(fresh)}회만 ──────────")
        for arm in arms:
            v = [micro(keys, reps[(arm, r)], golden) for r in fresh]
            print(f"  {arm}  F1 {statistics.mean(v):.4f}")
        for a, b in itertools.combinations(arms, 2):
            diffs = [row_mean(a, k, fresh) - row_mean(b, k, fresh) for k in keys]
            p, pos, neg = sign_test(diffs)
            sig = "유의" if p < 0.05 else "유의하지 않음"
            print(f"  {a}−{b}  Δ{statistics.mean(diffs):+.4f}  "
                  f"부호검정 {a}{pos}:{b}{neg} p={p:.4g} → {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
