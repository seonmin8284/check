"""간선 단위 회계 — 네트워크형 검토의 결론이자, 그 결론의 상시 검사.

capability graph 를 가중 네트워크로 바꾸면 무엇이 달라지는지 재봤고, 답은
"성능은 안 달라진다"였다. 이유가 지표에 있다. Fβ 에서 호출을 하나 더 붙였을
때의 기대 변화를 micro 혼동행렬에서 풀면

    M = (1+β²)tp,  N = (1+β²)tp + β²fn + fp
    E[ΔF] ∝ q(1+β²)N − M   →   붙일 가치가 있다 ⟺ q > F_β/(1+β²) =: τ*

β=2 에서 τ* ≈ 0.148 이다. 이렇게 낮은 임계값에서는 가중 네트워크의 출력이
합집합으로 붕괴한다 — 어떤 간선이든 하나 붙으면 이미 τ* 를 넘는다. 실제로
누적 지지수에 임계값을 걸어봐도 ≥1(=합집합)이 최적이었고, 다른 뺄셈 장치도
전부 손해였다. 닫은 방향은 파일 끝 CLOSED 참조.

살아남은 것은 **회계 단위**뿐이다. 규칙 단위로 세면 가치를 정반대로 읽는다.

    themed issuer→news channel → get_news         단독지지  8/104   q_m 0.625
    themed issuer→news channel → get_stock_news   단독지지 77/104   q_m 0.234

행 단위 정밀도는 전자가 0.97, 후자가 0.40 이라 후자가 나빠 보이지만, 규칙의
재현율을 지고 있는 것은 후자다. 전자는 발화의 92%가 이미 다른 경로가 부른
자리다. 그래서 여기서는 **한계 정밀도 q_m** 을 쓴다 — 그 간선이 유일 지지자일
때의 정답률. 간선 채택 여부는 q_m 과 τ* 의 비교 하나로 끝나고, 조합 탐색이
필요 없다(optimize_route 의 전방선택, refit_b 의 후방제거를 대체한다).

이 스크립트가 검사하는 주장:

    현재 간선 집합에는 q_m < τ* 인 간선이 하나도 없다.

이게 성립하면 그래프는 네트워크형의 고정점이고, 더 뺄 것이 없다는 뜻이다.
깨지면 exit code 1 을 낸다.

재개 조건도 함께 찍는다. τ* 는 β 의 함수이므로 지표가 바뀌면 간선 집합이
바뀐다. 이 프로젝트가 F1 과 F2 사이를 두 번 오간 이력(route.py 의 RETIRED)이
정확히 그것이고, β 표가 그 이력을 재유도한다.

실행:
    .venv/Scripts/python.exe edge_audit.py
    .venv/Scripts/python.exe edge_audit.py --beta 1     # F1 로 돌아가면
"""

import argparse
import statistics
import sys
from collections import defaultdict

import optimize_route as O
import refit_b as F
import route as R

MIN_SOLO = 10  # 단독 지지가 이만큼은 돼야 q_m 을 신뢰한다


def marginal_precision(keys, parses, golden):
    """{(규칙, 함수): (단독지지 횟수, 그 중 정답)} — 간선의 한계 정밀도."""
    solo = defaultdict(lambda: [0, 0])
    row = defaultdict(lambda: [0, 0])
    for k in keys:
        for rec in parses[k]:
            sup = defaultdict(list)
            for g in R.goals_of(rec):
                for fn in R.route(g, [])[1]:
                    sup[fn].append(("BASE", fn))
                for rule in R.ALL_RULES:
                    if not rule.fire(g):
                        continue
                    add = (R.judgment_bundle(g)
                           if rule.name == "assessment bundle" else rule.add)
                    for fn in add:
                        sup[fn].append((rule.name, fn))
            for fn, es in sup.items():
                ok = fn in golden[k]
                for e in set(es):
                    row[e][0] += 1
                    row[e][1] += ok
                if len(set(es)) == 1:
                    e = es[0]
                    solo[e][0] += 1
                    solo[e][1] += ok
    return solo, row


def micro_fbeta(keys, parses, golden, beta):
    tp = fp = fn = 0
    for k in keys:
        for rec in parses[k]:
            got, want = R.predict(rec), golden[k]
            tp += len(got & want)
            fp += len(got - want)
            fn += len(want - got)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if b2 * p + r else 0.0
    return f, p, r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=2.0)
    args = ap.parse_args()

    golden = F.load_golden()
    parses = O.load_parses(golden, O.PARSE_SETS["e"])
    keys = sorted(k for k in golden if parses[k])
    reps = statistics.mean(len(parses[k]) for k in keys)
    print(f"행 {len(keys)} × 파스 {reps:.0f}회 · 규칙 {len(R.ALL_RULES)}개")

    solo, row = marginal_precision(keys, parses, golden)
    edges = sorted(
        ((n, f, h / t, t, row[(n, f)][1] / row[(n, f)][0])
         for (n, f), (t, h) in solo.items() if t >= MIN_SOLO and n != "BASE"),
        key=lambda x: x[2],
    )

    fb, p, r = micro_fbeta(keys, parses, golden, args.beta)
    tau = fb / (1 + args.beta * args.beta)
    print(f"micro F{args.beta:g} {fb:.4f}  P {p:.3f} R {r:.3f}"
          f"   →  τ* = F/(1+β²) = {tau:.4f}\n")

    print("── 간선 (한계 정밀도 오름차순) ─────────────────────────")
    print(f"  {'여유':>5} {'q_m':>6} {'q_행':>6} {'단독':>5}  간선")
    for n, f, qm, t, qr in edges:
        mark = "  ← τ* 미달" if qm < tau else ""
        print(f"  {qm/tau:>4.1f}배 {qm:>6.3f} {qr:>6.3f} {t:>5}  {n} → {f}{mark}")

    below = [e for e in edges if e[2] < tau]
    print(f"\n── 고정점 검사 ────────────────────────────────────────")
    if below:
        print(f"  τ* 미달 간선 {len(below)}개 — 고정점이 아니다. 잘라야 한다:")
        for n, f, qm, t, _ in below:
            print(f"    {n} → {f}   q_m {qm:.3f}  단독 {t}")
    else:
        print(f"  τ* 미달 간선 0개. 최저 간선이 τ* 의 {edges[0][2]/tau:.1f}배 "
              f"({edges[0][0]} → {edges[0][1]}, q_m {edges[0][2]:.3f}).")
        print("  현재 간선 집합은 네트워크형의 고정점이다 — 뺄셈으로 얻을 것이 없다.")

    print("\n── 재개 조건: β 가 바뀌면 ─────────────────────────────")
    print(f"  {'β':>4} {'F_β':>7} {'τ*':>7}  τ* 미달 간선")
    for beta in (0.5, 1.0, 1.5, 2.0, 3.0):
        f_, _, _ = micro_fbeta(keys, parses, golden, beta)
        t_ = f_ / (1 + beta * beta)
        dead = [f"{n} → {f}" for n, f, qm, _, _ in edges if qm < t_]
        s = "(없음)" if not dead else f"{len(dead)}개: " + ", ".join(dead[:2])
        if len(dead) > 2:
            s += f" 외 {len(dead)-2}"
        print(f"  {beta:>4.1f} {f_:>7.4f} {t_:>7.4f}  {s}")
    print("  β ≥ 1.5 에서는 간선 집합이 불변이다. F1 로 되돌릴 때만 다시 열린다.")

    return 1 if below else 0


# ─────────────────────────────────────────────────────────────
# CLOSED — 네트워크형에서 재보고 닫은 방향 (2026-09-02)
#
# 전부 "뺄셈" 장치다. τ* 가 0.148 로 낮아 뺄 자격이 있는 간선이 없다는 것이
# 공통 원인이고, 다시 열려면 β 를 1 이하로 내리는 결정이 선행돼야 한다.
#
#   누적 임계값      3파스 × goal × 규칙 지지수에 임계값. ≥1(=합집합)이 최적.
#                    지지수와 정밀도는 단조 상관하지만(1개 0.542 → 4개 0.957)
#                    F2 에서 현금화되지 않는다. 정보가 있는 것과 쓸 수 있는
#                    것은 다르다.
#   의존성 억제      파서가 dependencies 를 주는데(파스의 26%, binds 92%가
#                    context) route.py 는 안 읽는다. 하류 goal 의 호출 정밀도
#                    0.699 < 고립 0.796 로 신호는 실재. 그러나 억제하면
#                    F2 0.788 → 0.748. 0.699 는 τ* 의 4.7배라 자를 수 없다.
#   의도 마스크      skill_route.ALLOWED 로 출력을 거르면 P 0.693 → 0.774,
#                    F2 0.788 → 0.792 (골든 의도 = oracle, 재현율 상한 0.968).
#                    그러나 상위 2개만 쓰면 0.774, 20% 오분류면 0.692 로
#                    무마스크보다 나쁘다. 완전한 다중레이블 분류가 전제다.
#                    ALLOWED 를 생성적으로 쓰는 것(∪)은 F2 0.685 로 논외.
#
# 살아 있는 덧셈 레버는 파스 다양성뿐이다. 합집합 한계수익이 1→2 +0.0160,
# 2→3 +0.0079 로 아직 양수다(4번째 ≈ +0.004 외삽). 규칙 변경으로 얻은 어떤
# 값보다 크다 — 다음 이득은 라우터가 아니라 파서 쪽에 있다.
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(main())
