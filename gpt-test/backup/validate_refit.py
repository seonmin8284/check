"""재적합안 검증 — sealed 58행, 사전 선언한 4개 구성만.

적합셋 215행에서 나온 후보를 근거 두께 순으로 늘어놓고, sealed 에서 한 번에
잰다. 구성은 **sealed 를 보기 전에** 확정했다.

    현행          아무것도 안 바꾼 것
    최소안        단독으로 이득이 확인된 변경만 (comparative assessment 제거)
    보수안        + forward 묶음에 stock_news, themed issuer 규칙 제거
                  (근거 14행 / 발화 54회)
    전체 재적합    + current 묶음을 stock_news 단독으로 (근거 **4행**)

'전체'와 '보수'의 차이가 곧 "4행짜리 근거가 일반화되는가"에 대한 답이다.
지난 라운드에 1행짜리 근거를 기각했던 것과 같은 판단을 이번엔 실측으로
확인한다.

실행:
    .venv/Scripts/python.exe validate_refit.py --confirm
"""

import sys

import refit_b as F
import route as R

ORIG = {
    ("issuer", "current"): ("get_company_evaluation", "get_financial_data"),
    ("issuer", "forward"): ("get_company_evaluation", "get_financial_data"),
    ("issuer", "past"): ("get_stock_news",),
}

FWD_NEWS = ("get_company_evaluation", "get_financial_data", "get_stock_news")

# (이름, 판단묶음 덮어쓰기, 제거할 규칙, 근거 설명)
CONFIGS = [
    ("현행", None, set(), "—"),
    ("최소안", None, {"comparative assessment"}, "단독 +0.0065, 발화 12회"),
    ("보수안", {("issuer", "forward"): FWD_NEWS},
     {"comparative assessment", "themed issuer→news channel"},
     "+ forward 14행 / themed 54회"),
    ("전체 재적합", {("issuer", "forward"): FWD_NEWS,
                ("issuer", "current"): ("get_stock_news",)},
     {"comparative assessment", "themed issuer→news channel"},
     "+ current 근거 4행 ← 얇음"),
]


def apply(bundles):
    for s, v in ORIG.items():
        R.JUDGMENT_BUNDLE[s] = v
    for s, v in (bundles or {}).items():
        R.JUDGMENT_BUNDLE[s] = v


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--confirm" not in sys.argv:
        print("sealed 58행을 여는 것은 되돌릴 수 없다. --confirm 을 붙여라.")
        return 1

    golden, split = F.load_golden(), F.load_split()
    parses = F.load_parses(golden)
    fit = sorted(k for k in golden if split.get(k) in ("dev", "burned"))
    seal = sorted(k for k in golden if split.get(k) == "sealed")

    print(f"!! sealed {len(seal)}행 개봉. 이 뒤로 burned 다.\n")
    base = list(R.ALL_RULES)

    print(f"  {'구성':<14}{'적합215':>9}{'SEALED58':>10}{'P':>7}{'R':>7}"
          f"{'완전일치':>9}   근거")
    rows = []
    for name, bundles, drop, why in CONFIGS:
        apply(bundles)
        rules = [r for r in base if r.name not in drop]
        ff = F.micro(fit, parses, golden, rules)[0]
        sf, sp_, sr, se = F.micro(seal, parses, golden, rules)
        rows.append((name, ff, sf))
        print(f"  {name:<14}{ff:>9.4f}{sf:>10.4f}{sp_:>7.3f}{sr:>7.3f}"
              f"{se:>6}/{len(seal)}   {why}")
    apply(None)

    print("\n── 판정 ───────────────────────────────────────")
    b_fit, b_seal = rows[0][1], rows[0][2]
    for name, ff, sf in rows[1:]:
        d_fit, d_seal = ff - b_fit, sf - b_seal
        keep = "채택" if d_seal > 0 else "기각"
        print(f"  {name:<14} 적합 {d_fit:+.4f} → sealed {d_seal:+.4f}   {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
