"""D 파스 -> 의도분류표 코드. LLM 에 묻지 않고 goal 에서 유도한다.

D안은 프롬프트에서 STEP 5(INTENT)를 걷어내 intent 를 내지 않는다. 그런데
의도 코드가 라우팅에는 안 쓰여도 리포팅·분석·골든 대조에는 쓸모가 있다.
그걸 LLM 에 질의당 971 토큰을 더 주고 받을 이유는 없다 — goal 이 이미
(domain, facet, type, horizon) + 엔티티를 갖고 있고, 의도는 그것들의 함수다.

route_intent.py 의 INTENT_MAP 은 intent -> (domain, facet, type) 이다. 여기서
하는 일은 그 역방향인데, 역이 함수가 아니다. 다섯 자리가 다대일이다.

    I1-1 / I1-2   개념·용어  vs  상품 설명
    I2-1 / I2-2   세금       vs  법규·규제
    I3-1 / I3-2   앱 사용법  vs  업무 절차
    I6-1 / I6-2   기본 정보  vs  기업 개요
    I6-5 / I6-6   배당       vs  재무

그래서 판별자를 따로 준다. 넷은 엔티티 타입으로 갈리고(app_feature, product),
나머지는 값 문자열에 기댄다 — route.py 의 3층과 같은 성격이라 같은 방식으로
따로 계상한다.

실행:
    .venv/Scripts/python.exe route_intent_d.py          # 골든 대조
"""

import csv
import json
import sys
from collections import Counter

import refit_b as F
import route as R

STRUCT, LEXICON, FALLBACK = "struct", "lexicon", "fallback"

# 값 문자열 판별자 — 구조 신호가 없는 자리
TAX_CUE = ("세금", "세율", "과세", "양도소득", "배당소득", "세액공제",
           "종합과세", "비과세", "원천징수", "금투세", "금융투자소득")
DIVIDEND_CUE = ("배당", "배당금", "배당수익률", "배당성향", "배당기준일", "배당락")
OVERVIEW_CUE = ("개요", "사업", "매출 구성", "사업 구조", "무엇을 하는", "어떤 회사")


def _has(g, *types):
    return any(g.ents.get(t) for t in types)


def classify(g) -> tuple[str, str]:
    """Goal -> (intent_id, 판정 근거층)"""
    d, f, t = g.domain, g.facet, g.type
    hz = g.horizon

    # ── internal: 앱이냐 업무냐 ───────────────────────────
    if d == "internal":
        if f in ("knowledge",):
            return ("I1-2" if _has(g, "product") else "I1-1"), STRUCT
        if f in ("regulation",):
            return ("I2-1" if g.text_has(TAX_CUE) else "I2-2"), LEXICON
        if f == "fx":
            return "I9-1", STRUCT
        # app_feature 가 붙으면 "어디서 보나"(I3-1), 아니면 "무엇을 하나"(I3-2)
        return ("I3-1" if _has(g, "app_feature") else "I3-2"), STRUCT

    # ── finance_legal: 제도 자체 ──────────────────────────
    if d == "finance_legal":
        if f == "knowledge":
            return ("I1-2" if _has(g, "product") else "I1-1"), STRUCT
        if f == "howto":
            return "I2-3", STRUCT
        if f == "ipo":
            return "I4-2", STRUCT
        if f in ("regulation", "disclosure"):
            return ("I2-1" if g.text_has(TAX_CUE) else "I2-2"), LEXICON
        return ("I2-1" if g.text_has(TAX_CUE) else "I2-2"), LEXICON

    # ── 공통 facet ────────────────────────────────────────
    if f == "fx":
        return "I9-1", STRUCT
    if f == "knowledge":
        return ("I1-2" if _has(g, "product") else "I1-1"), STRUCT
    if f == "regulation":
        return ("I2-1" if g.text_has(TAX_CUE) else "I2-2"), LEXICON
    if f == "howto":
        return ("I3-1" if _has(g, "app_feature") else "I3-2"), STRUCT
    if f == "valuation":
        return "I8-1", STRUCT
    if f in ("estimate", "target_price"):
        return "I8-2", STRUCT
    if f == "sector_map":
        return "I5-2", STRUCT
    if f == "ipo":
        return ("I4-1" if d == "issuer" else "I4-2"), STRUCT

    # ── issuer ────────────────────────────────────────────
    if d == "issuer":
        if f == "profile":
            return ("I6-2" if g.text_has(OVERVIEW_CUE) else "I6-1"), LEXICON
        if f == "news":
            return "I6-3", STRUCT
        if f == "disclosure":
            return "I6-4", STRUCT
        if f == "fundamentals":
            if g.text_has(DIVIDEND_CUE):
                return "I6-5", LEXICON
            # 아직 오지 않은 실적은 컨센서스(I8-2), 확정 실적은 재무(I6-6),
            # 원인·영향 해석은 기업 분석(I6-7)
            if hz == "forward":
                return "I8-2", STRUCT
            if t == "analysis":
                return "I6-7", STRUCT
            return "I6-6", STRUCT
        if f == "scoring" or (f == "none" and t in ("assessment", "recommendation")):
            return "I6-8", STRUCT
        if f in ("price", "flow", "short"):
            # 기간이 지목되거나 과거를 보면 추이 분석(I7-2), 아니면 현재(I7-1)
            past = hz == "past" or t == "analysis" or g.has_period
            return ("I7-2" if past else "I7-1"), STRUCT
        if f == "screening":
            return "I7-2", STRUCT

    # ── market ────────────────────────────────────────────
    if d == "market":
        if f == "news":
            # 업종·테마를 경유하면 산업 동향(I5-2), 아니면 시장 전망(I5-1)
            themed = _has(g, "theme", "sector", "market_event")
            return ("I5-2" if themed else "I5-1"), STRUCT
        if f == "none" and t in ("assessment", "recommendation"):
            themed = _has(g, "theme", "sector")
            return ("I5-2" if themed else "I5-1"), STRUCT
        if f == "screening":
            themed = _has(g, "theme", "sector")
            return ("I5-2" if themed else "I7-2"), STRUCT
        if f in ("price", "flow", "short"):
            if t == "assessment":
                return "I5-1", STRUCT
            past = hz == "past" or t == "analysis" or g.has_period
            return ("I7-2" if past else "I7-1"), STRUCT
        if f in ("fundamentals", "profile"):
            return "I5-2", STRUCT
        if f == "disclosure":
            return "I6-4", STRUCT

    return "I9-1", FALLBACK


# ─────────────────────────────────────────────────────────────
# 팬아웃 — goal 하나가 의도 여러 개를 받는 자리
#
# 골든은 질의당 의도 2.36 개를 붙이는데 goal 은 1.55 개다. 그래서 goal 하나에
# 의도 하나를 매기면 재현율이 0.47 에서 막힌다(개수가 충분한 행만 보면 0.76).
# route.py 가 goal 하나에 함수 여러 개를 붙이는 것과 같은 사정이다.
#
# 어떤 쌍을 붙일지는 골든의 공기 빈도가 알려준다. 그리고 그 쌍들이 route.py 의
# EXPANSION 규칙과 그대로 대응한다 — theme→sector 가 I5-2, product→knowledge
# 가 I1-2, document→policy 가 I2-2 다. 같은 신호가 함수와 의도를 동시에 가른다.
# ─────────────────────────────────────────────────────────────

APP_CUE = ("확인", "조회", "어디서", "어디에", "화면", "메뉴", "앱", "보는")
REGULATED_CUE = ("서류", "요건", "미성년", "실명", "한도제한", "휴면", "해지",
                 "자금세탁", "필요성", "가능")


def fanout(g) -> set[str]:
    """주 의도 외에 함께 붙는 의도들."""
    out = set()
    d, f, t = g.domain, g.facet, g.type

    if d == "internal":
        # 절차를 묻되 "어디서 보나"가 섞이면 앱 사용법이 같이 붙는다 (I3-1+I3-2 21회)
        if f == "howto":
            if _has(g, "app_feature") or g.text_has(APP_CUE):
                out |= {"I3-1", "I3-2"}
            # 상품 엔티티가 붙으면 상품 설명이 따라온다 (I1-2+I3-2 20회)
            if _has(g, "product"):
                out.add("I1-2")
            # 서류·요건은 법정 요건 (I2-2+I3-2 13회)
            if _has(g, "document") or g.text_has(REGULATED_CUE):
                out.add("I2-2")

    # 업황을 경유하면 산업 동향이 함께 (I5-2+I6-8 35회, I5-2+I6-3 15,
    # I5-2+I6-6 13, I5-1+I5-2 21)
    if _has(g, "theme", "sector", "market_event"):
        out.add("I5-2")

    if d == "issuer":
        # 판단 목표는 근거가 되는 실적·뉴스도 같이 지목된다
        # (I6-6+I6-8 10회, I6-3+I6-8 11회)
        if f == "none" and t in ("assessment", "recommendation"):
            out.add("I6-8")
            if g.horizon == "forward":
                out.add("I8-2")
        # 실적 분석은 재무 조회를 겸한다 (I6-6+I6-7 8회)
        if f == "fundamentals" and t == "analysis":
            out |= {"I6-6", "I6-7"}
        # 기간이 지목된 시세는 현재가와 추이를 둘 다 본다 (I7-1+I7-2 12회)
        if f in ("price", "flow", "short") and g.has_period:
            out |= {"I7-1", "I7-2"}

    if d == "market":
        # 지수 수준 질의는 시장 전망과 시세가 겹친다 (I5-1+I7-1 12회)
        if f in ("price", "flow", "short") and t == "assessment":
            out |= {"I5-1", "I7-1"}
    return out


def predict(rec: dict) -> set[str]:
    """파스 레코드 -> 의도 코드 집합 (발화 단위)."""
    out = set()
    for g in R.goals_of(rec):
        out.add(classify(g)[0])
        out |= fanout(g)
    return out


def _load(arm_suffix, golden):
    out = {}
    for s in F.SRCS:
        try:
            fh = open(s + arm_suffix, encoding="utf-8-sig", newline="")
        except FileNotFoundError:
            continue
        with fh as f:
            for r in csv.DictReader(f):
                if not r.get("json"):
                    continue
                k = (s, int(r["idx"]))
                if k in golden:
                    try:
                        out[k] = json.loads(r["json"])
                    except json.JSONDecodeError:
                        pass
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden = F.load_golden()
    gi = {}
    with open("golden_labels.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            gi[(r["source"], int(r["idx"]))] = {
                x.strip() for x in r["intent_ids"].split(";") if x.strip()
            }

    D = _load("_out_e.csv", golden)
    print(f"파스 {len(D)}행\n")

    def score(pred_fn, parses, name):
        P = Rc = n = ex = 0
        for k, rec in parses.items():
            want = gi.get(k)
            if not want:
                continue
            got = pred_fn(rec)
            if not got:
                continue
            P += len(got & want) / len(got)
            Rc += len(got & want) / len(want)
            ex += got == want
            n += 1
        p, r = P / n, Rc / n
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {name:<28} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}  "
              f"집합일치 {ex/n:.1%}  (n={n})")
        return p, r

    print("── 골든 intent_ids 대조 ───────────────────────")
    # 과거 비교: B 가 LLM 으로 낸 intent 는 P 0.811 / R 0.477 / F1 0.601.
    # 유도기가 토큰 0 으로 그보다 낫다는 것이 채택 근거였다 (backup/ 참조).
    score(predict, D, "goal 에서 유도")

    print("\n── 유도 근거층 분포 (D) ───────────────────────")
    layer = Counter()
    for rec in D.values():
        for g in R.goals_of(rec):
            layer[classify(g)[1]] += 1
    tot = sum(layer.values())
    for lay in (STRUCT, LEXICON, FALLBACK):
        print(f"  {lay:<10} {layer[lay]:>4}  {layer[lay]/tot:>6.1%}")

    print("\n── 오분류 상위 (D) ────────────────────────────")
    miss, extra = Counter(), Counter()
    for k, rec in D.items():
        want = gi.get(k)
        if not want:
            continue
        got = predict(rec)
        for i in want - got:
            miss[i] += 1
        for i in got - want:
            extra[i] += 1
    print("  누락:", ", ".join(f"{i}×{c}" for i, c in miss.most_common(6)))
    print("  오출:", ", ".join(f"{i}×{c}" for i, c in extra.most_common(6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
