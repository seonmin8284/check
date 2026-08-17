"""
응답 품질 인적 평가용 층화 표본 생성.

    python build_review_sample.py --data ./data --out ./review --n 1000

무작위 표집은 쓰지 않는다. 주 관심(OTH 폴백·저관련성)이 비중만큼만 잡혀
판정이 불가능해지기 때문이다. 대조군을 반드시 포함시킨다 —
나쁜 것만 보면 "나쁘다"는 결론밖에 나오지 않는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mts_analysis import coverage as cov
from mts_analysis import relevance as REL
from mts_analysis import schema as sch
from mts_analysis import turns as T

# 층 이름 → (목표 비중, 설명)
STRATA = {
    "S1_폴백": (0.30, "OTH 분류 실패 → 폴백 함수 강제 호출 (주 관심)"),
    "S2_저관련": (0.20, "관련성 coverage 하위 20% (무관 응답 자동 스크리닝)"),
    "S3_행동실패": (0.20, "재질문·형식 재요청을 유발한 응답"),
    "S4_대조군": (0.20, "정상 성공 — 기준선 없이는 해석 불가"),
    "S5_의도균등": (0.10, "고비중 의도 균등 배분 (커버리지)"),
}

REVIEW_COLUMNS = [
    "review_id", "stratum", "query_id", "session_id", "ts",
    "질문", "응답본문",
    # ── 평가자 입력란 (아래만 채우면 됩니다) ──
    "R_관련성", "R_정확성", "R_완결성", "R_구조성", "R_위해성", "R_메모",
]

HIDDEN_COLUMNS = [   # 평가자에게 보이면 앵커링되므로 별도 파일로 분리
    "review_id", "l2_intent", "intent_pred", "tool_called", "outcome",
    "fail_code", "f4_compliance", "rel_coverage", "rel_jaccard", "rel_bigram",
    "rel_missed", "응답유형", "st_수치밀도", "st_해석문장수", "st_요약선행문",
    "turn_kind", "latency_ms",
]


def load(data_dir: Path) -> pd.DataFrame:
    for ext in (".pkl", ".parquet", ".csv"):
        p = data_dir / f"queries{ext}"
        if p.exists():
            break
    else:
        raise FileNotFoundError(f"queries 파일 없음: {data_dir}")
    df = (pd.read_pickle(p) if p.suffix == ".pkl"
          else pd.read_parquet(p) if p.suffix == ".parquet"
          else pd.read_csv(p, sep=None, engine="python"))
    return sch.ensure_optional(sch.normalize(df, "queries"))


def build(q: pd.DataFrame, n: int, seed: int = 0,
          oth_codes=("OTH", "기타", "ETC", "OTHERS"),
          fallback_tools=("get_news_and_work", "news_and_work")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = q.copy()

    # 층 판정에 필요한 파생
    d = cov.derive_fail_codes(d)
    d = REL.add_relevance(d)
    d = REL.add_structure(d)
    fu = T.classify_followups(d)
    d = d.merge(fu[["query_id", "turn_kind"]], on="query_id", how="left")
    # 응답이 유발한 후속(다음 턴)이 복구성인지
    nxt = fu.sort_values(["session_id", "ts"]).copy()
    nxt["_next_kind"] = nxt.groupby("session_id")["turn_kind"].shift(-1)
    d = d.merge(nxt[["query_id", "_next_kind"]], on="query_id", how="left")

    is_oth = (d["intent_pred"].astype(str).str.upper()
                .isin([c.upper() for c in oth_codes]))
    tc = d["tool_called"].fillna("").astype(str).str.lower()
    is_fb = tc.apply(lambda s: any(f in s for f in fallback_tools))
    thr = d["rel_coverage"].quantile(.20)
    low_rel = d["rel_coverage"] <= thr
    bad_next = d["_next_kind"].isin(["REPEAT", "FORMAT"]).fillna(False)
    normal_ok = (d["outcome"].eq("success") & ~is_oth & ~is_fb
                 & ~low_rel & ~bad_next)

    pools = {
        "S1_폴백": d[is_oth | is_fb],
        "S2_저관련": d[low_rel & ~(is_oth | is_fb)],
        "S3_행동실패": d[bad_next & ~(is_oth | is_fb) & ~low_rel],
        "S4_대조군": d[normal_ok],
    }

    picked, used = [], set()
    for name, (share, _) in STRATA.items():
        want = int(round(n * share))
        if name == "S5_의도균등":
            continue
        pool = pools[name]
        pool = pool[~pool["query_id"].isin(used)]
        take = min(want, len(pool))
        if take == 0:
            print(f"  ⚠ {name}: 대상 0건 — 건너뜀")
            continue
        idx = rng.choice(pool.index, size=take, replace=False)
        sel = pool.loc[idx].assign(stratum=name)
        picked.append(sel)
        used |= set(sel["query_id"])
        if take < want:
            print(f"  ⚠ {name}: {want}건 요청 / {take}건만 확보")

    # S5 — 남은 할당을 고비중 의도에 균등 배분
    rest = int(round(n * STRATA["S5_의도균등"][0]))
    remain = d[~d["query_id"].isin(used)]
    if rest > 0 and len(remain):
        top = remain["l2_intent"].value_counts().head(10).index
        per = max(rest // max(len(top), 1), 1)
        parts = []
        for it in top:
            sub = remain[remain["l2_intent"].eq(it)]
            k = min(per, len(sub))
            if k:
                parts.append(sub.loc[rng.choice(sub.index, k, replace=False)])
        if parts:
            picked.append(pd.concat(parts).assign(stratum="S5_의도균등"))

    out = pd.concat(picked, ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)  # 순서 섞기
    out.insert(0, "review_id", [f"R{i:04d}" for i in range(1, len(out) + 1)])
    out["질문"] = out["query_text"] if "query_text" in out else ""
    out["응답본문"] = out["ANSWER"].map(REL.strip_html) if "ANSWER" in out else ""
    if "answer_text" in out.columns:
        out["응답본문"] = out["answer_text"]
    for c in ("R_관련성", "R_정확성", "R_완결성", "R_구조성", "R_위해성", "R_메모"):
        out[c] = ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./review")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=50,
                    help="이중 평가용 파일럿 건수 (IAA 측정)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q = load(Path(args.data))
    print(f"전체 {len(q):,}건에서 {args.n:,}건 층화 표집")

    s = build(q, args.n, args.seed)
    print(f"\n확보 {len(s):,}건")
    print(s["stratum"].value_counts().to_string())

    sheet = s[[c for c in REVIEW_COLUMNS if c in s.columns]].drop(columns=["stratum"])
    sheet.to_csv(out / "review_sheet.csv", index=False, encoding="utf-8-sig")
    hidden = s[[c for c in HIDDEN_COLUMNS if c in s.columns]].assign(
        stratum=s["stratum"])
    hidden.to_csv(out / "review_keys.csv", index=False, encoding="utf-8-sig")

    # 파일럿: 이중 평가로 IAA 측정 (본 작업 전 가이드 검증)
    pilot = sheet.head(args.pilot)
    for who in ("A", "B"):
        pilot.to_csv(out / f"pilot_{who}.csv", index=False, encoding="utf-8-sig")

    print(f"\n→ {out/'review_sheet.csv'}   평가자 배포용 (라벨·프록시 숨김)")
    print(f"→ {out/'review_keys.csv'}     정답·프록시 (평가 후 조인용, 배포 금지)")
    print(f"→ {out/'pilot_A.csv'}, pilot_B.csv   파일럿 {args.pilot}건 이중 평가")
    print("\n순서: 파일럿 이중 평가 → IAA 확인 → 가이드 보정 → 본 평가")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
