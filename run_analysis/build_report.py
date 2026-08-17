"""
현업 보고서 생성기.

    python3 build_report.py --out ./out --report ./out/business_report.html

run_analysis.py 가 만든 CSV 들을 읽어, 통계 용어 없이 읽히는 보고서를 만든다.

설계 원칙
  1) 근거의 신뢰도를 먼저 밝힌다. 확정 / 방향만 / 측정불가 를 앞에서 선언한다.
  2) '측정의 문제'와 '제품의 문제'를 분리한다. 섞으면 "잘하고 있냐"에 답을 못 한다.
  3) 회수 추정은 별도 섹션이 아니라 우선순위 항목 안에 신뢰도와 함께 붙인다.
  4) 마지막은 의사결정 요청으로 닫는다.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mts_analysis.schema import label_ko

# ─────────────────────────────────────────────── 로딩

FILES = {
    "cohort": "cohort_retention.csv",
    "mixcf": "mix_counterfactual.csv",
    "entrymix": "entry_mix_over_time.csv",
    "entrymix_shift": "entry_mix_shift.csv",
    "entryret": "entry_retention.csv",
    "northstar": "north_star.csv",
    "mixadj": "mix_adjusted.csv",
    "priority": "priority.csv",
    "coverage": "coverage_matrix.csv",
    "recovery": "recovery_potential.csv",
    "ceiling": "recovery_ceiling.csv",
    "legacy": "legacy_bridge.csv",
    "uncovered": "legacy_uncovered.csv",
    "demand": "demand_supply_map.csv",
    "effort": "effort_by_intent.csv",
    "abandon": "intent_abandonment.csv",
    "segment": "segment_over_time.csv",
    "segcross": "segment_crosstab.csv",
    "exitpts": "exit_points.csv",
    "relbyintent": "relevance_by_intent.csv",
    "structure": "structure_impact.csv",
    "effsucc": "effective_success.csv",
    "oneshot": "oneshot_profile.csv",
    "transition": "transition_by_outcome.csv",
    "sessprof": "session_profile.csv",
    "sesstop": "session_top_intent.csv",
    "latency": "latency_impact.csv",
    "retdrv": "retention_drivers.csv",
    "activation": "activation_candidates.csv",
    "effsucc_i": "effective_success.csv",
    "gaps": "data_gaps.csv",
    "newret": "new_vs_returning.csv",
    "halluc": "hallucination_risk.csv",
    "usage": "usage_cycle.csv",
    "km": "km_curve.csv",
    "slotint": "slot_by_intent.csv",
    "attr_comb": "attribution_combined.csv",
    "shift": "shift_share.csv",
    "need": "underlying_need.csv",
    "need_q": "need_quadrant.csv",
    "need_ctx": "context_conditional.csv",
    "attr": "attribution_session.csv",
    "attr_fix": "attribution_singlefix.csv",
    "attr_exp": "attribution_exposure.csv",
    "attr_cnt": "attribution_count.csv",
    "lat_path": "latency_by_path.csv",
    "lat_effect": "latency_effect.csv",
    "lat_session": "latency_by_session.csv",
    "bands": "uncertainty_bands.csv",
}


def load_findings(out_dir: Path) -> dict:
    p = out_dir / "findings.json"
    if not p.exists():
        return {}
    import json
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load(out_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for k, fn in FILES.items():
        p = out_dir / fn
        if p.exists():
            try:
                data[k] = pd.read_csv(p)
            except Exception:
                pass
    return data


# 코드가 들어갈 수 있는 컬럼 — 자연어 라벨로 치환한다
_CODE_COLS = ("l2_intent", "l1_stage", "의도", "진입질문", "유형", "first_intent",
              "_gold", "from", "to", "응답유형", "단계", "stage", "주의도",
              "주로 쓰는 기능", "l1_stage", "sess", "상위 의도")

_CODE_RE = re.compile(r"[A-Z]{3,5}\.[a-z_]+")


def _sub_codes(v) -> str:
    """'EVAL.price(6%), SVC.auth(5%)' 처럼 코드가 문장 안에 섞인 경우도 치환."""
    return _CODE_RE.sub(lambda m: label_ko(m.group(0)), str(v))

# 실패 원인 코드도 외부 문서에는 자연어로 쓴다
FAIL_LABEL_KO = {
    "D1": "참고할 자료 자체가 없음",
    "D2": "자료 공급이 끊김",
    "D3": "자료는 있으나 이 질문은 담기지 않음",
    "T1": "처리할 기능이 없음",
    "T2": "기능은 있는데 호출되지 않음 (분류 실패)",
    "S1": "질문만으로 조회 조건을 못 정함",
    "X1": "앞 질문 맥락을 못 이어감",
    "A1": "본인 인증이 필요한데 처리되지 않음",
    "C1": "규정상 답할 수 없어 차단 (정상)",
    "C2": "차단하지 않아도 될 것을 차단",
    "C3": "차단한 뒤 대체 안내를 주지 못함",
    "M1": "위 어디에도 해당하지 않는 응답 오류",
}


_SEG_NUM = "①②③④⑤⑥⑦⑧"


def seg_label(v) -> str:
    """'S2: EVALUATE+DISCOVER' → '② 종목 살펴보기·찾기형' 으로."""
    t = str(v).strip()
    if not t.upper().startswith("S") or ":" not in t:
        return label_ko(t)
    head, _, tail = t.partition(":")
    try:
        n = _SEG_NUM[int(head[1:])]
    except Exception:
        n = head
    parts = [label_ko(x.strip()) for x in tail.split("+") if x.strip()]
    return f"{n} " + "·".join(parts) + "형"


def humanize(df: pd.DataFrame) -> pd.DataFrame:
    """표에 남은 내부 코드를 자연어로 바꾼다. 외부 보고서에 코드를 노출하지 않는다."""
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    # 세그먼트 라벨(S0: STAGE+STAGE)은 별도 규칙으로 변환
    for c in d.columns:
        if c in ("사용자 부류", "seg", "세그먼트") or (
                c == d.columns[0]
                and d[c].astype(str).str.match(r"^S\d+\s*:").any()):
            d[c] = d[c].map(seg_label)
    d = d.rename(columns={c: seg_label(c) for c in d.columns
                          if str(c).upper().startswith("S") and ":" in str(c)})
    _STAGES = ("LEARN", "DISCOVER", "EVALUATE", "EXECUTE",
               "MONITOR", "SETTLE", "SERVICE", "RECOVER")
    d = d.rename(columns={c: label_ko(c) for c in d.columns if c in _STAGES})
    for c in d.columns:
        if c in ("fail_code", "실패코드", "원인 코드"):
            d[c] = d[c].map(lambda v: FAIL_LABEL_KO.get(str(v).strip(), v))
            continue
        if c in _CODE_COLS or c == d.columns[0]:
            try:
                if d[c].astype(str).str.contains(_CODE_RE, regex=True).any():
                    d[c] = d[c].map(_sub_codes)
                elif c in ("l1_stage", "유형") or d[c].astype(str).isin(
                        ["LEARN", "DISCOVER", "EVALUATE", "EXECUTE",
                         "MONITOR", "SETTLE", "SERVICE", "RECOVER"]).any():
                    d[c] = d[c].map(label_ko)
            except Exception:
                pass
    rename = {"l2_intent": "질문 유형", "l1_stage": "질문 갈래",
              "first_intent": "첫 질문 유형", "_gold": "질문 유형",
              "n": "건수", "질의량": "질문 수", "사용자수": "사용자 수",
              "sess": "시간대", "period": "시기", "신규": "신규 사용자",
              "실패율": "실패 비율", "성공률": "성공 비율",
              "잔존율": "30일 후 재방문율", "고비용비율": "여러 번 물은 비율",
              "중앙턴": "평균 문답 횟수", "질의량추세": "질문 수 추세",
              "fail_code": "무엇이 문제인가", "실패코드": "무엇이 문제인가",
              "단계": "질문 갈래", "설명": "문제 내용",
              "비중": "전체 대비 비중", "처방": "필요한 조치",
              "주의도": "주로 쓰는 기능", "사용자": "사용자 수",
              "질의": "질문 수", "판단성질의비중": "판단 요구 질문 비중",
              "의도다양성": "쓰는 기능 가짓수",
              "실패질의량": "해당 질문 수", "회수비중": "개선 시 회수 몫"}
    d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
    return d


def pct(x, d=1):
    try:
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return "—"


def num(x, d=0):
    try:
        return f"{float(x):,.{d}f}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────── 분석 조각

def f_scale(D) -> dict | None:
    """규모: 재방문율 하락 폭과 시점."""
    if "cohort" not in D:
        return None
    c = D["cohort"].copy()
    key = c.columns[0]
    col = "D30" if "D30" in c.columns else ("D7" if "D7" in c.columns else None)
    if col is None:
        return None
    c = c[c[col].notna()]
    if len(c) < 6:
        return None
    head = c.head(4)[col].mean()
    tail = c.tail(4)[col].mean()
    peak = c.loc[c[col].idxmax()]
    # 하락이 시작된 시점: 최고치 이후 3개 구간 연속 하락하는 첫 지점
    v = c[col].to_numpy()
    turn = None
    for i in range(int(c[col].idxmax() - c.index[0]) + 1, len(v) - 3):
        if v[i] > v[i+1] > v[i+2] > v[i+3]:
            turn = c[key].iloc[i]
            break
    return {"초기": head, "최근": tail, "배수": head / tail if tail else np.nan,
            "최고시기": str(peak[key]), "최고값": float(peak[col]),
            "꺾인시기": str(turn) if turn is not None else None,
            "지표": col, "구간수": len(c)}


def f_quality_stable(D) -> dict | None:
    """같은 기간 응답 품질(세션 해결률)이 흔들렸는가."""
    if "northstar" not in D:
        return None
    n = D["northstar"]
    col = ("무마찰해결률" if "무마찰해결률" in n.columns
           else "세션해결률" if "세션해결률" in n.columns
           else "세션성공률" if "세션성공률" in n.columns else None)
    if col is None or len(n) < 4:
        return None
    v = n[col].astype(float)
    fb = (float(n["폴백세션비율"].mean()) if "폴백세션비율" in n.columns
          else np.nan)
    return {"지표명": col, "평균": float(v.mean()), "최소": float(v.min()),
            "최대": float(v.max()), "변동폭": float(v.max() - v.min()),
            "추세": float(v.iloc[-3:].mean() - v.iloc[:3].mean()),
            "폴백세션비율": fb}


def f_mix(D) -> dict | None:
    """유입 구성 변화가 하락을 얼마나 설명하는가."""
    if "mixcf" not in D:
        return None
    m = D["mixcf"]
    if "관측잔존" not in m.columns or len(m) < 3:
        return None
    obs_drop = float(m["관측잔존"].iloc[0] - m["관측잔존"].iloc[-1])
    loss = float(m["믹스손실"].tail(3).mean()) if "믹스손실" in m.columns else np.nan
    share = abs(loss) / obs_drop if obs_drop > 0 else np.nan
    return {"관측하락": obs_drop, "믹스손실": loss, "설명력": share}


def f_entry_shift(D) -> pd.DataFrame | None:
    if "entrymix_shift" not in D:
        return None
    s = D["entrymix_shift"].copy()
    c0 = s.columns[0]
    s = s.rename(columns={c0: "유형"})
    return s.sort_values("변화")


def f_fallback(D) -> dict | None:
    """분류 실패 규모 — legacy_bridge 의 OTHER 행에서 추정."""
    if "legacy" not in D:
        return None
    lg = D["legacy"].copy()
    c0 = lg.columns[0]
    row = lg[lg[c0].astype(str).str.upper().str.contains("OTH")]
    if row.empty:
        return None
    n_oth = float(row["n"].iloc[0])
    total = float(lg["n"].sum())
    return {"건수": n_oth, "전체": total, "비중": n_oth / total,
            "분해도": float(row["분해도"].iloc[0]) if "분해도" in row else np.nan,
            "의도수": float(row["신규의도수"].iloc[0])
            if "신규의도수" in row else np.nan}


def f_uncovered(D) -> dict | None:
    if "uncovered" not in D:
        return None
    u = D["uncovered"].copy()
    c0 = u.columns[0]
    u = u.rename(columns={c0: "의도"})
    return {"합계": float(u["비중"].sum()) if "비중" in u.columns else np.nan,
            "표": u.head(8)}


def f_entry_ret(D) -> dict | None:
    if "entryret" not in D:
        return None
    e = D["entryret"].copy()
    if "유의" in e.columns:
        e["유의"] = e["유의"].fillna("")
        sig = e[e["유의"].astype(str).str.contains("★")]
    else:
        sig = e
    if sig.empty:
        return None
    c0 = sig.columns[0]
    sig = sig.rename(columns={c0: "진입질문"})
    top = sig.nlargest(5, "잔존율")[["진입질문", "n", "잔존율"]]
    bot = sig.nsmallest(5, "잔존율")[["진입질문", "n", "잔존율"]]
    return {"상위": top, "하위": bot,
            "격차": float(top["잔존율"].mean() / bot["잔존율"].mean())
            if bot["잔존율"].mean() else np.nan}


def f_priority(D) -> pd.DataFrame | None:
    """개선 우선순위 — 지원 범위 밖(OOS) 항목은 제외."""
    if "priority" not in D:
        return None
    p = D["priority"].copy()
    if "l2_intent" not in p.columns:
        return None
    p = p[~p["l2_intent"].astype(str).str.startswith("OOS.")]
    keep = [c for c in ["l2_intent", "단계", "fail_code", "질의량", "실패율",
                        "담당", "처방"] if c in p.columns]
    return p[keep].head(8)


def f_c3(D) -> dict | None:
    if "coverage" not in D or "C3" not in D["coverage"].columns:
        return None
    cv = D["coverage"]
    c3 = float(cv["C3"].sum())
    c1 = float(cv["C1"].sum()) if "C1" in cv.columns else 0.0
    tot = c3 + c1
    return {"회수실패": c3, "차단전체": tot,
            "실패율": c3 / tot if tot else np.nan}


def f_effort(D) -> pd.DataFrame | None:
    if "effort" not in D:
        return None
    e = D["effort"].copy()
    c0 = e.columns[0]
    e = e.rename(columns={c0: "진입질문"})
    keep = [c for c in ["진입질문", "n", "중앙턴", "고비용비율"] if c in e.columns]
    return e.nlargest(6, "고비용비율")[keep] if "고비용비율" in e.columns else None


def f_demand(D) -> pd.DataFrame | None:
    """수요 억눌림 — 정책적 차단 대상과 지원 범위 밖은 제외."""
    if "demand" not in D:
        return None
    d = D["demand"].copy()
    c0 = d.columns[0]
    d = d.rename(columns={c0: "의도"})
    if "사분면" not in d.columns:
        return None
    s = d[d["사분면"].astype(str).str.contains("억눌림")]
    drop = ("EVAL.verdict", "DISC.recommend_open", "MON.rebalance",
            "MON.loss_reaction")
    s = s[~s["의도"].isin(drop)]
    s = s[~s["의도"].astype(str).str.startswith(("REC.", "OOS.", "RISK."))]
    keep = [c for c in ["의도", "질의량", "사용자수", "성공률", "질의량추세"]
            if c in s.columns]
    return s[keep].head(6)


def f_protector(D) -> dict | None:
    """차단 관련 — 인과가 아니라 운영 사실만."""
    if "coverage" not in D:
        return None
    cv = D["coverage"]
    c3 = float(cv["C3"].sum()) if "C3" in cv.columns else 0.0
    c1 = float(cv["C1"].sum()) if "C1" in cv.columns else 0.0
    tot = c3 + c1
    ex = None
    if "exitpts" in D:
        e = D["exitpts"].copy()
        c0 = e.columns[0]
        row = e[e[c0].astype(str).str.strip() == "C3"]
        if not row.empty and "lift" in row.columns:
            ex = float(row["lift"].iloc[0])
    return {"차단": tot, "회수실패": c3,
            "회수실패율": c3 / tot if tot else np.nan, "종료위험배수": ex}


def f_abandon(D) -> pd.DataFrame | None:
    if "abandon" not in D:
        return None
    a = D["abandon"].copy()
    c0 = a.columns[0]
    a = a.rename(columns={c0: "의도"})
    keep = [c for c in ["의도", "실패후_재시도율", "성공후_재시도율", "포기효과"]
            if c in a.columns]
    if "포기효과" not in a.columns:
        return None
    a = a[~a["의도"].astype(str).str.startswith(("OOS.", "REC.", "RISK."))]
    return a.nlargest(6, "포기효과")[keep]


def f_structure(D) -> pd.DataFrame | None:
    if "structure" not in D:
        return None
    st = D["structure"].copy()
    c0 = st.columns[0]
    st = st.rename(columns={c0: "응답 형태"})
    keep = [c for c in ["응답 형태", "n", "복구성후속률", "실질성공률",
                        "관련성_coverage"] if c in st.columns]
    return st[keep]


def f_relevance(D) -> pd.DataFrame | None:
    if "relbyintent" not in D:
        return None
    r = D["relbyintent"].copy()
    c0 = r.columns[0]
    r = r.rename(columns={c0: "의도"})
    if "coverage" not in r.columns:
        return None
    r = r[~r["의도"].astype(str).str.startswith(("OOS.", "REC.", "RISK."))]
    return r.nsmallest(6, "coverage")[["의도", "n", "coverage"]]


def f_ceiling(D) -> pd.DataFrame | None:
    if "ceiling" not in D:
        return None
    c = D["ceiling"].copy()
    keep = [x for x in ["시나리오", "세션성공률", "무마찰세션률",
                        "Δ무마찰세션률"] if x in c.columns]
    return c[keep]


CSS_FORMAL = """
:root{--fg:#111;--mut:#555;--line:#c9c9c9;--soft:#f2f2f2}
*{box-sizing:border-box}
body{margin:0;color:var(--fg);background:#fff;font-size:14px;line-height:1.72;
     font-family:"Malgun Gothic","맑은 고딕","Apple SD Gothic Neo",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:34px 30px 90px}
.cover{border:1px solid var(--line);padding:26px 28px;margin-bottom:30px}
.cover h1{font-size:23px;margin:0 0 18px;letter-spacing:-.5px;font-weight:700}
.cover table{width:100%;border:0;margin:0;font-size:13px}
.cover td{border:0;padding:3px 0;vertical-align:top}
.cover td:first-child{width:96px;color:var(--mut)}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;
   border-bottom:1.5px solid var(--fg);font-weight:700}
h3{font-size:14.5px;margin:22px 0 6px;font-weight:700}
h4{font-size:13.5px;margin:16px 0 4px;color:#333;font-weight:700}
p{margin:8px 0}
.lead{color:var(--mut);font-size:13px;margin:0 0 12px}
ul,ol{margin:8px 0;padding-left:20px}
li{margin:3px 0}
table{border-collapse:collapse;width:100%;margin:10px 0 16px;font-size:13px}
th,td{padding:6px 9px;text-align:left;border:0;border-bottom:1px solid #e6e6e6}
thead th,tr:first-child th{border-top:1.5px solid var(--fg);
   border-bottom:1px solid var(--fg);background:transparent;font-weight:700}
table tr:last-child td{border-bottom:1.5px solid var(--fg)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.note{border:1px solid var(--line);background:var(--soft);padding:9px 13px;
      margin:10px 0;font-size:13px}
.note b{font-weight:700}
.fn{font-size:12px;color:var(--mut);margin:14px 0 0;padding-top:8px;
    border-top:1px solid #e6e6e6}
.fn div{margin:2px 0}
sup{font-size:10px;vertical-align:super;color:var(--mut)}
.toc{border:1px solid var(--line);padding:14px 20px;margin-bottom:26px;font-size:13px}
.toc b{display:block;margin-bottom:6px}
.toc ol{margin:0;padding-left:18px}
.toc a{color:#111;text-decoration:none}
@media print{.wrap{max-width:none;padding:0}body{font-size:11px}
  h2{page-break-after:avoid}table{page-break-inside:avoid}
  .cover{page-break-after:always}.toc{page-break-after:always}}
"""


# ─────────────────────────────────────────────── 렌더링

CSS = """
:root{--fg:#1b1b1b;--mut:#6b6b6b;--line:#e3e3e3;--accent:#0b57d0;
      --warn:#b45309;--warnbg:#fffbeb;--ok:#166534;--okbg:#f0fdf4;
      --bad:#b91c1c;--badbg:#fef2f2}
*{box-sizing:border-box}
body{margin:0;color:var(--fg);background:#fff;line-height:1.7;
     font-family:"Malgun Gothic","맑은 고딕","Apple SD Gothic Neo",sans-serif;font-size:15px}
.wrap{max-width:940px;margin:0 auto;padding:40px 28px 90px}
h1{font-size:27px;margin:0 0 4px;letter-spacing:-.4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:28px}
h2{font-size:21px;margin:44px 0 6px;padding-top:16px;border-top:2px solid var(--fg)}
.lead{color:var(--mut);font-size:14px;margin:0 0 16px}
h3{font-size:16.5px;margin:26px 0 8px}
p{margin:10px 0}
table{border-collapse:collapse;width:100%;margin:12px 0 18px;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#fafafa;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.box{border-left:4px solid var(--accent);background:#f7f9ff;padding:12px 16px;
     margin:14px 0;border-radius:0 5px 5px 0}
.box.warn{border-color:var(--warn);background:var(--warnbg)}
.box.bad{border-color:var(--bad);background:var(--badbg)}
.box.ok{border-color:var(--ok);background:var(--okbg)}
.box b{display:block;margin-bottom:4px}
.big{font-size:30px;font-weight:700;letter-spacing:-1px}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}
.kpi div{flex:1;min-width:150px;border:1px solid var(--line);border-radius:8px;
         padding:14px 16px}
.kpi span{display:block;color:var(--mut);font-size:12.5px;margin-bottom:4px}
.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:10px;
     border:1px solid var(--line);color:var(--mut);margin-left:6px;vertical-align:2px}
.tag.a{background:var(--okbg);border-color:#bbf7d0;color:var(--ok)}
.tag.b{background:var(--warnbg);border-color:#fde68a;color:var(--warn)}
.tag.c{background:#f4f4f5;color:var(--mut)}
ol,ul{margin:10px 0;padding-left:22px}
li{margin:5px 0}
.part{margin:52px 0 -8px;padding:10px 18px;background:#1b1b1b;color:#fff;
      border-radius:6px;font-size:15px;font-weight:700;letter-spacing:-.2px}
.hyp{border:1px solid var(--line);border-left:5px solid var(--mut);
     border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0}
.hyp.bad{border-left-color:var(--bad);background:#fffafa}
.hyp.ok{border-left-color:var(--ok);background:#fafffb}
.hyp.warn{border-left-color:var(--warn);background:#fffdf7}
.hyp b{display:block;font-size:15.5px;margin-bottom:8px}
h4{font-size:14.5px;margin:20px 0 6px;color:#333}
.hyp p{margin:5px 0;font-size:14px}
.hyp i{display:inline-block;min-width:76px;color:var(--mut);font-style:normal;
       font-size:12.5px;font-weight:600}
@media print{.wrap{max-width:none;padding:0}body{font-size:11.5px}
  h2{page-break-before:auto;page-break-after:avoid}table{page-break-inside:avoid}
  .part{background:#fff;color:#000;border-top:3px solid #000;border-radius:0;
        page-break-before:always}
  .hyp{page-break-inside:avoid}}
"""


def T(df: pd.DataFrame, cols_fmt: dict | None = None) -> str:
    if df is None or len(df) == 0:
        return "<p>해당 데이터가 없습니다.</p>"
    df = humanize(df).copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "string":
            df[c] = df[c].astype(object).where(df[c].notna(), "")
            df[c] = df[c].replace({"nan": "", "None": "", "<NA>": "", np.nan: ""})
    fmt = {}
    for k, v in (cols_fmt or {}).items():
        fmt[k] = v
        for a, b in [("질의량", "질문 수"), ("실패율", "실패 비율"),
                     ("성공률", "성공 비율"), ("잔존율", "30일 후 재방문율"),
                     ("고비용비율", "여러 번 물은 비율"), ("n", "건수"),
                     ("비중", "전체 대비 비중")]:
            if k == a:
                fmt[b] = v
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                txt = fmt[c](v)
                tds.append(f'<td class="num">{_html.escape(txt)}</td>')
            elif isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v):
                fv = float(v)
                if float(fv).is_integer() and abs(fv) >= 1:
                    txt = f"{fv:,.0f}"
                elif abs(fv) < 1:
                    txt = f"{fv:.3f}"
                else:
                    txt = f"{fv:,.1f}"
                tds.append(f'<td class="num">{txt}</td>')
            else:
                txt = "" if (v is None or (isinstance(v, float) and pd.isna(v))
                             or str(v) in ("nan", "None", "<NA>", "NaT")) else str(v)
                tds.append(f"<td>{_html.escape(txt)}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def build(D: dict, meta: dict, F: dict | None = None) -> str:
    F = F or {}
    P: list[str] = []
    A = lambda s: P.append(s)
    H2 = lambda n, t, lead="": A(f'<h2>{n}. {t}</h2>'
                                + (f'<p class="lead">{lead}</p>' if lead else ""))

    sc = f_scale(D); qs = f_quality_stable(D); mx = f_mix(D)
    fb = f_fallback(D); uc = f_uncovered(D); er = f_entry_ret(D)
    pr = f_priority(D); ef = f_effort(D); dm = f_demand(D); es = f_entry_shift(D)
    pt = f_protector(D); ab = f_abandon(D); stt = f_structure(D)
    rv = f_relevance(D); cl = f_ceiling(D)

    A("<h1>챗봇 서비스 진단 보고</h1>")
    A(f'<div class="sub">작성 {meta["now"]} · 분석 대상 기간 {meta["period"]}</div>')

    # ════════════════════ 1부
    A('<div class="part">1부 · 무슨 일이 일어나고 있나</div>')

    # ── 1. 한 장 요약
    H2(1, "한 장 요약")
    k = []
    ns = F.get("세션", {})
    if ns.get("세션 성공률") is not None:
        k.append(f'<div><span>한 번에 목적을 이룬 세션</span>'
                 f'<b class="big">{pct(F.get("세션결과분포",{}).get("RESOLVED"))}</b></div>')
    if sc:
        k.append(f'<div><span>가입 30일 후 재방문율 (최근)</span>'
                 f'<b class="big">{pct(sc["최근"])}</b></div>')
    if F.get("폴백", {}).get("OTH 비중") is not None:
        k.append(f'<div><span>질문 유형 인식 실패</span>'
                 f'<b class="big">{pct(F["폴백"]["OTH 비중"])}</b></div>')
    if k:
        A('<div class="kpi">' + "".join(k) + "</div>")
    A("<h3>이 보고서가 답하는 다섯 가지</h3>")
    rows = [
        ("사용자는 왜 떠나는가",
         "떠나기 전에 <b>기능 하나를 먼저 포기</b>합니다. 그 신호가 서비스 이탈보다 먼저 나타납니다. (5·6장)"),
        ("무엇이 원인인가",
         "가장 큰 원인은 <b>질문 유형을 잘못 알아듣는 것</b>이고, 그 다음이 "
         "<b>규정상 거절 후 대안을 주지 않는 것</b>입니다. (2·5장)"),
        ("남는 사용자는 왜 남는가",
         "첫 주에 <b>서로 다른 기능을 여러 개 써본 사용자</b>가 남습니다. (8장)"),
        ("누구에게 초점을 맞출 것인가",
         "규모는 줄었지만 잔존이 높은 <b>시장·종목 탐색형</b>입니다. (9장)"),
        ("얼마나 좋아지는가",
         "확정 효과와 추정 효과를 나눠 10장에 정리했습니다."),
    ]
    A(T(pd.DataFrame(rows, columns=["질문", "답"])))
    A('<div class="box warn"><b>예상과 달랐던 것</b>'
      "현업에서 원인으로 자주 지목되던 <b>규정 차단</b>과 <b>응답 속도</b>는, "
      "확인해 보니 이탈과 직접 연결되지 않았습니다. 자세한 내용은 7장에 있습니다.</div>")

    if "bands" in D and len(D["bands"]):
        A("<h3>지금 모르는 폭</h3>")
        A("<p>일부 숫자는 하나의 값이 아니라 구간으로만 말할 수 있습니다. "
          "그 폭이 곧 <b>아직 확보하지 못한 데이터의 크기</b>입니다. "
          "무엇을 확보하면 좁아지는지도 함께 적었습니다.</p>")
        bd = D["bands"].copy()
        _isrt = bd["지표"].astype(str).str.contains("위험비|배수")
        for c in ("하한", "상한", "폭"):
            bd[c] = [f"{float(v):.2f}배" if r else pct(v)
                     for v, r in zip(bd[c], _isrt)]
        A(T(bd))
    mj = (F or {}).get("_오판사례")
    if mj:
        A("<h3>이번 분석에서 잘못 볼 뻔했던 것</h3>")
        A("<p>데이터가 없거나 부정확해 처음에 다른 결론이 날 뻔했던 사례입니다. "
          "같은 이유로 다음에도 틀릴 수 있습니다.</p>")
        A(T(pd.DataFrame(mj)))

    # ── 2. 숫자를 믿기 전에
    H2(2, "숫자를 믿기 전에",
       "제품 이야기를 하기 전에, 우리가 보고 있는 숫자 자체를 먼저 교정해야 합니다.")
    A("<p>챗봇은 질문을 먼저 유형으로 나눈 뒤 알맞은 기능을 부릅니다. "
      "그런데 유형 판별에 실패하면 <b>전부 '기타'로 넘기고, 기타는 무조건 "
      "뉴스·업무 검색을 호출</b>합니다. 이 검색은 무엇이든 답을 만들어 내보내므로, "
      "<b>질문과 상관없는 답이 나가도 시스템은 '성공'으로 기록</b>합니다.</p>")
    ofb = F.get("폴백", {})
    if ofb:
        A(f'<div class="box bad"><b>인식 실패가 전체의 {pct(ofb.get("OTH 비중"))}</b>'
          f'이 응답들은 지금 성공률 안에 들어 있습니다. 폴백을 모두 실패로 본다면 '
          f'성공률은 {pct(ofb.get("보고 성공률"))}에서 '
          f'{pct(ofb.get("폴백 제외 성공률(하한)"))}까지 내려갑니다. '
          f'실제 값은 이 사이 어딘가이며, 표본 검수로 좁힐 예정입니다.</div>')
        if ofb.get("coverage 격차") is not None:
            A(f"<p>질문의 핵심어가 답변에 얼마나 담기는지 보면, 인식 실패 건은 "
              f"{ofb.get('관련성 coverage (폴백)'):.2f}로 정상 건 "
              f"{ofb.get('관련성 coverage (정상)'):.2f}보다 낮습니다. 다만 격차가 "
              f"크지 않아, 이 지표만으로 오답을 가려내기는 어렵습니다."
              f'<span class="tag b">방향</span></p>')
    if uc and uc["합계"] == uc["합계"]:
        A(f"<p>분류 체계 자체에도 공백이 있습니다. 기존 분류로는 담기지 않던 "
          f'질문이 전체의 <b>{pct(uc["합계"])}</b>입니다.<span class="tag a">확정</span></p>')
        A(T(uc["표"], {"비중": lambda v: pct(v, 2)}))
    om = (F or {}).get("운영분류", {})
    if om.get("정합률") is not None:
        A("<h3>기존 분류 체계의 해상도</h3>")
        A(f"<p>운영 중인 분류를 재정리한 기준과 대조하면 정합률이 "
          f"<b>{pct(om['정합률'])}</b>입니다. 한 카테고리가 여러 성격의 질문을 "
          f'묶고 있다는 뜻입니다.<span class="tag a">확정</span></p>')
        if "legacy" in D and len(D["legacy"]):
            lg = D["legacy"].copy()
            c0 = lg.columns[0]
            lg = lg.rename(columns={c0: "기존 분류", "n": "건수",
                                    "신규의도수": "실제 포함된 질문 종류",
                                    "분해도": "흩어진 정도"})
            keep = [c for c in ["기존 분류", "건수", "정합률",
                                "실제 포함된 질문 종류", "흩어진 정도"]
                    if c in lg.columns]
            A(T(lg[keep].head(8), {"정합률": lambda v: pct(v)}))
            A('<div class="box"><b>읽는 법</b>'
              "흩어진 정도가 1에 가까울수록 그 카테고리 하나가 여러 종류의 "
              "질문을 담고 있다는 뜻입니다. 라우팅이 정확할 수 없는 구조입니다.</div>")

    if "relbyintent" in D and len(D["relbyintent"]):
        rvv = D["relbyintent"].copy()
        c0 = rvv.columns[0]
        rvv = rvv.rename(columns={c0: "질문 유형", "coverage": "핵심어 반영률",
                                  "n": "건수"})
        A("<h3>질문의 핵심어가 답변에 담기는가</h3>")
        A("<p>사용자가 말한 종목명·지표명이 답변 안에 실제로 등장하는 비율입니다. "
          '낮을수록 동문서답에 가깝습니다.<span class="tag b">방향</span></p>')
        A(T(rvv.head(6), {"핵심어 반영률": lambda v: f"{float(v):.2f}"}))
        if ofb.get("coverage 격차") is not None:
            A(f'<div class="box warn"><b>이 지표만으로는 부족합니다</b>'
              f"인식 실패 건과 정상 건의 반영률 격차가 "
              f"{ofb['coverage 격차']:.2f}에 그칩니다. 무관한 답을 자동으로 "
              f"가려내기에는 판별력이 약하며, <b>표본 검수가 필요한 이유</b>입니다.</div>")

    qdf = (F or {}).get("질의왜곡", {})
    if qdf.get("전체 보존율(중앙)") is not None:
        A("<h3>질문이 검색에 전달되는 과정에서 얼마나 보존되는가</h3>")
        A("<p>챗봇은 사용자의 질문을 그대로 쓰지 않고 검색용으로 다시 씁니다. "
          "이 과정에서 핵심어가 빠지면, 이후 답이 어긋나는 것은 예정된 "
          "결과입니다.</p>")
        rows = [("질문이 검색어에 보존된 비율(중앙값)",
                 f"{qdf['전체 보존율(중앙)']:.2f}"),
                ("보존율 50% 미만", pct(qdf.get("보존율 0.5 미만 비중")))]
        if qdf.get("질문→쿼리 손실(검색 단계)") is not None:
            rows += [("검색 단계에서 잃은 몫",
                      pct(qdf["질문→쿼리 손실(검색 단계)"])),
                     ("답변 생성 단계에서 잃은 몫",
                      pct(qdf.get("쿼리→응답 손실(생성 단계)")))]
        A(T(pd.DataFrame(rows, columns=["항목", "값"])))
        _s1 = qdf.get("질문→쿼리 손실(검색 단계)")
        _s2 = qdf.get("쿼리→응답 손실(생성 단계)")
        if _s1 is not None and _s2 is not None:
            A('<div class="box ok"><b>어디를 고쳐야 하는가</b>'
              + ("검색 단계 손실이 더 큽니다. 모델을 바꿔도 해결되지 않으며 "
                 "<b>검색어 재작성 로직</b>을 손봐야 합니다."
                 if _s1 > _s2 else
                 "생성 단계 손실이 더 큽니다. 검색은 제대로 되는데 답변을 "
                 "만드는 과정에서 어긋나고 있습니다.") + "</div>")

    hzf = (F or {}).get("환각위험", {})
    if hzf.get("전체위험군비율") is not None:
        A("<h3>근거 없이 만들어진 답변</h3>")
        A(f"<p>자료 조회 없이 곧바로 생성된 응답이 <b>"
          f"{pct(hzf['전체위험군비율'])}</b>입니다. 사실 확인이 필요한 질문에서는 "
          f'틀린 내용이 그럴듯하게 나갈 수 있습니다.<span class="tag b">방향</span></p>')
        if "halluc" in D and len(D["halluc"]):
            hcd = D["halluc"].copy()
            c0 = hcd.columns[0]
            hcd = hcd.rename(columns={c0: "질문 유형", "n": "건수",
                                      "위험군비율": "근거 없는 응답 비율",
                                      "무툴응답": "자료 조회 없이 답한 비율"})
            A(T(hcd.head(6), {"근거 없는 응답 비율": lambda v: pct(v),
                              "자료 조회 없이 답한 비율": lambda v: pct(v)}))

    if "slotint" in D and len(D["slotint"]):
        sl = D["slotint"].copy()
        c0 = sl.columns[0]
        sl = sl.rename(columns={c0: "질문 유형", "n": "건수",
                                "target복원율": "질문 대상 인식률"})
        keep = [c for c in ["질문 유형", "건수", "질문 대상 인식률"]
                if c in sl.columns]
        A("<h3>질문의 대상을 제대로 잡고 있는가</h3>")
        A("<p>어떤 종목·어떤 기간을 묻는지 시스템이 인식한 비율입니다. "
          "낮으면 답변 정확도뿐 아니라 <b>되묻기 여부를 판정하는 정확도까지</b> "
          '함께 떨어집니다.<span class="tag a">확정</span></p>')
        A(T(sl[keep].head(6), {"질문 대상 인식률": lambda v: pct(v)}))

    bdf = (F or {}).get("차단정의", {})
    if bdf.get("kappa") is not None:
        A("<h3>차단 판정의 신뢰도</h3>")
        A(f"<p>차단 여부를 두 가지 방법으로 각각 집계해 대조했습니다 — "
          f"로그에 남은 표기와 응답 구조로 추정한 값입니다. 일치도는 "
          f"<b>{bdf['kappa']:.2f}</b>입니다."
          + (" 두 방법이 잘 맞으므로 이후 차단 관련 수치를 신뢰할 수 있습니다."
             if bdf["kappa"] >= .8 else
             " 일치도가 낮아 차단 관련 수치는 방향성 참고로만 사용해야 합니다.")
          + '<span class="tag a">확정</span></p>')

    A('<div class="box"><b>이것이 뜻하는 것</b>'
      "성능을 논하기 전에 <b>무엇이 실패인지부터 다시 정의</b>해야 합니다. "
      "지금은 실패의 상당 부분이 성공으로 집계되어, 개선 효과를 잴 기준선이 "
      "없습니다.</div>")

    # ── 3. 우리 사용자는 누구인가
    H2(3, "우리 사용자는 누구인가",
       "이후 모든 장이 이 구성 위에서 읽힙니다.")
    if "newret" in D and len(D["newret"]):
        A("<h3>매달 누가 서비스를 채우고 있는가</h3>")
        nr = D["newret"].copy()
        A(T(nr.tail(10), {"신규 비중": lambda v: pct(v),
                          "재방문 비중": lambda v: pct(v)}))
        uc2 = (F or {}).get("사용자구성", {})
        if uc2.get("판정"):
            A(f'<div class="box warn"><b>이것이 이후 논의의 전제입니다</b>'
              f'{_html.escape(uc2["판정"])}</div>')
    A("<h3>어떤 부류로 나뉘는가</h3>")
    if "segcross" in D:
        sx = D["segcross"].copy()
        c0 = sx.columns[0]
        sx = sx.rename(columns={c0: "사용자 부류"})
        keep = [c for c in ["사용자 부류", "사용자", "질의", "성공률",
                            "판단성질의비중", "주의도"] if c in sx.columns]
        A(T(sx[keep], {"성공률": lambda v: pct(v),
                       "판단성질의비중": lambda v: pct(v)}))
    if "segment" in D:
        sg = D["segment"].copy()
        A("<h3>시기별 구성 변화</h3>")
        A("<p>어떤 부류가 늘고 줄었는지 보면, 서비스의 성격이 어디로 "
          '이동했는지 드러납니다.<span class="tag a">확정</span></p>')
        A(T(sg.tail(8)))
    ucf = (F or {}).get("사용주기", {})
    if ucf:
        A("<h3>얼마나 자주 다시 오는가</h3>")
        A("<p>재방문 간격입니다. 이 값이 리텐션을 며칠 기준으로 볼지 결정합니다.</p>")
        A(T(pd.DataFrame([(k, f"{v:.0f}일") for k, v in ucf.items()],
                         columns=["지표", "값"])))
        adv = (F or {}).get("사용주기권고")
        if adv:
            A(f'<div class="box"><b>추적 주기</b>{_html.escape(str(adv))}</div>')

    if es is not None:
        A("<h3>새로 들어오는 사람들의 첫 질문이 이렇게 바뀌었습니다</h3>")
        A(T(es.head(8), {"초기": lambda v: pct(v), "최근": lambda v: pct(v),
                         "변화": lambda v: f"{float(v)*100:+.1f}%p"}))
        A('<div class="box"><b>읽는 법</b>'
          "이것은 <b>같은 사람이 변한 것이 아니라</b>, 새로 들어오는 사람들의 "
          "구성이 바뀌었다는 뜻입니다. 시장·종목을 살펴보러 오던 사람이 줄고 "
          "계좌·앱 업무를 처리하러 오는 사람이 늘었습니다.</div>")

    # ── 4. 하루 중 언제 무엇을 묻는가
    H2(4, "하루 중 언제 무엇을 묻는가",
       "시간대별로 질문의 성격이 뚜렷하게 갈립니다. 화면 구성과 응답 목표를 "
       "시간대별로 달리 잡을 근거가 됩니다.")
    if "sessprof" in D:
        sp = D["sessprof"].copy()
        A(T(sp, {c: (lambda v: pct(v, 0)) for c in sp.columns[1:]}))
    if "sesstop" in D:
        st = D["sesstop"].copy()
        c0 = st.columns[0]
        st = st.rename(columns={c0: "시간대"})
        A("<h3>시간대별 가장 많은 질문</h3>")
        A(T(st))
    A('<div class="box ok"><b>여기서 나오는 아이디어</b><ul>'
      "<li><b>장전</b> — 개장 전 브리핑 카드(전망·주요 일정)를 첫 화면에</li>"
      "<li><b>개장</b> — 응답 속도를 최우선으로. 이 구간만 별도 성능 목표</li>"
      "<li><b>장중</b> — 예상과 달리 계좌·앱 업무 문의가 많습니다. "
      "장중을 '거래 지원' 전용으로 설계하면 실제 수요와 어긋납니다</li>"
      "<li><b>장후</b> — 오늘의 내 계좌 요약을 먼저 제안</li>"
      "<li><b>야간</b> — 긴 답변이 허용되는 구간. 학습·안내형 콘텐츠에 적합</li>"
      "</ul></div>")

    # ════════════════════ 2부
    A('<div class="part">2부 · 왜 떠나는가</div>')

    # ── 5. 이탈은 세 층이다
    H2(5, "이탈은 세 층이다",
       "'이탈'을 한 덩어리로 보면 처방이 나오지 않습니다. 성격이 다른 세 가지를 "
       "분리했습니다.")
    dist = F.get("세션결과분포", {})
    rows = []
    if dist:
        rows.append(("① 대화 중단", "목적을 못 이루고 대화를 끝냄",
                     pct(dist.get("ABANDONED")), "AI·데이터"))
    if ab is not None and len(ab):
        top = ab.iloc[0]
        rows.append(("② 기능 포기", "한 번 실패한 기능을 다시 쓰지 않음",
                     f"최대 {float(top['포기효과'])*100:+.0f}%p 차이", "기능별 담당"))
    if sc:
        rows.append(("③ 서비스 이탈", "30일 이상 다시 오지 않음",
                     f"최근 {pct(sc['최근'])} 잔존", "프로덕트"))
    if rows:
        A(T(pd.DataFrame(rows, columns=["층", "무슨 일인가", "규모", "담당"])))
    dist2 = F.get("세션결과분포", {})
    if dist2:
        A("<h3>① 대화 중단 — 결과 분포</h3>")
        A(T(pd.DataFrame([
            ("한 번에 해결", pct(dist2.get("RESOLVED"))),
            ("여러 번 물어 해결", pct(dist2.get("RESOLVED_HARD"))),
            ("해결 못 하고 종료", pct(dist2.get("ABANDONED"))),
            ("상담원·불만으로 전환", pct(dist2.get("DEFLECTED"))),
        ], columns=["대화 결과", "비율"])))
        ns2 = F.get("세션", {})
        if ns2.get("고비용 성공 비율") is not None:
            A(f"<p>주목할 것은 <b>여러 번 물어 해결</b>한 비율입니다"
              f"({pct(ns2['고비용 성공 비율'])}). 최종적으로는 성공이지만 "
              f'사용자 입장에서는 실패에 가깝습니다.<span class="tag a">확정</span></p>')

    A('<div class="box warn"><b>세 층은 순서대로 일어납니다</b>'
      "대화가 끊기고 → 그 기능을 포기하고 → 결국 서비스를 떠납니다. "
      "<b>기능 포기는 서비스 이탈보다 먼저 나타나는 신호</b>이므로, "
      "이탈을 기다리지 말고 이쪽을 조기 경보로 쓰는 편이 낫습니다."
      '<span class="tag b">방향</span></div>')
    if ab is not None and len(ab):
        A("<h3>한 번 실패하면 다시 묻지 않는 기능</h3>")
        A("<p>같은 기능에서 실패한 사용자가 나중에 그 기능을 다시 쓰는 비율입니다. "
          "성공했을 때와 차이가 클수록, 한 번의 실패로 그 기능을 포기한다는 "
          "뜻입니다.</p>")
        A(T(ab, {"실패후_재시도율": lambda v: pct(v),
                 "성공후_재시도율": lambda v: pct(v),
                 "포기효과": lambda v: f"{float(v)*100:+.1f}%p"}))
    if "abandon" in D and len(D["abandon"]):
        pass
    if "cohort" in D and len(D["cohort"]):
        A("<h3>③ 서비스 이탈 — 가입 시기별 재방문율</h3>")
        ch = D["cohort"].copy()
        c0 = ch.columns[0]
        ch = ch.rename(columns={c0: "가입 시기", "사용자": "가입자 수"})
        A(T(ch.tail(10)))
        A("<p>빈칸은 아직 그 시점이 도래하지 않아 판정할 수 없는 구간입니다. "
          "0으로 읽으면 안 됩니다.</p>")
    if "km" in D and len(D["km"]):
        km = D["km"].copy()
        c0 = km.columns[0]
        km = km.rename(columns={c0: "경과일"})
        A("<h3>전체 이용자의 잔존 곡선</h3>")
        A(T(km, {"생존확률": lambda v: pct(v)}))
    plf = (F or {}).get("평탄화", {})
    if plf.get("판정"):
        A(f'<div class="box warn"><b>잔존 곡선 판정</b>'
          f'{_html.escape(str(plf["판정"]))}</div>')
    if sc:
        A("<h3>서비스 이탈 — 재방문율이 계속 떨어지고 있습니다</h3>")
        A(f"<p>가입 시기별로 나눠 보면 초기 집단 {pct(sc['초기'])}에서 최근 집단 "
          f"{pct(sc['최근'])}으로, 약 {sc['배수']:.1f}배 하락했습니다."
          + (f" 하락이 뚜렷해지는 시점은 <b>{sc['꺾인시기']}</b> 무렵입니다."
             if sc.get("꺾인시기") else "")
          + '<span class="tag a">확정</span></p>')
    rows = []
    if mx and mx["설명력"] == mx["설명력"]:
        rows.append(("들어오는 사람들의 구성이 바뀌었다", pct(mx["설명력"]),
                     "새 유입에서 탐색형이 줄고 업무형이 늘었습니다."))
    if qs:
        worse = qs["추세"] < -0.03
        rows.append(("응답 품질이 나빠졌다",
                     "일부 있음" if worse else "악화 근거 못 찾음(검수 전)",
                     f"한 번에 해결된 세션 비율은 {pct(qs['최소'])}~{pct(qs['최대'])} "
                     f"사이, 최근 3주는 초기 대비 {qs['추세']*100:+.1f}%p."))
    if mx and mx["설명력"] == mx["설명력"]:
        rows.append(("설명되지 않는 부분", pct(max(1 - mx["설명력"], 0)),
                     "같은 목적으로 들어온 사용자가 예전보다 덜 남습니다."))
    if rows:
        A("<h3>재방문율 하락의 원인을 셋으로 나눠 봤습니다</h3>")
        A(T(pd.DataFrame(rows, columns=["원인", "설명하는 몫", "내용"])))
        if "shift" in D and len(D["shift"]):
            sh2 = D["shift"].copy()
            A("<h4>기존 이용자가 변한 것인가, 사람이 바뀐 것인가</h4>")
            A("<p>비중 변화를 <b>같은 사람의 행동 변화</b>와 <b>사람이 바뀐 것</b>"
              "으로 나눠 봤습니다.</p>")
            A(T(sh2, {"기여": lambda v: f"{float(v):+.4f}",
                      "기여율": lambda v: pct(v)}))
        if "mixcf" in D and len(D["mixcf"]):
            mc2 = D["mixcf"].copy()
            c0 = mc2.columns[0]
            mc2 = mc2.rename(columns={c0: "시기", "관측잔존": "실제 재방문율",
                                      "믹스손실": "유입 구성 탓 손실"})
            keep = [c for c in ["시기", "신규", "실제 재방문율",
                                "유입 구성 탓 손실"] if c in mc2.columns]
            A("<h4>유입 구성이 초기와 같았다면</h4>")
            A(T(mc2[keep].tail(8), {"실제 재방문율": lambda v: pct(v),
                                    "유입 구성 탓 손실": lambda v: pct(v)}))
        A('<div class="box warn"><b>가장 큰 몫이 아직 설명되지 않았습니다</b>'
          "품질 변화와 유입 구성 변화를 합쳐도 하락의 일부만 설명됩니다. "
          "나머지는 <b>어디서 어떻게 들어오는가</b>의 문제일 가능성이 큽니다. "
          "앱 개편·배너 위치·진입 경로 변경 이력을 확인해야 좁혀집니다."
          '<span class="tag b">방향</span></div>')

    if "attr" in D and len(D["attr"]):
        at = D["attr"].copy()
        c0 = at.columns[0]
        at = at.rename(columns={c0: "문제"})
        A("<h3>어떤 문제가 대화를 끝냈는가</h3>")
        A("<p>실패로 끝난 대화를, <b>그 대화를 끝낸 마지막 실패</b>의 원인으로 "
          "나눴습니다. 마지막 실패는 대화당 하나뿐이라 겹치지 않고 합이 "
          '100%가 됩니다.<span class="tag a">확정</span></p>')
        keep = [c for c in ["문제", "세션수", "기여율", "종료위험배수",
                            "겪은세션비율", "담당", "처방"] if c in at.columns]
        A(T(at[keep].rename(columns={
            "세션수": "끝낸 대화 수", "기여율": "차지하는 몫",
            "종료위험배수": "대화를 끝낼 위험(평균=1)",
            "겪은세션비율": "이 문제를 겪은 대화 비율",
            "처방": "필요한 조치"}),
            {"차지하는 몫": lambda v: pct(v),
             "이 문제를 겪은 대화 비율": lambda v: pct(v),
             "대화를 끝낼 위험(평균=1)": lambda v: f"{float(v):.2f}배"}))
        hl = (F or {}).get("기여도요약")
        if hl:
            A(f'<div class="box ok"><b>한 문장으로</b>{_html.escape(hl)}</div>')
        A('<div class="box"><b>두 열을 함께 보십시오</b>'
          "<b>차지하는 몫</b>은 빈도이고 <b>대화를 끝낼 위험</b>은 치명도입니다. "
          "드물지만 치명적인 문제는 몫이 작아도 우선순위가 높습니다.</div>")
    if "attr_fix" in D and len(D["attr_fix"]):
        fx = D["attr_fix"].copy()
        keep = [c for c in ["시나리오", "치환건수", "무마찰해결률", "Δ무마찰"]
                if c in fx.columns]
        A("<h3>하나씩만 고쳤을 때 얼마나 좋아지는가</h3>")
        A("<p>각 문제를 <b>단독으로</b> 해결했다고 가정한 값입니다. 순서대로 "
          "누적한 것이 아니라 서로 비교할 수 있습니다. 다만 '고치면 그 질문이 "
          '성공한다\'는 가정 위의 상한입니다.<span class="tag c">가정 위 추정</span></p>')
        A(T(fx[keep].rename(columns={
            "시나리오": "고치는 대상", "치환건수": "해당 질문 수",
            "무마찰해결률": "한 번에 해결된 대화 비율",
            "Δ무마찰": "개선폭"}),
            {"한 번에 해결된 대화 비율": lambda v: pct(v),
             "개선폭": lambda v: f"{float(v)*100:+.1f}%p"}))
    if "attr_exp" in D and len(D["attr_exp"]):
        ae = D["attr_exp"].copy()
        keep = [c for c in ["문제", "겪은 사용자", "겪은쪽 이탈률",
                            "안겪은쪽 이탈률", "이탈률 차이",
                            "겪은쪽 세션수", "안겪은쪽 세션수"]
                if c in ae.columns]
        A("<h3>문제를 겪은 사용자는 실제로 더 떠났는가</h3>")
        A('<p>문제별로 따로 계산했으며 <b>합산하지 않습니다</b>. 한 사용자가 '
          '여러 문제를 겪기 때문입니다.<span class="tag b">방향</span></p>')
        A(T(ae[keep].head(7), {"겪은쪽 이탈률": lambda v: pct(v),
                               "안겪은쪽 이탈률": lambda v: pct(v),
                               "이탈률 차이": lambda v: f"{float(v)*100:+.1f}%p"}))
        A('<div class="box warn"><b>이 표를 인과로 읽지 마십시오</b>'
          "'세션수' 열을 함께 보십시오. 겪은 쪽이 훨씬 많이 사용한 사용자라면, "
          "이탈률 차이는 문제의 효과가 아니라 <b>오래 쓴 사람이 문제를 많이 "
          "만난 결과</b>입니다.</div>")
    if "attr_comb" in D and len(D["attr_comb"]):
        cb2 = D["attr_comb"].copy()
        keep = [c for c in ["문제", "겪은세션비율", "종료위험배수", "기여율",
                            "이탈률 차이", "Δ무마찰", "근거", "담당"]
                if c in cb2.columns]
        A("<h3>종합 — 근거 등급과 함께</h3>")
        A(T(cb2[keep].rename(columns={
            "겪은세션비율": "겪은 대화 비율", "종료위험배수": "대화를 끝낼 위험",
            "기여율": "중단 기여", "Δ무마찰": "단독 해결 효과",
            "근거": "근거 수준"}),
            {"겪은 대화 비율": lambda v: pct(v), "중단 기여": lambda v: pct(v),
             "대화를 끝낼 위험": lambda v: f"{float(v):.2f}배",
             "이탈률 차이": lambda v: f"{float(v)*100:+.1f}%p",
             "단독 해결 효과": lambda v: f"{float(v)*100:+.1f}%p"}))
        A('<div class="box"><b>근거 수준이 다릅니다</b>'
          "'준-인과'는 비교군을 맞춰 확인한 것이고, '상관'은 함께 나타난다는 "
          "뜻입니다. 상관 항목을 원인으로 단정하지 마십시오.</div>")
    if "attr_cnt" in D and len(D["attr_cnt"]):
        ac = D["attr_cnt"].copy()
        c0 = ac.columns[0]
        ac = ac.rename(columns={c0: "겪은 문제 수"})
        A("<h3>문제를 여러 개 겪을수록 나빠지는가</h3>")
        A(T(ac, {"이탈률": lambda v: pct(v)}))
        A('<div class="box warn"><b>이 표를 그대로 읽지 마십시오</b>'
          "질문을 많이 한 사용자가 문제도 많이 겪습니다. 평균 질문 수가 "
          "함께 늘고 있다면 <b>문제가 많아서 떠난 것이 아니라, 오래 쓴 사람이 "
          "문제를 많이 만난 것</b>일 수 있습니다.</div>")

    # ── 6. 여정 어디서 끊기는가
    H2(6, "여정 어디서 끊기는가",
       "사용자는 '찾고 → 살펴보고 → 주문한다'는 흐름을 따릅니다. "
       "실패했을 때 다음에 어디로 가는지를 보면 이탈 경로가 드러납니다.")
    if "transition" in D:
        tr = D["transition"].copy()
        keep = [c for c in tr.columns if c in ("from", "to", "실패후", "성공후", "차이")]
        A("<h3>실패한 뒤 어디로 가는가</h3>")
        A("<p>같은 이동이라도 앞선 질문이 성공했을 때와 실패했을 때 확률이 "
          "다릅니다. 차이가 클수록 <b>실패가 밀어낸 이동</b>입니다.</p>")
        A(T(tr[keep].head(8), {"실패후": lambda v: pct(v),
                               "성공후": lambda v: pct(v),
                               "차이": lambda v: f"{float(v)*100:+.1f}%p"}))
        A('<div class="box"><b>읽는 법</b>'
          "계좌·앱 업무나 배당·세금 질문에 실패하면 <b>오류 신고·상담원 연결로 "
          "이동하는 확률이 크게 오릅니다</b>. 업무 처리 실패가 상담 비용으로 "
          "옮겨 붙고 있다는 뜻입니다.</div>")
    if "exitpts" in D:
        ex = D["exitpts"].copy()
        c0 = ex.columns[0]
        ex = ex.rename(columns={c0: "질문 유형"})
        keep = [c for c in ["질문 유형", "n", "종료율", "lift"] if c in ex.columns]
        A("<h3>대화가 여기서 끝난다</h3>")
        A('<p>해당 질문 뒤에 대화가 끝날 확률이 평균의 몇 배인지 나타냅니다.'
          '<span class="tag a">확정</span></p>')
        A(T(ex[keep].head(8), {"종료율": lambda v: pct(v),
                               "lift": lambda v: f"{float(v):.2f}배"}))

    # ── 7. 예상과 달랐던 것
    H2(7, "예상과 달랐던 것",
       "현업에서 원인으로 자주 지목되는 가설들을 실제 데이터로 확인했습니다. "
       "맞은 것도 있고, 그렇지 않은 것도 있습니다.")

    def hyp(title, checked, result, means, limit, verdict="기각"):
        cls = "bad" if verdict == "기각" else ("ok" if verdict == "확인" else "warn")
        A(f'<div class="hyp {cls}"><b>가설 — {title}</b>'
          f"<p><i>확인한 것</i> {checked}</p>"
          f"<p><i>결과</i> {result}</p>"
          f"<p><i>뜻하는 것</i> {means}</p>"
          f"<p><i>한계</i> {limit}</p></div>")

    cxf = F.get("차단생존", {})
    ptf = F.get("차단", {})
    if cxf:
        lim = ("차단이 없던 기간이 데이터에 없어 정책 자체의 효과는 측정할 수 "
               "없습니다. 아래는 '차단을 겪은 사람과 안 겪은 사람'의 비교입니다.")
        hr = float(cxf.get("조정HR", float("nan")))
        pv = float(cxf.get("p", 1.0) or 1.0)
        ci = cxf.get("CI") or [np.nan, np.nan]
        sig = pv < 0.05 and not (ci[0] <= 1 <= ci[1])
        near1 = abs(hr - 1) < 0.15
        c3tail = (f" 반면 <b>거절한 뒤 아무 대안도 주지 않은 경우</b>, 대화 종료 "
                  f"확률이 평균의 <b>{ptf.get('종료위험배수', 0):.1f}배</b>로 전체에서 "
                  f"가장 높습니다."
                  if ptf and ptf.get("종료위험배수") else "")
        if sig and hr > 1.15:
            res = (f"차단을 겪은 쪽의 이탈 위험이 <b>{hr:.2f}배</b> 높았습니다 "
                   f"(95% 구간 {ci[0]:.2f}~{ci[1]:.2f}).")
            mean = ("차단 경험과 이탈이 함께 나타납니다. 다만 <b>차단 때문인지, "
                    "그런 질문을 하는 사용자라서인지는 구분되지 않습니다</b> — "
                    "두 집단은 애초에 다른 질문을 하는 사람들입니다." + c3tail)
            vd = "보류"
        elif near1 or not sig:
            res = (f"두 집단의 이탈 위험이 거의 같았습니다 ({hr:.2f}배, "
                   f"95% 구간 {ci[0]:.2f}~{ci[1]:.2f}).")
            mean = ("차단 경험 자체로는 이후 이탈이 갈리지 않습니다. 문제는 "
                    "<b>거절한 뒤 아무 대안도 주지 않는 것</b>입니다." + c3tail)
            vd = "기각"
        else:
            res = (f"차단을 겪은 쪽의 이탈 위험이 오히려 낮았습니다 ({hr:.2f}배).")
            mean = ("차단 경험은 이탈을 예측하지 못합니다. 차단을 겪는 사용자는 "
                    "적극적으로 질문하는 층이라 오히려 더 오래 남습니다." + c3tail)
            vd = "기각"
        hyp("규정상 차단이 사용자를 떠나게 한다",
            "차단을 겪은 사용자와 겪지 않은 사용자의 이후 이탈 위험 비교",
            res, mean, lim, verdict=vd)
    ltf = (F or {}).get("지연", {})
    if ltf:
        knee = ltf.get("꺾임") or []
        drop = ltf.get("최대낙차", float("nan"))
        sh = (F or {}).get("지연분산", {})
        step_exp = sh.get("호출 스텝수 설명력")
        extra = ""
        if step_exp is not None and step_exp == step_exp and step_exp > 0.3:
            extra = (f" 또한 응답 시간 차이의 상당 부분({pct(step_exp)})이 "
                     "<b>한 번에 몇 개의 기능을 연쇄 호출했는가</b>로 설명됩니다 — "
                     "속도 문제가 아니라 처리 방식의 문제입니다.")
        if not knee:
            res = (f"응답 경로(차단·분류실패·정상)를 나눈 뒤 정상 응답 안에서만 "
                   f"보면, 응답이 느려져도 대화 지속률의 낙차가 "
                   f"{pct(drop)}p 수준에 그쳤습니다.")
            mean = ("차단·분류실패를 걷어내고 정상 응답만 본 결과, 속도가 이탈을 "
                    "만든다는 근거는 <b>확인되지 않았습니다</b>. 다만 '없다'가 "
                    "확정된 것은 아니므로, 속도 개선의 성과를 이탈 감소로 "
                    "예단하지 마십시오." + extra)
            vd = "보류"
        else:
            res = (f"정상 응답 안에서 대화 지속률이 "
                   f"{min(knee):.0f}~{max(knee):.0f}초 구간에서 꺾였습니다 "
                   f"(최대 낙차 {pct(drop)}p).")
            mean = ("인내 한계가 존재합니다. 시간대별 응답 목표를 이 값 아래로 "
                    "잡는 것이 실효적입니다." + extra)
            vd = "확인"
        hyp("응답이 느려서 떠난다",
            "응답 경로를 먼저 나누고, 질문 유형·호출 스텝수를 통제한 뒤 "
            "지연 구간별 대화 지속률 비교", res, mean,
            "어려운 질문일수록 느리고 실패도 잦습니다. 경로와 유형을 통제해도 "
            "잔여 교란이 남을 수 있습니다.", verdict=vd)
        if "lat_path" in D:
            A("<h4>경로별 응답 시간과 결과</h4>")
            A("<p>같은 '느림'이라도 경로가 다르면 의미가 다릅니다. "
              "분류에 실패한 응답은 오히려 빨리 돌아옵니다.</p>")
            A(T(D["lat_path"], {"다음턴발생률": lambda v: pct(v),
                                "되묻기율": lambda v: pct(v),
                                "세션해결률": lambda v: pct(v)}))
        if "lat_session" in D:
            A("<h4>시간대별 — 같은 5초라도 체감이 다릅니다</h4>")
            ls = D["lat_session"]
            A(T(ls, {c: (lambda v: pct(v)) for c in ls.columns[1:]}))
    elif "latency" in D:
        lt = D["latency"]
        try:
            f0 = float(lt["실패율"].iloc[0]); f1 = float(lt["실패율"].iloc[-1])
        except Exception:
            f0 = f1 = float("nan")
        rev = (f0 == f0 and f1 == f1 and f0 > f1 + 0.1)
        if rev:
            res = (f"가장 빠른 구간의 실패율이 {pct(f0)}로 가장 느린 구간 "
                   f"{pct(f1)}보다 오히려 높았습니다. 인과가 반대입니다.")
            mean = ("느려서 실패한 것이 아니라 <b>실패한 응답이 빨리 돌아오는 "
                    "구조</b>입니다(기능 호출 없이 즉시 반환). 속도는 체감 품질 "
                    "관점의 과제이되, 이탈 원인으로 보고하면 잘못된 결론이 됩니다.")
            vd = "기각"
        else:
            try:
                e0 = float(lt["성공건_세션종료율"].dropna().iloc[0])
                e1 = float(lt["성공건_세션종료율"].dropna().iloc[-1])
            except Exception:
                e0 = e1 = float("nan")
            res = (f"성공한 응답만 놓고 보면, 대화 종료율이 빠른 구간 {pct(e0)}에서 "
                   f"느린 구간 {pct(e1)}으로 움직였습니다.")
            mean = ("응답이 느릴수록 대화를 끝낼 확률이 다소 올라갑니다. 다만 "
                    "차이가 크지 않아 <b>이탈의 주된 원인으로 보기는 어렵습니다</b>.")
            vd = "보류"
        hyp("응답이 느려서 떠난다",
            "응답 시간 구간별 대화 종료율과 실패율 비교", res, mean,
            "어려운 질문일수록 느리고 동시에 실패도 잦아, 속도만의 효과를 "
            "이 데이터로 완전히 분리할 수는 없습니다.", verdict=vd)
        A(T(lt.head(7), {c: (lambda v: pct(v)) for c in
                         ["실패율", "세션종료율", "복구성후속률", "성공건_세션종료율"]}))
    dsf = F.get("의존도충격", {})
    if dsf:
        hyp("외부 자료 공급 중단이 이탈을 키웠다",
            "해당 자료에 많이 의존하던 사용자와 그렇지 않은 사용자의 이후 행동 비교",
            "두 집단의 이탈률 차이가 뚜렷하지 않았습니다.",
            "자료 중단이 곧바로 이탈로 이어졌다고 보기는 어렵습니다. "
            "다만 관측 기간이 짧아 <b>판단 보류</b>가 정확한 표현입니다.",
            "중단 이후 관측 기간이 짧아 대부분의 사용자가 기계적으로 "
            "'이탈'로 잡혔습니다. 이 결과는 근거로 쓰지 마십시오.", verdict="보류")
    scf = F.get("자기검열", {})
    if scf:
        hyp("차단당한 사용자는 그 질문만 안 하게 된다",
            "차단 경험 전후로 같은 사용자의 '판단을 요구하는 질문' 비중 변화",
            f"차단 이후 해당 비중이 {pct(scf.get('차단전_판단성비중'))}에서 "
            f"{pct(scf.get('차단후_판단성비중'))}로 줄었습니다.",
            "그 질문 하나만 안 하는 것이 아니라 <b>비슷한 질문 전체를 덜 하게</b> "
            "됩니다. 수요가 사라진 것이 아니라 눌린 것이므로, 질문 수만 보면 "
            "이 수요는 보이지 않습니다.",
            "차단당하려면 그런 질문을 해야 하므로 직전 비중이 원래 높습니다. "
            "방향은 신뢰할 수 있으나 크기는 확정할 수 없습니다.", verdict="확인")
    mtf = F.get("멀티턴", {})
    if mtf and mtf.get("오즈비") == mtf.get("오즈비"):
        hyp("여러 번 묻는 건 사용자가 원래 그런 것이다",
            "앞선 답변의 성패에 따라 되묻기가 얼마나 늘어나는지 비교",
            f"앞선 답변이 실패했을 때 되묻기가 <b>{mtf['오즈비']:.1f}배</b> "
            "늘었습니다.",
            "되묻기는 사용자 습관이 아니라 <b>한 번에 답하지 못한 결과</b>입니다. "
            "되묻기 비율을 품질 지표로 쓸 수 있습니다.",
            "질문의 대상(종목)이 라벨에 없어 되묻기 판정 정확도에 한계가 "
            "있습니다.", verdict="확인")

    H2(7.5, "사용자가 실제로 알고 싶었던 것",
       "지금까지는 '실패'를 봤습니다. 여기서는 <b>성공했는데도 부족한 것</b>을 봅니다.")
    A("<p>「삼성전자 주가 알려줘」에 <b>70,100원</b>이라고 답하면 틀린 답이 "
      "아닙니다. 그런데 사용자가 정말 알고 싶었던 것은 대개 "
      "<b>왜 그렇게 움직였는지</b>, <b>앞으로 어떨지</b>입니다.</p>")
    A("<p>어떤 질문 뒤에 무엇이 따라오는지를 세면, <b>첫 답변이 채워주지 못한 "
      "것</b>이 드러납니다. 아래는 실패한 뒤의 되묻기가 아니라, "
      "<b>성공한 뒤에도 이어서 물은</b> 경우만 모은 것입니다."
      '<span class="tag a">확정</span></p>')
    if "need" in D and len(D["need"]):
        nd = D["need"].copy()
        keep = [c for c in ["표면 질문", "이어서 묻는 것", "건수"] if c in nd.columns]
        A(T(nd[keep].head(12)))
        A('<div class="box"><b>읽는 법</b>'
          "괄호 안 배수는 전체 평균 대비입니다. 2배면 그 질문에 고유한 후속이라는 "
          "뜻이고, 곧 <b>그 답변에 함께 담겼어야 할 정보</b>입니다.</div>")
    if "need_q" in D and len(D["need_q"]):
        nq = D["need_q"].copy()
        star = nq[nq["구분"].astype(str).str.startswith("★")] if "구분" in nq else nq
        if len(star):
            A("<h3>답은 맞는데 여러 번 묻게 만드는 질문</h3>")
            A("<p>되묻기(실패)가 적은데도 뒤에 질문이 계속 이어지는 유형입니다. "
              "<b>실패 지표에는 전혀 잡히지 않습니다.</b></p>")
            keep = [c for c in ["질문", "자기완결률", "되묻기율", "연쇄깊이"]
                    if c in star.columns]
            A(T(star[keep].head(8), {"자기완결률": lambda v: pct(v),
                                     "되묻기율": lambda v: pct(v)}))
            A('<div class="box ok"><b>여기서 나오는 과제</b>'
              "이건 모델을 바꿔서 풀리는 문제가 아니라 <b>응답 구성(템플릿) 과제</b>"
              "입니다. 위 후속 분포를 근거로 각 답변에 무엇을 함께 담을지 "
              "정하면 됩니다. 예정된 세 가지 개선 어디에도 들어 있지 않습니다.</div>")
    if "need_ctx" in D and len(D["need_ctx"]):
        A("<h3>같은 질문이라도 맥락에 따라 원하는 것이 다릅니다</h3>")
        A("<p>직전에 무엇을 하고 있었느냐에 따라 이어서 묻는 것이 달라집니다. "
          "<b>맥락에 따라 답변 구성을 달리해야 한다</b>는 근거이며, "
          "멀티턴 전환 계획과 직접 맞물립니다.</p>")
        A(T(D["need_ctx"].head(10)))

    # ════════════════════ 3부
    A('<div class="part">3부 · 왜 남는가</div>')

    H2(8, "남는 사용자는 무엇이 다른가",
       "떠나는 이유만큼 남는 이유도 중요합니다. 여기서 유지 레버가 나옵니다.")
    acf = F.get("활성화", {})
    if acf:
        cand = str(acf.get("후보", ""))
        head = ("첫 주에 여러 기능을 써본 사용자가 남습니다"
                if ("고유의도수" in cand or "고유단계수" in cand)
                else "첫 주에 자주 써본 사용자가 남습니다"
                if ("질의수" in cand or "세션수" in cand or "활동일수" in cand)
                else "첫 주의 특정 경험이 재방문을 가릅니다")
        A(f'<div class="box ok"><b>{head}</b>'
          f"가입 첫 주에 <b>{_html.escape(cand)}</b> 조건을 "
          f"충족한 사용자의 30일 후 재방문율은 "
          f"<b>{pct(acf.get('충족시_잔존'))}</b>로, 그렇지 않은 사용자 "
          f"{pct(acf.get('미충족시_잔존'))}의 약 "
          f"{float(acf.get('리프트', 0)):.1f}배입니다. "
          f"다만 해당 사용자는 전체의 {pct(acf.get('충족비율'))}에 불과합니다.</div>")
    if "activation" in D:
        ad = D["activation"].copy()
        keep = [c for c in ["후보", "충족비율", "충족시_잔존", "미충족시_잔존",
                            "리프트"] if c in ad.columns]
        A("<h3>어떤 첫 경험이 재방문으로 이어지는가</h3>")
        A(T(ad[keep].head(7), {"충족비율": lambda v: pct(v),
                               "충족시_잔존": lambda v: pct(v),
                               "미충족시_잔존": lambda v: pct(v),
                               "리프트": lambda v: f"{float(v):.2f}배"}))
        A('<div class="box"><b>주의</b>'
          "이것은 상관관계입니다. '여러 기능을 쓰게 하면 남는다'가 아니라 "
          "'여러 기능을 써본 사람이 남았다'는 뜻입니다. 실제 인과는 "
          "실험으로 확인해야 합니다.</div>")
    if er:
        A("<h3>두 번째 방문을 만드는 첫 질문</h3>")
        A(f"<p>처음 던진 질문에 따라 30일 후 재방문율이 <b>약 "
          f"{er['격차']:.1f}배</b> 차이 납니다. 통계적으로 유의한 것만 "
          f'추렸습니다.<span class="tag a">확정</span></p>')
        A(T(er["상위"], {"잔존율": lambda v: pct(v)}))
        A("<h3>반대로, 재방문으로 이어지지 않는 첫 질문</h3>")
        A(T(er["하위"], {"잔존율": lambda v: pct(v)}))
    if "retdrv" in D:
        rd = D["retdrv"].copy()
        c0 = rd.columns[0]
        rd = rd.rename(columns={c0: "질문 유형"})
        keep = [c for c in ["질문 유형", "성공효과"] if c in rd.columns]
        if "성공효과" in rd.columns:
            A("<h3>성공했을 때 재방문이 특히 늘어나는 기능</h3>")
            A("<p>개인별 성향을 제거하고, 같은 사람 안에서 성공과 실패를 "
              '비교했습니다.<span class="tag b">방향</span></p>')
            A(T(rd[keep].head(6), {"성공효과": lambda v: f"{float(v)*100:+.2f}%p"}))

    # ════════════════════ 4부
    A('<div class="part">4부 · 무엇을 할 것인가</div>')

    H2(9, "어느 사용자에 초점을 맞출 것인가",
       "모두를 위해 고칠 수는 없습니다. 규모·성공률·잔존을 함께 놓고 판단했습니다.")
    focus = pd.DataFrame([
        ("시장·종목 탐색형", "줄어드는 중", "중간", "높음",
         "★ 되찾을 대상 — 잔존이 가장 높은데 유입이 줄고 있습니다"),
        ("계좌·앱 업무형", "늘어나는 중", "높음", "낮음",
         "현상 유지 — 잘 처리되지만 재방문으로 이어지지 않습니다"),
        ("오류·상담형", "유지", "낮음", "낮음",
         "실패 감축 대상 — 다른 실패의 결과로 생겨납니다"),
    ], columns=["사용자 부류", "규모 추세", "성공률", "재방문", "판단"])
    A(T(focus))
    A('<div class="box ok"><b>권고</b>'
      "<b>시장·종목 탐색형을 되찾는 것</b>을 이번 분기의 초점으로 제안합니다. "
      "이유는 셋입니다. ① 첫 질문이 탐색형인 사용자의 재방문율이 가장 높습니다. "
      "② 그런데 신규 유입에서 이 비중이 가장 크게 줄었습니다. "
      "③ 첫 주에 여러 기능을 써보게 하는 유지 레버와 방향이 일치합니다.</div>")
    A("<p>구체적으로는 <b>진입 화면의 추천 질문을 탐색형으로 재배치</b>하는 것이 "
      "가장 빠른 시험입니다. 4장의 시간대별 성격과 결합하면, 장전·야간에는 "
      "탐색형을, 개장·장중에는 시세·주문형을 앞세우는 배치가 됩니다.</p>")

    H2(10, "지금 무엇을 하고 있는가",
       "예정된 세 가지 변경이 앞의 문제들과 어떻게 맞물리는지 먼저 정리합니다.")
    A(T(pd.DataFrame([
        ("분류 모델 고도화", "질문 유형 인식 실패 · 기능 미호출", "직접 해결",
         "가장 큰 문제를 정면으로 겨냥합니다. 다만 전환 <b>전에</b> "
         "'인식 실패를 실패로 기록'하는 조치가 선행되어야 효과를 잴 수 있습니다."),
        ("별도 분류 모델 도입 (속도)", "응답 속도", "부분 해당",
         "속도가 이탈을 만든다는 근거는 아직 확인되지 않았습니다(7장). "
         "성과를 속도가 아니라 <b>분류 정확도</b>로 평가하는 편이 정확합니다."),
        ("멀티턴 전환", "되묻기 · 앞 질문 맥락 유실", "부분 해당",
         "되묻기 경험은 나아지고, 맥락별 응답 구성(7.5장)의 토대가 됩니다. "
         "다만 답을 못 하는 원인(자료 부재·규정 차단)은 그대로 남습니다."),
    ], columns=["예정된 변경", "겨냥하는 문제", "정합성", "판단"])))
    A('<div class="box bad"><b>계획에 들어 있지 않은 문제</b>'
      "아래 셋은 예정된 변경 어디에도 없고, 분류 정확도가 올라가도 그대로 "
      "남습니다.<ul>"
      "<li><b>규정상 거절 후 대체 자료 제공</b> — 대화 종료 위험이 가장 높은 지점</li>"
      "<li><b>본인 계좌·자료 부재 대응</b> — 인증 처리와 자료 조달</li>"
      "<li><b>응답 구성(템플릿)</b> — 답은 맞는데 여러 번 묻게 만드는 지점(7.5장)</li>"
      "</ul></div>")

    H2(11, "무엇을 해야 하는가",
       "위에서 다루지 못하는 것을 포함해, 손대야 할 순서입니다.")
    A("<h3>1순위 — 질문 유형 인식 실패를 '실패'로 기록</h3>")
    if ofb:
        A(f"<p>전체의 {pct(ofb.get('OTH 비중'))}가 유형 판별에 실패해 뉴스·업무 "
          f"검색으로 넘어갑니다. 무관한 답을 내보내는 대신 <b>모른다고 답하거나 "
          f"되묻는 경로</b>를 만들어야 합니다. 이 조치 없이는 이후 어떤 개선도 "
          f'효과를 측정할 수 없습니다.<span class="tag a">확정</span></p>')
    if pt and pt["회수실패율"] == pt["회수실패율"]:
        A("<h3>2순위 — 거절 시 대체 안내 제공</h3>")
        A(f"<p>규정상 답할 수 없는 질문을 거절하는 것은 정상입니다. 문제는 "
          f"<b>거절만 하고 볼 자료를 주지 않는 경우가 {pct(pt['회수실패율'])}</b>"
          f"라는 점입니다."
          + (f" 이 경우 대화가 끝날 확률이 평균의 <b>{pt['종료위험배수']:.1f}배</b>로 "
             "전체에서 가장 높습니다." if pt.get("종료위험배수") else "")
          + '<span class="tag a">확정</span></p>')
    if pr is not None and len(pr):
        A("<h3>3순위 — 실패가 몰려 있는 기능</h3>")
        A(T(pr, {"실패율": lambda v: pct(v), "질의량": lambda v: num(v)}))
        A("<p>인증 처리·분류 개선·자료 조달로 담당이 갈리므로 병행 가능합니다.</p>")
    A("<h3>4순위 — 응답 구성 개선</h3>")
    A("<p>7.5장에서 확인된, <b>답은 맞는데 여러 번 묻게 만드는</b> 질문들입니다. "
      "각 답변에 무엇을 함께 담을지 정하는 작업이며, 모델 교체와 무관하게 "
      '별도로 진행할 수 있습니다.<span class="tag b">방향</span></p>')
    if ef is not None and len(ef):
        A(T(ef, {"고비용비율": lambda v: pct(v)}))
    if dm is not None and len(dm):
        A("<h3>5순위 — 아직 손대지 않은 기회</h3>")
        A("<p>질문 수는 적은데 줄고 있고 성공률도 낮은 기능입니다. "
          "<b>계속 실패해서 아예 묻지 않게 된 것인지</b> 확인이 필요합니다."
          '<span class="tag b">방향</span></p>')
        A(T(dm, {"성공률": lambda v: pct(v), "질의량": lambda v: num(v)}))

    H2(12, "얼마나 좋아지는가",
       "신뢰도가 다른 세 종류의 효과를 섞지 않았습니다.")
    A(T(pd.DataFrame([
        ("확정", "질문 유형 인식 실패를 '실패'로 기록",
         "수치는 오히려 낮아집니다. 대신 <b>처음으로 정확한 기준선</b>이 생깁니다."),
        ("가정 위 추정", "자료·기능·인증·거절 후속 처리 해결",
         "아래 표의 상한까지 오를 수 있습니다."),
        ("실험 필요", "진입 추천 질문 재배치 · 응답 구성 개선",
         "개선이 예상되나 A/B 로 확인해야 합니다."),
    ], columns=["신뢰도", "무엇을", "예상 효과"])))
    if "northstar" in D and len(D["northstar"]):
        ns3 = D["northstar"].copy()
        c0 = ns3.columns[0]
        ren = {c0: "주차", "무마찰해결률": "한 번에 해결된 대화",
               "세션해결률": "결국 해결된 대화", "폴백세션비율": "인식 실패 포함 대화",
               "이탈세션률": "해결 못한 대화", "NorthStar(보조)": "주 1회 이상 성공(보조)"}
        ns3 = ns3.rename(columns=ren)
        keep = [c for c in ["주차", "진입사용자", "한 번에 해결된 대화",
                            "결국 해결된 대화", "해결 못한 대화",
                            "인식 실패 포함 대화", "주 1회 이상 성공(보조)"]
                if c in ns3.columns]
        A("<h3>추적 지표 추이</h3>")
        A("<p>개선 전후를 비교할 기준선입니다. <b>'주 1회 이상 성공'은 천장에 "
          "붙어 개선을 추적할 수 없으므로</b>, <b>한 번에 해결된 대화 비율</b>을 "
          '주 지표로 쓰는 것을 제안합니다.<span class="tag a">확정</span></p>')
        A(T(ns3[keep].tail(10), {c: (lambda v: pct(v)) for c in
                                 ["한 번에 해결된 대화", "결국 해결된 대화",
                                  "해결 못한 대화", "인식 실패 포함 대화",
                                  "주 1회 이상 성공(보조)"]}))

    A('<div class="box warn"><b>먼저 합의가 필요합니다</b>'
      "첫 번째 조치를 하면 <b>보고되는 성공률이 떨어집니다.</b> "
      "이것은 악화가 아니라 교정입니다. 착수 전에 관련 부서와 이 점을 "
      "합의해 두어야 나중에 혼선이 없습니다.</div>")
    if "attr_fix" in D and len(D["attr_fix"]):
        fx2 = D["attr_fix"].copy()
        keep = [c for c in ["시나리오", "치환건수", "무마찰해결률", "Δ무마찰"]
                if c in fx2.columns]
        A("<h3>문제별 단독 해결 효과</h3>")
        A("<p>각 문제를 <b>하나씩만</b> 해결했다고 가정한 값이라 서로 비교할 수 "
          '있습니다.<span class="tag c">가정 위 상한</span></p>')
        A(T(fx2[keep].rename(columns={
            "시나리오": "고치는 대상", "치환건수": "해당 질문 수",
            "무마찰해결률": "한 번에 해결된 대화 비율", "Δ무마찰": "개선폭"}),
            {"한 번에 해결된 대화 비율": lambda v: pct(v),
             "개선폭": lambda v: f"{float(v)*100:+.1f}%p"}))
    if cl is not None and len(cl):
        A("<h3>순차 해결 시 상한</h3>")
        A(T(cl, {"세션성공률": lambda v: pct(v), "무마찰세션률": lambda v: pct(v),
                 "Δ무마찰세션률": lambda v: f"{float(v)*100:+.1f}%p"}))

    H2(13, "의사결정 요청")
    if "gaps" in D and len(D["gaps"]):
        g = D["gaps"]
        need2 = g[g["상태"].astype(str).eq("결핍")]
        if len(need2):
            A("<h3>답하지 못한 질문과 필요한 데이터</h3>")
            A("<p>아래는 <b>분석의 한계가 아니라 데이터 요청</b>입니다.</p>")
            cols = [c for c in ["질문", "필요", "확보시", "난이도", "담당", "일정"]
                    if c in need2.columns]
            A(T(need2[cols].rename(columns={
                "필요": "필요한 데이터", "확보시": "확보하면 가능해지는 것"})))
    if (F or {}).get("결핍", {}).get("불능비율") is not None:
        A(f'<div class="box warn"><b>설계한 분석의 '
          f'{pct(F["결핍"]["불능비율"])}가 결론에 이르지 못했습니다</b>'
          f"위 데이터가 확보되면 대부분 해소됩니다.</div>")
    A("<h3>부서별 요청</h3>")
    A(T(pd.DataFrame([
        ("개발", "인식 실패를 '실패'로 기록 · 무관한 답 대신 되묻기 경로 신설",
         "착수 여부"),
        ("준법", "거절 시 제공 가능한 대체 정보의 범위 확정", "범위 승인"),
        ("프로덕트", "추적 지표를 재방문율에서 '한 번에 해결된 대화 비율'로 전환 · "
                     "진입 추천 질문 재배치 시험", "지표 전환 승인"),
        ("기획", "분류 모델 전환 일정에 '거절 후 대체 안내'와 '응답 구성 개선'을 "
                 "함께 포함할지", "로드맵 반영"),
        ("데이터", "답변 가능성 라벨 · 질문 대상(종목) · 진입 경로 기록 추가",
         "작업 반영"),
    ], columns=["대상", "요청 내용", "필요한 결정"])))
    A('<div class="box"><b>이 보고서가 답하지 못한 것</b>'
      "재방문율 하락의 가장 큰 몫이 아직 설명되지 않았습니다(5장). "
      "또한 답변이 실제로 질문에 맞는지는 사람이 표본을 확인해야 하며, "
      "그 작업이 예정되어 있습니다.</div>")

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>챗봇 서비스 진단 보고 {_html.escape(meta['now'])}</title>
<style>{CSS}</style></head><body><div class="wrap">
{''.join(P)}
</div></body></html>"""


def build_formal(D: dict, meta: dict, F: dict | None = None) -> str:
    """부서 간 공식 배포용. 개조식·명사형 표제·별첨 분리."""
    F = F or {}
    P: list[str] = []
    A = lambda x: P.append(x)
    FN: list[str] = []

    def fn(text: str) -> str:
        FN.append(text)
        return f"<sup>{len(FN)}</sup>"

    def flush_fn():
        if FN:
            A('<div class="fn">'
              + "".join(f"<div>{i+1}) {_html.escape(t)}</div>"
                        for i, t in enumerate(FN)) + "</div>")
            FN.clear()

    def note(title, text):
        A(f'<div class="note"><b>{title}</b> {text}</div>')

    sc = f_scale(D); qs = f_quality_stable(D); mx = f_mix(D)
    fb = f_fallback(D); uc = f_uncovered(D); er = f_entry_ret(D)
    pr = f_priority(D); ef = f_effort(D); dm = f_demand(D); es = f_entry_shift(D)
    pt = f_protector(D); ab = f_abandon(D); stt = f_structure(D)
    rv = f_relevance(D); cl = f_ceiling(D)
    ofb = F.get("폴백", {}); dist = F.get("세션결과분포", {})

    # ── 표지
    A('<div class="cover"><h1>MTS 챗봇 서비스 이용 실태 분석 결과</h1><table>'
      f'<tr><td>문서번호</td><td>{_html.escape(meta.get("docno",""))}</td></tr>'
      f'<tr><td>작성부서</td><td>{_html.escape(meta.get("dept",""))}</td></tr>'
      f'<tr><td>작성일</td><td>{meta["now"]}</td></tr>'
      f'<tr><td>분석기간</td><td>{_html.escape(meta["period"])}</td></tr>'
      f'<tr><td>배포범위</td><td>{_html.escape(meta.get("scope",""))}</td></tr>'
      '</table></div>')

    A('<div class="toc"><b>목 차</b><ol>'
      '<li>Ⅰ. 개요</li><li>Ⅱ. 요약</li><li>Ⅲ. 분석 결과</li>'
      '<li>Ⅳ. 개선 방안</li><li>Ⅴ. 협조 요청</li>'
      '<li>별첨 1. 지표 정의 · 별첨 2. 데이터 제약 사항 · 별첨 3. 가설 검증 상세</li>'
      "</ol></div>")

    # ── Ⅰ. 개요
    A("<h2>Ⅰ. 개요</h2>")
    A(T(pd.DataFrame([
        ("목적", "챗봇 이용 실태를 진단하고 개선 과제의 우선순위를 도출함"),
        ("분석 범위", "챗봇 질의 로그 전수. 재어노테이션된 의도 라벨 기준"),
        ("분석 기간", meta["period"]),
        ("데이터 출처", "챗봇 대화 로그 · 함수 호출 기록 · 의도 재어노테이션 결과"),
        ("분석 방법", "의도·여정 단계별 집계, 이용자 코호트 분석, "
                     "가설별 대조군 비교"),
        ("제약 사항", "일부 항목은 현행 데이터로 측정이 불가하며 별첨 2에 정리함"),
    ], columns=["구분", "내용"])))

    # ── Ⅱ. 요약
    A("<h2>Ⅱ. 요약</h2>")
    A("<h3>1. 핵심 결과</h3>")
    rows = []
    if ofb.get("OTH 비중") is not None:
        rows.append(("측정 신뢰도",
                     f"질의 유형 판별 실패분이 전체의 {pct(ofb['OTH 비중'])}이며, "
                     f"해당 응답이 성공으로 집계되고 있음{fn('로그 집계치. 별도 검증 불요.')}"))
    if sc:
        rows.append(("재방문율",
                     f"가입 시기별 30일 재방문율이 {pct(sc['초기'])}에서 "
                     f"{pct(sc['최근'])}으로 하락함(약 {sc['배수']:.1f}배)"))
    if mx and mx["설명력"] == mx["설명험"] if False else (mx and mx["설명력"] == mx["설명력"]):
        rows.append(("하락 요인",
                     f"이용자 구성 변화로 설명되는 몫은 {pct(mx['설명력'])}이며, "
                     f"나머지는 미규명 상태임{fn('진입 경로 로그 부재로 추가 규명 불가. 별첨 2 참조.')}"))
    if pt and pt["회수실패율"] == pt["회수실패율"]:
        rows.append(("규정상 거절 처리",
                     f"거절 후 대체 정보를 제공하지 않은 비율이 {pct(pt['회수실패율'])}이며, "
                     f"해당 시점의 대화 종료 위험이 가장 높음"))
    if er:
        rows.append(("진입 질의",
                     f"최초 질의 유형에 따라 30일 재방문율이 약 {er['격차']:.1f}배 "
                     f"차이를 보임"))
    if rows:
        A(T(pd.DataFrame(rows, columns=["항목", "내용"])))
    hl = F.get("기여도요약")
    if hl:
        note("종합", _html.escape(hl))

    A("<h3>2. 조치 요청 사항</h3>")
    A(T(pd.DataFrame([
        ("개발", "질의 유형 판별 실패를 실패로 기록하도록 변경", "착수 여부"),
        ("준법", "거절 시 제공 가능한 대체 정보의 범위 확정", "범위 승인"),
        ("프로덕트", "추적 지표를 재방문율에서 세션 해결률로 전환", "지표 전환 승인"),
        ("기획", "기 계획 과제에 거절 후속 처리·응답 구성 개선 포함 여부", "로드맵 반영"),
        ("데이터", "답변 가능성 라벨·질의 대상·진입 경로 기록 추가", "작업 반영"),
    ], columns=["대상 부서", "요청 내용", "필요 결정"])))
    flush_fn()

    # ── Ⅲ. 분석 결과
    A("<h2>Ⅲ. 분석 결과</h2>")

    A("<h3>Ⅲ-1. 측정 신뢰도 검토</h3>")
    A("<p>챗봇은 질의를 유형별로 분류한 뒤 해당 기능을 호출하는 구조임. "
      "유형 판별에 실패한 질의는 전량 기타로 분류되어 뉴스·업무 검색 기능이 "
      "호출되며, 해당 기능은 항상 응답을 생성하므로 질의와 무관한 응답도 "
      "성공으로 기록됨.</p>")
    if ofb:
        A(T(pd.DataFrame([
            ("유형 판별 실패 비중", pct(ofb.get("OTH 비중"))),
            ("현행 보고 성공률", pct(ofb.get("보고 성공률"))),
            ("판별 실패분 제외 시 성공률(하한)", pct(ofb.get("폴백 제외 성공률(하한)"))),
        ], columns=["구분", "값"])))
        note("유의사항",
             "실제 성공률은 위 두 값 사이에 위치함. 확정을 위해서는 표본 검수가 "
             "필요하며 별첨 2에 관련 사항을 정리함.")
    if uc and uc["합계"] == uc["합계"]:
        A(f"<p>기존 분류 체계로 포괄되지 않던 질의가 전체의 "
          f"{pct(uc['합계'])}로 확인됨.</p>")
        A(T(uc["표"].head(6), {"비중": lambda v: pct(v, 2)}))

    om = F.get("운영분류", {})
    if om.get("정합률") is not None:
        A("<h4>가. 기존 분류 체계의 해상도</h4>")
        A(f"<p>운영 분류와 재정리 기준의 정합률 {pct(om['정합률'])}{fn('한 카테고리가 여러 성격의 질문을 포함. 분해도 1에 가까울수록 라우팅 정확도 확보 곤란')}. "
          "카테고리별 분해 정도는 아래와 같음.</p>")
        if "legacy" in D and len(D["legacy"]):
            lg = D["legacy"].copy(); c0 = lg.columns[0]
            lg = lg.rename(columns={c0: "기존 분류", "n": "건수",
                                    "신규의도수": "포함 질문 종류",
                                    "분해도": "분해 정도"})
            keep = [c for c in ["기존 분류", "건수", "정합률", "포함 질문 종류",
                                "분해 정도"] if c in lg.columns]
            A(T(lg[keep].head(8), {"정합률": lambda v: pct(v)}))
    if "relbyintent" in D and len(D["relbyintent"]):
        rvv = D["relbyintent"].copy(); c0 = rvv.columns[0]
        rvv = rvv.rename(columns={c0: "질문 유형", "coverage": "핵심어 반영률",
                                  "n": "건수"})
        A("<h4>나. 질문 핵심어의 응답 반영률</h4>")
        A("<p>질의에 포함된 종목명·지표명이 응답에 등장하는 비율. "
          "낮을수록 응답과 질의의 불일치 가능성이 높음.</p>")
        A(T(rvv.head(6), {"핵심어 반영률": lambda v: f"{float(v):.2f}"}))
        if fb and (F.get("폴백", {}).get("coverage 격차") is not None):
            A(f"<p>단, 인식 실패 건과 정상 건의 반영률 격차가 "
              f"{F['폴백']['coverage 격차']:.2f}에 그쳐 자동 판별에는 한계가 있음"
              f"{fn('표본 인적 검수가 필요한 근거')}.</p>")
    qdf = F.get("질의왜곡", {})
    if qdf.get("전체 보존율(중앙)") is not None:
        A("<h4>다. 검색어 재작성 과정의 질의 보존</h4>")
        rows = [("질의의 검색어 보존율(중앙값)", f"{qdf['전체 보존율(중앙)']:.2f}"),
                ("보존율 50% 미만 비중", pct(qdf.get("보존율 0.5 미만 비중")))]
        if qdf.get("질문→쿼리 손실(검색 단계)") is not None:
            rows += [("검색 단계 손실", pct(qdf["질문→쿼리 손실(검색 단계)"])),
                     ("생성 단계 손실", pct(qdf.get("쿼리→응답 손실(생성 단계)")))]
        A(T(pd.DataFrame(rows, columns=["항목", "값"])))
        _s1, _s2 = (qdf.get("질문→쿼리 손실(검색 단계)"),
                    qdf.get("쿼리→응답 손실(생성 단계)"))
        if _s1 is not None and _s2 is not None:
            note("판단", "검색 단계 손실이 우세 — 모델 교체가 아닌 검색어 재작성 "
                 "로직의 개선이 필요함" if _s1 > _s2 else
                 "생성 단계 손실이 우세 — 검색 결과를 응답으로 옮기는 과정의 "
                 "개선이 필요함")
    hzf = F.get("환각위험", {})
    if hzf.get("전체위험군비율") is not None:
        A("<h4>라. 자료 조회 없이 생성된 응답</h4>")
        A(f"<p>사실 확인이 필요한 영역에서 자료 조회 없이 생성된 응답 "
          f"{pct(hzf['전체위험군비율'])}{fn('오정보가 사실처럼 제시될 위험 구간')}.</p>")
        if "halluc" in D and len(D["halluc"]):
            hcd = D["halluc"].copy(); c0 = hcd.columns[0]
            hcd = hcd.rename(columns={c0: "질문 유형", "n": "건수",
                                      "위험군비율": "근거 미확보 비율",
                                      "무툴응답": "자료 미조회 비율"})
            A(T(hcd.head(6), {"근거 미확보 비율": lambda v: pct(v),
                              "자료 미조회 비율": lambda v: pct(v)}))
    if "slotint" in D and len(D["slotint"]):
        sl = D["slotint"].copy(); c0 = sl.columns[0]
        sl = sl.rename(columns={c0: "질문 유형", "n": "건수",
                                "target복원율": "질의 대상 인식률"})
        A("<h4>마. 질의 대상 인식률</h4>")
        A("<p>질의 대상(종목·기간)의 인식 비율. 응답 정확도 및 재질의 판정 "
          "정확도에 직접 영향.</p>")
        A(T(sl[[c for c in ["질문 유형", "건수", "질의 대상 인식률"]
                if c in sl.columns]].head(6),
            {"질의 대상 인식률": lambda v: pct(v)}))
    bdf = F.get("차단정의", {})
    if bdf.get("kappa") is not None:
        A("<h4>바. 차단 판정의 신뢰도</h4>")
        A(f"<p>로그 표기 기준과 응답 구조 추정 기준의 일치도 "
          f"{bdf['kappa']:.2f}. "
          + ("두 기준이 일치하므로 차단 관련 수치의 신뢰도 확보."
             if bdf["kappa"] >= .8 else
             "일치도가 낮아 차단 관련 수치는 방향성 참고에 한함.") + "</p>")

    A("<h3>Ⅲ-2. 이용자 구성 현황</h3>")
    if "newret" in D and len(D["newret"]):
        A(T(D["newret"].tail(8), {"신규 비중": lambda v: pct(v),
                                  "재방문 비중": lambda v: pct(v)}))
        uc2 = F.get("사용자구성", {})
        if uc2.get("판정"):
            note("해석", _html.escape(uc2["판정"]))
    if "segcross" in D:
        sx = D["segcross"].copy()
        sx = sx.rename(columns={sx.columns[0]: "이용자 부류"})
        keep = [c for c in ["이용자 부류", "사용자", "질의", "성공률",
                            "판단성질의비중", "주의도"] if c in sx.columns]
        A("<h4>이용자 부류별 현황</h4>")
        A(T(sx[keep], {"성공률": lambda v: pct(v),
                       "판단성질의비중": lambda v: pct(v)}))
    if es is not None:
        A("<h4>신규 이용자의 최초 질의 구성 변화</h4>")
        A(T(es.head(8), {"초기": lambda v: pct(v), "최근": lambda v: pct(v),
                         "변화": lambda v: f"{float(v)*100:+.1f}%p"}))
        note("해석", "기존 이용자의 행동 변화가 아니라 신규 유입 구성의 "
                     "변화에 해당함.")

    ucf = F.get("사용주기", {})
    if ucf:
        A("<h4>재방문 간격</h4>")
        A(T(pd.DataFrame([(k, f"{v:.0f}일") for k, v in ucf.items()],
                         columns=["지표", "값"])))
        if F.get("사용주기권고"):
            note("추적 주기", _html.escape(str(F["사용주기권고"])))

    A("<h3>Ⅲ-3. 시간대별 이용 패턴</h3>")
    if "sessprof" in D:
        sp = D["sessprof"].copy()
        A(T(sp, {c: (lambda v: pct(v, 0)) for c in sp.columns[1:]}))
    if "sesstop" in D:
        st = D["sesstop"].copy()
        st = st.rename(columns={st.columns[0]: "시간대"})
        A(T(st))
    A("<p>시간대별 질의 성격이 뚜렷하게 구분됨. 특히 장중 시간대에 계좌·앱 "
      "업무 관련 질의가 다수를 차지하여, 해당 시간대를 거래 지원 전용으로 "
      "설계할 경우 실제 수요와 상이할 수 있음.</p>")

    A("<h3>Ⅲ-4. 이탈 유형별 분석</h3>")
    rows = []
    if dist:
        rows.append(("대화 중단", "목적 미달성 상태로 대화 종료",
                     pct(dist.get("ABANDONED")), "AI·데이터"))
    if ab is not None and len(ab):
        rows.append(("기능 포기", "특정 기능 실패 후 재시도하지 않음",
                     f"최대 {float(ab.iloc[0]['포기효과'])*100:+.0f}%p", "기능별 담당"))
    if sc:
        rows.append(("서비스 이탈", "30일 이상 미방문",
                     f"최근 {pct(sc['최근'])} 잔존", "프로덕트"))
    if rows:
        A(T(pd.DataFrame(rows, columns=["유형", "정의", "규모", "담당"])))
        note("해석", "세 유형은 순차적으로 발생하는 것으로 관측됨. 기능 포기는 "
                     "서비스 이탈에 선행하는 지표로 활용 가능함.")
    if "attr" in D and len(D["attr"]):
        at = D["attr"].copy()
        at = at.rename(columns={at.columns[0]: "문제"})
        A("<h4>대화 중단 원인별 귀속</h4>")
        keep = [c for c in ["문제", "세션수", "기여율", "종료위험배수",
                            "겪은세션비율", "담당"] if c in at.columns]
        A(T(at[keep].rename(columns={
            "세션수": "해당 대화 수", "기여율": "구성비",
            "종료위험배수": "종료 위험(평균=1)", "겪은세션비율": "발생 대화 비율"}),
            {"구성비": lambda v: pct(v), "발생 대화 비율": lambda v: pct(v),
             "종료 위험(평균=1)": lambda v: f"{float(v):.2f}"}))
        note("산출 방법", "대화를 종료시킨 마지막 실패의 원인으로 귀속함. "
                          "대화당 1건이므로 구성비 합은 100%임.")
    if sc and mx:
        A("<h4>재방문율 하락 요인 분해</h4>")
        rows = [("이용자 구성 변화", pct(mx["설명력"])),
                ("응답 품질 변화",
                 "악화 근거 미확인" + fn("표본 검수 이전 결과로, 품질 문제가 없다는 "
                                        "확정 판단은 아님.")),
                ("미규명", pct(max(1 - mx["설명력"], 0)))]
        A(T(pd.DataFrame(rows, columns=["요인", "설명 몫"])))

    dist2 = F.get("세션결과분포", {})
    if dist2:
        A("<h4>가. 대화 결과 분포</h4>")
        A(T(pd.DataFrame([
            ("1회 문답으로 해결", pct(dist2.get("RESOLVED"))),
            ("복수 문답 후 해결", pct(dist2.get("RESOLVED_HARD"))),
            ("미해결 종료", pct(dist2.get("ABANDONED"))),
            ("상담 채널 전환", pct(dist2.get("DEFLECTED"))),
        ], columns=["대화 결과", "비율"])))
    if "cohort" in D and len(D["cohort"]):
        A("<h4>나. 가입 시기별 재방문율</h4>")
        ch = D["cohort"].copy(); c0 = ch.columns[0]
        ch = ch.rename(columns={c0: "가입 시기", "사용자": "가입자 수"})
        A(T(ch.tail(10)))
        A(f"<p>공란은 관측 기간 미도래 구간{fn('0으로 해석 불가')}.</p>")
    if "km" in D and len(D["km"]):
        km = D["km"].copy(); c0 = km.columns[0]
        km = km.rename(columns={c0: "경과일"})
        A("<h4>다. 전체 잔존 곡선</h4>")
        A(T(km, {"생존확률": lambda v: pct(v)}))
    plf = F.get("평탄화", {})
    if plf.get("판정"):
        note("잔존 곡선 판정", _html.escape(str(plf["판정"])))
    if "shift" in D and len(D["shift"]):
        A("<h4>라. 비중 변화의 요인 분해</h4>")
        A("<p>기존 이용자의 행동 변화분과 이용자 구성 변화분으로 분해.</p>")
        A(T(D["shift"], {"기여": lambda v: f"{float(v):+.4f}",
                         "기여율": lambda v: pct(v)}))
    if "mixcf" in D and len(D["mixcf"]):
        mc2 = D["mixcf"].copy(); c0 = mc2.columns[0]
        mc2 = mc2.rename(columns={c0: "시기", "관측잔존": "실제 재방문율",
                                  "믹스손실": "유입 구성 요인 손실"})
        A("<h4>마. 유입 구성 고정 시 반사실 추정</h4>")
        A(T(mc2[[c for c in ["시기", "신규", "실제 재방문율",
                             "유입 구성 요인 손실"] if c in mc2.columns]].tail(8),
            {"실제 재방문율": lambda v: pct(v),
             "유입 구성 요인 손실": lambda v: pct(v)}))
    if "attr_exp" in D and len(D["attr_exp"]):
        ae = D["attr_exp"].copy()
        A("<h4>바. 문제 경험자와 미경험자의 이탈률 비교</h4>")
        A(f"<p>문제별 개별 산출이며 합산 불가{fn('한 이용자가 복수 문제를 경험하므로 중복')}. "
          "경험자의 이용량이 많을 경우 선택 편향이 존재.</p>")
        keep = [c for c in ["문제", "겪은 사용자", "겪은쪽 이탈률",
                            "안겪은쪽 이탈률", "이탈률 차이", "겪은쪽 세션수",
                            "안겪은쪽 세션수"] if c in ae.columns]
        A(T(ae[keep].head(7), {"겪은쪽 이탈률": lambda v: pct(v),
                               "안겪은쪽 이탈률": lambda v: pct(v),
                               "이탈률 차이": lambda v: f"{float(v)*100:+.1f}%p"}))
    if "attr_cnt" in D and len(D["attr_cnt"]):
        ac2 = D["attr_cnt"].copy(); c0 = ac2.columns[0]
        ac2 = ac2.rename(columns={c0: "경험 문제 수"})
        A("<h4>사. 경험 문제 수별 이탈률</h4>")
        A(T(ac2, {"이탈률": lambda v: pct(v)}))
        note("해석 유의", "질의량이 많은 이용자가 문제도 다수 경험. "
             "평균 질의수를 함께 확인할 것.")
    if "attr_comb" in D and len(D["attr_comb"]):
        cb2 = D["attr_comb"].copy()
        A("<h4>아. 문제별 종합 (근거 등급 포함)</h4>")
        keep = [c for c in ["문제", "겪은세션비율", "종료위험배수", "기여율",
                            "이탈률 차이", "Δ무마찰", "근거", "담당"]
                if c in cb2.columns]
        A(T(cb2[keep].rename(columns={
            "겪은세션비율": "경험 대화 비율", "종료위험배수": "대화 종료 위험",
            "기여율": "중단 기여", "Δ무마찰": "단독 해결 효과", "근거": "근거 등급"}),
            {"경험 대화 비율": lambda v: pct(v), "중단 기여": lambda v: pct(v),
             "대화 종료 위험": lambda v: f"{float(v):.2f}배",
             "이탈률 차이": lambda v: f"{float(v)*100:+.1f}%p",
             "단독 해결 효과": lambda v: f"{float(v)*100:+.1f}%p"}))
        note("근거 등급", "준-인과는 비교군 통제 후 산출, 상관은 동시 발생 관계. "
             "상관 항목의 인과 해석 불가.")

    A("<h3>Ⅲ-5. 이용 흐름상 중단 지점</h3>")
    if "transition" in D:
        tr = D["transition"].copy()
        keep = [c for c in tr.columns if c in ("from", "to", "실패후", "성공후", "차이")]
        A(T(tr[keep].head(8), {"실패후": lambda v: pct(v), "성공후": lambda v: pct(v),
                               "차이": lambda v: f"{float(v)*100:+.1f}%p"}))
        A("<p>계좌·앱 업무 및 배당·세금 관련 질의 실패 시 오류 신고·상담원 연결로 "
          "이동하는 비율이 유의하게 상승함. 업무 처리 실패가 상담 수요로 "
          "전이되는 것으로 판단됨.</p>")

    A("<h3>Ⅲ-6. 기존 가설 검증 결과</h3>")
    A('<p class="lead">현업에서 원인으로 지목되어 온 사항을 데이터로 확인한 '
      "결과임. 상세 내용은 별첨 3에 수록함.</p>")
    rows = []
    cxf = F.get("차단생존", {})
    if cxf:
        hr = float(cxf.get("조정HR", float("nan")))
        rows.append(("규정상 차단이 이탈을 유발함",
                     f"차단 경험 여부에 따른 이탈 위험비 {hr:.2f}배",
                     "미확인" if abs(hr - 1) < .15 else "판단 보류",
                     "차단 미시행 기간 부재로 정책 효과 자체는 측정 불가"))
    ltf = F.get("지연", {})
    if ltf:
        rows.append(("응답 지연이 이탈을 유발함",
                     f"정상 응답 기준 대화 지속률 낙차 {pct(ltf.get('최대낙차'))}p",
                     "판단 보류",
                     "차단·판별실패 경로 제외 후 재측정한 결과이며 확정 아님"))
    mtf = F.get("멀티턴", {})
    if mtf and mtf.get("오즈비") == mtf.get("오즈비"):
        rows.append(("반복 질의는 이용자 습관임",
                     f"직전 실패 시 반복 질의 발생 {mtf['오즈비']:.1f}배",
                     "기각",
                     "반복 질의는 응답 실패의 결과로 확인됨"))
    scf = F.get("자기검열", {})
    if scf:
        rows.append(("차단은 해당 질의만 제한함",
                     f"차단 후 판단성 질의 비중 {pct(scf.get('차단전_판단성비중'))} "
                     f"→ {pct(scf.get('차단후_판단성비중'))}",
                     "기각",
                     "유사 질의 전반이 감소하나 크기는 확정 불가"))
    if rows:
        A(T(pd.DataFrame(rows, columns=["가설", "확인 결과", "판정", "비고"])))

    A("<h3>Ⅲ-7. 잠재 정보 요구 분석</h3>")
    A("<p>응답이 성공한 경우에도 동일 대상에 대해 추가 질의가 이어지는 사례를 "
      "분석함. 후속 질의는 최초 응답이 충족하지 못한 정보 요구로 해석됨.</p>")
    if "need" in D and len(D["need"]):
        nd = D["need"].copy()
        keep = [c for c in ["표면 질문", "이어서 묻는 것", "건수"] if c in nd.columns]
        A(T(nd[keep].head(10).rename(columns={
            "표면 질문": "질의 유형", "이어서 묻는 것": "후속 질의(비중, 평균 대비)"})))
    if "need_q" in D and len(D["need_q"]):
        nq = D["need_q"].copy()
        star = nq[nq["구분"].astype(str).str.startswith("★")] if "구분" in nq else nq
        if len(star):
            A("<h4>단회 응답으로 충족되지 않는 질의 유형</h4>")
            keep = [c for c in ["질문", "자기완결률", "되묻기율", "연쇄깊이"]
                    if c in star.columns]
            A(T(star[keep].head(8).rename(columns={"질문": "질의 유형"}),
                {"자기완결률": lambda v: pct(v), "되묻기율": lambda v: pct(v)}))
            note("해석", "반복 질의(실패)와는 구분되는 항목으로, 현행 실패 지표에는 "
                         "포착되지 않음. 응답 구성 개선 과제에 해당함.")

    A("<h3>Ⅲ-8. 지속 이용자 특성</h3>")
    acf = F.get("활성화", {})
    if acf:
        A(f"<p>가입 후 첫 주에 「{_html.escape(str(acf.get('후보','')))}」 조건을 "
          f"충족한 이용자의 30일 재방문율은 {pct(acf.get('충족시_잔존'))}로, "
          f"미충족 이용자 {pct(acf.get('미충족시_잔존'))} 대비 약 "
          f"{float(acf.get('리프트', 0)):.1f}배임. 해당 이용자는 전체의 "
          f"{pct(acf.get('충족비율'))}에 해당함"
          + fn("상관관계이며 인과관계는 별도 실험으로 확인 필요.") + ".</p>")
    if er:
        A("<h4>최초 질의 유형별 재방문율</h4>")
        A(T(er["상위"].rename(columns={"진입질문": "질의 유형", "잔존율": "재방문율"}),
            {"재방문율": lambda v: pct(v)}))
        A(T(er["하위"].rename(columns={"진입질문": "질의 유형", "잔존율": "재방문율"}),
            {"재방문율": lambda v: pct(v)}))
    flush_fn()

    # ── Ⅳ. 개선 방안
    A("<h2>Ⅳ. 개선 방안</h2>")
    A("<h3>Ⅳ-1. 우선 대상 선정</h3>")
    A(T(pd.DataFrame([
        ("시장·종목 탐색형", "감소", "중간", "높음", "우선 대상"),
        ("계좌·앱 업무형", "증가", "높음", "낮음", "현행 유지"),
        ("오류·상담형", "유지", "낮음", "낮음", "실패 감축 대상"),
    ], columns=["이용자 부류", "규모 추세", "성공률", "재방문율", "구분"])))
    note("선정 근거",
         "① 최초 질의가 탐색형인 이용자의 재방문율이 가장 높음 "
         "② 신규 유입에서 해당 비중의 감소 폭이 가장 큼 "
         "③ Ⅲ-8의 지속 이용 요인과 방향이 일치함")

    A("<h3>Ⅳ-2. 기 계획 과제 검토</h3>")
    A(T(pd.DataFrame([
        ("분류 모델 고도화", "유형 판별 실패·기능 미호출", "직접 대응",
         "전환 이전에 판별 실패의 실패 기록 조치가 선행되어야 효과 측정이 가능함"),
        ("별도 분류 모델 도입", "응답 속도", "부분 대응",
         "속도와 이탈의 관계는 미확인 상태이므로 성과 지표를 분류 정확도로 "
         "설정함이 타당함"),
        ("멀티턴 전환", "반복 질의·맥락 유실", "부분 대응",
         "응답 불가 원인(자료 부재·규정 제한)은 해소되지 않음"),
    ], columns=["과제", "대응 문제", "정합성", "검토 의견"])))
    note("미포함 사항",
         "다음 세 항목은 기 계획 과제에 포함되어 있지 않으며, 분류 정확도 "
         "개선으로 해소되지 않음. ① 거절 후 대체 정보 제공 ② 본인 계좌·자료 "
         "부재 대응 ③ 응답 구성 개선")

    A("<h3>Ⅳ-3. 개선 과제</h3>")
    rows = [("1", "질의 유형 판별 실패의 실패 기록 및 되묻기 경로 신설",
             "개발", "측정 기준선 확보"),
            ("2", "규정상 거절 시 대체 정보 제공 체계 수립", "준법·AI",
             "대화 종료 위험 최고 지점 해소"),
            ("3", "실패 집중 기능 개선(인증 처리·자료 조달·라우팅)", "부서별",
             "실패율 감축"),
            ("4", "응답 구성(템플릿) 개선", "AI·기획", "단회 충족률 제고"),
            ("5", "수요 위축 의심 기능 점검", "프로덕트", "잠재 수요 확인")]
    A(T(pd.DataFrame(rows, columns=["순위", "과제", "담당", "기대 효과"])))
    if pr is not None and len(pr):
        A("<h4>실패 집중 기능(3순위 상세)</h4>")
        A(T(pr, {"실패율": lambda v: pct(v), "질의량": lambda v: num(v)}))

    A("<h3>Ⅳ-4. 기대 효과</h3>")
    A(T(pd.DataFrame([
        ("확정", "판별 실패의 실패 기록",
         "지표상 성공률은 하락함. 정확한 기준선 확보가 목적임"),
        ("추정", "자료·기능·인증·거절 후속 처리 개선",
         "아래 표의 상한까지 개선 가능함(가정 기반 산출)"),
        ("실험 필요", "진입 질의 재배치·응답 구성 개선",
         "개선이 예상되나 A/B 검증 필요"),
    ], columns=["신뢰도", "과제", "내용"])))
    note("유의사항",
         "1순위 과제 시행 시 보고 성공률이 하락함. 이는 지표 악화가 아니라 "
         "측정 기준의 교정에 해당하며, 시행 이전에 관련 부서 간 합의가 필요함.")
    if "attr_fix" in D and len(D["attr_fix"]):
        fx2 = D["attr_fix"].copy()
        keep = [c for c in ["시나리오", "치환건수", "무마찰해결률", "Δ무마찰"]
                if c in fx2.columns]
        A(T(fx2[keep].rename(columns={
            "시나리오": "개선 대상", "치환건수": "해당 질의 수",
            "무마찰해결률": "단회 해결률", "Δ무마찰": "개선폭"}),
            {"단회 해결률": lambda v: pct(v),
             "개선폭": lambda v: f"{float(v)*100:+.1f}%p"}))
    flush_fn()

    # ── Ⅴ. 협조 요청
    if "northstar" in D and len(D["northstar"]):
        ns3 = D["northstar"].copy(); c0 = ns3.columns[0]
        ns3 = ns3.rename(columns={
            c0: "주차", "무마찰해결률": "1회 문답 해결률",
            "세션해결률": "최종 해결률", "폴백세션비율": "인식 실패 포함 대화",
            "이탈세션률": "미해결률", "NorthStar(보조)": "주간 성공 이용자(보조)"})
        A("<h4>추적 지표 추이</h4>")
        A(f"<p>개선 전후 비교 기준선. '주간 성공 이용자'는 천장 효과로 개선 추적이 "
          f"곤란하므로{fn('복수 대화 시 대부분 충족되어 변화 감지 불가')} "
          f"<b>1회 문답 해결률</b>을 주 지표로 전환할 것을 제안함.</p>")
        keep = [c for c in ["주차", "진입사용자", "1회 문답 해결률", "최종 해결률",
                            "미해결률", "인식 실패 포함 대화",
                            "주간 성공 이용자(보조)"] if c in ns3.columns]
        A(T(ns3[keep].tail(10), {c: (lambda v: pct(v)) for c in
                                 ["1회 문답 해결률", "최종 해결률", "미해결률",
                                  "인식 실패 포함 대화", "주간 성공 이용자(보조)"]}))
    flush_fn()

    A("<h2>Ⅴ. 협조 요청</h2>")
    A(T(pd.DataFrame([
        ("개발", "유형 판별 실패의 실패 기록 및 되묻기 경로 신설", "착수 여부"),
        ("준법", "거절 시 제공 가능한 대체 정보 범위 확정", "범위 승인"),
        ("프로덕트", "추적 지표를 재방문율에서 단회 해결률로 전환", "지표 전환 승인"),
        ("기획", "기 계획 과제에 거절 후속 처리·응답 구성 개선 포함 여부",
         "로드맵 반영"),
        ("데이터", "답변 가능성 라벨·질의 대상·진입 경로 기록 추가", "작업 반영"),
    ], columns=["대상 부서", "요청 내용", "필요 결정"])))

    # ── 별첨
    A("<h2>별첨 1. 지표 정의</h2>")
    A(T(pd.DataFrame([
        ("단회 해결률", "반복 질의 없이 한 번에 목적이 달성된 대화의 비율"),
        ("세션 해결률", "소요 횟수와 무관하게 목적이 달성된 대화의 비율"),
        ("재방문율", "가입 시점 기준 30일 경과 후 재이용한 이용자 비율"),
        ("종료 위험", "해당 질의 이후 대화가 종료될 확률의 전체 평균 대비 배수"),
        ("기능 포기", "특정 기능에서 실패한 이용자가 이후 세션에서 해당 기능을 "
                      "재시도하지 않는 현상"),
        ("자기완결률", "응답 성공 후 동일 대상에 대한 추가 질의가 발생하지 않은 비율"),
        ("유형 판별 실패", "질의 유형 분류에 실패하여 기타로 처리된 건"),
    ], columns=["지표", "정의"])))

    A("<h2>별첨 2. 데이터 제약 사항</h2>")
    if "bands" in D and len(D["bands"]):
        A("<h3>구간으로만 제시 가능한 지표</h3>")
        bd = D["bands"].copy()
        _isrt = bd["지표"].astype(str).str.contains("위험비|배수")
        for c in ("하한", "상한", "폭"):
            bd[c] = [f"{float(v):.2f}" if r else pct(v)
                     for v, r in zip(bd[c], _isrt)]
        A(T(bd))
    if "gaps" in D and len(D["gaps"]):
        g = D["gaps"]
        need2 = g[g["상태"].astype(str).eq("결핍")]
        if len(need2):
            A("<h3>미확보 데이터 및 확보 시 가능한 분석</h3>")
            cols = [c for c in ["질문", "필요", "확보시", "난이도", "담당", "일정"]
                    if c in need2.columns]
            A(T(need2[cols].rename(columns={
                "질문": "현재 답변 불가한 사항", "필요": "필요 데이터",
                "확보시": "확보 시 가능한 분석"})))
    if F.get("결핍", {}).get("불능비율") is not None:
        A(f"<p>설계된 분석 항목 중 {pct(F['결핍']['불능비율'])}가 데이터 제약으로 "
          "결론 도출에 이르지 못함.</p>")
    mj = F.get("_오판사례")
    if mj:
        A("<h3>데이터 제약으로 인한 해석 변경 사례</h3>")
        A(T(pd.DataFrame(mj).rename(columns={
            "처음결론": "초기 해석", "보완후": "보완 후", "부족했던것": "제약 사항"})))

    A("<h2>별첨 3. 가설 검증 상세</h2>")
    for key, label in [("차단생존", "규정상 차단과 이탈의 관계"),
                       ("지연", "응답 지연과 이탈의 관계"),
                       ("자기검열", "차단이 질의 행태에 미치는 영향"),
                       ("멀티턴", "반복 질의의 발생 원인")]:
        v = F.get(key)
        if not v:
            continue
        A(f"<h3>{label}</h3>")
        A(T(pd.DataFrame([(k, str(val)) for k, val in v.items()
                          if not isinstance(val, (dict, list))],
                         columns=["항목", "값"])))
    if "lat_path" in D:
        A("<h3>응답 경로별 소요 시간 및 결과</h3>")
        A(T(D["lat_path"], {"다음턴발생률": lambda v: pct(v),
                            "되묻기율": lambda v: pct(v),
                            "세션해결률": lambda v: pct(v)}))

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTS 챗봇 서비스 이용 실태 분석 결과</title>
<style>{CSS_FORMAL}</style></head><body><div class="wrap">
{''.join(P)}
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out", help="run_analysis 산출 폴더")
    ap.add_argument("--report", default=None, help="저장 경로(.html)")
    ap.add_argument("--period", default="", help="분석 기간 표기")
    ap.add_argument("--style", default="formal", choices=["formal", "plain"],
                    help="formal: 부서 간 공식 배포용 / plain: 실무 논의용")
    ap.add_argument("--docno", default="", help="문서번호")
    ap.add_argument("--dept", default="", help="작성부서")
    ap.add_argument("--scope", default="", help="배포범위")
    args = ap.parse_args()

    od = Path(args.out)
    D = load(od)
    print(f"읽은 파일 {len(D)}/{len(FILES)}개")
    missing = [f for k, f in FILES.items() if k not in D]
    if missing:
        print(f"  없음: {', '.join(missing[:8])}"
              + (f" 외 {len(missing)-8}개" if len(missing) > 8 else ""))
    if not D:
        print("산출물이 없습니다. run_analysis.py 를 먼저 실행하십시오.")
        return 1

    meta = {"now": _dt.datetime.now().strftime("%Y-%m-%d"),
            "period": args.period or "run_analysis 설정 기간",
            "docno": args.docno, "dept": args.dept, "scope": args.scope}
    F = load_findings(od)
    _cfgp = od / "gaps_config.json"
    if _cfgp.exists():
        import json as _j
        try:
            _cfg = _j.loads(_cfgp.read_text(encoding="utf-8"))
            _mj = [r for r in _cfg.get("오판사례", [])
                   if any(str(v).strip() for v in r.values())]
            if _mj:
                F["_오판사례"] = _mj
        except Exception:
            pass
    print(f"분석 요약 {len(F)}항목" if F
          else "⚠ findings.json 없음 — 7장 가설 검증이 비게 됩니다. run_analysis 재실행 필요")
    html = (build_formal(D, meta, F) if args.style == "formal"
            else build(D, meta, F))
    default = ("report_formal.html" if args.style == "formal"
               else "report_plain.html")
    rp = Path(args.report) if args.report else od / default
    rp.write_text(html, encoding="utf-8")
    print(f"→ {rp.resolve()}  ({args.style})")
    print("  브라우저로 열고 Ctrl+P 로 PDF 저장, 또는 Word 에서 직접 열기")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())