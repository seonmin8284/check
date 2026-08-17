"""
전체 파이프라인 실행.

    python run_analysis.py --synth                 # 합성 데이터로 검증 (참값 대조)
    python run_analysis.py --data ./data --out ./out

실데이터 사용 시 ./data 에 아래 파일이 있어야 합니다.
    queries.parquet  orders.parquet  app_views.parquet  app_sessions.parquet
컬럼명이 다르면 mts_analysis/schema.py 의 COLUMN_ALIASES 만 수정하십시오.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="statsmodels")
import html as _html
import io
import re as _re
import sys
from pathlib import Path

import pandas as pd

from mts_analysis import coverage as cov
from mts_analysis import schema as sch
from mts_analysis import track_a as A
from mts_analysis import track_b as B
from mts_analysis import track_c as C
from mts_analysis import turns as T
from mts_analysis import segments as S
from mts_analysis import quality as Q
from mts_analysis import annotation as AN
from mts_analysis import sessions as SS
from mts_analysis import retention as RT
from mts_analysis import mechanism as MX
from mts_analysis import relevance as RV
from mts_analysis import guards as GD
from mts_analysis import acquisition as AQ
from mts_analysis import protector as PR
from mts_analysis import gaps as GP
from mts_analysis import latency as LT
from mts_analysis import attribution as AT
from mts_analysis import needs as ND
from mts_analysis import samples as SM

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


class Tee:
    """콘솔 출력을 그대로 유지하면서 사본을 버퍼에 모은다."""

    def __init__(self, stream):
        self.stream = stream
        self.buf = io.StringIO()

    def write(self, s):
        self.stream.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def to_markdown(text: str, meta: dict) -> str:
    """
    콘솔 출력을 마크다운으로 변환한다.
    ==== 구분선은 h2, ── 구분선은 h3, 나머지 본문은 고정폭 블록으로 감싼다.
    """
    lines = text.splitlines()
    out = ["# MTS 챗봇 분석 리포트", ""]
    out += [f"- 생성: {meta['now']}",
            f"- 입력: `{meta['src']}`",
            f"- 산출: `{meta['out']}`",
            f"- 기준일: 중단 {meta['outage']} / Protector {meta['protector']} "
            f"/ 종료 {meta['end']}", "", "---", ""]

    body, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if set(ln.strip()) == {"="} and ln.strip():
            title = lines[i + 1].strip() if i + 1 < len(lines) else ""
            body.append(("h2", title))
            i += 3 if (i + 2 < len(lines) and set(lines[i + 2].strip()) == {"="}) else 2
            continue
        if ln.startswith("── "):
            body.append(("h3", ln.replace("─", "").strip()))
            i += 1
            continue
        body.append(("t", ln))
        i += 1

    block: list[str] = []

    def flush_block():
        nonlocal block
        while block and not block[0].strip():
            block.pop(0)
        while block and not block[-1].strip():
            block.pop()
        if block:
            out.extend(["```text", *block, "```", ""])
        block = []

    for kind, val in body:
        if kind == "h2":
            flush_block()
            out += [f"## {val}", ""]
        elif kind == "h3":
            flush_block()
            out += [f"### {val}", ""]
        else:
            block.append(val)
    flush_block()
    return "\n".join(out)


_HTML_CSS = """
:root{--fg:#1a1a1a;--muted:#666;--line:#e0e0e0;--bg:#fff;--code:#f7f7f8;
      --warn:#b45309;--warnbg:#fffbeb;--ok:#166534;--okbg:#f0fdf4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font-family:"Malgun Gothic","맑은 고딕","Apple SD Gothic Neo",sans-serif;
     font-size:15px;line-height:1.65}
.wrap{max-width:1100px;margin:0 auto;padding:36px 28px 80px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.3px}
h2{font-size:20px;margin:38px 0 10px;padding-top:14px;border-top:2px solid var(--fg)}
h3{font-size:16px;margin:24px 0 8px;color:#333}
.meta{color:var(--muted);font-size:13px;margin-bottom:22px}
.meta div{margin:2px 0}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;
    padding:12px 14px;overflow-x:auto;margin:8px 0 14px;
    font-family:"D2Coding","Consolas","Courier New",monospace;
    font-size:12.5px;line-height:1.5;white-space:pre}
.callout{border-left:3px solid var(--warn);background:var(--warnbg);
         padding:9px 13px;margin:10px 0;border-radius:0 4px 4px 0;font-size:13.5px}
.callout.ok{border-color:var(--ok);background:var(--okbg)}
nav{background:#fafafa;border:1px solid var(--line);border-radius:8px;
    padding:14px 18px;margin:0 0 30px}
nav b{font-size:13px;color:var(--muted);display:block;margin-bottom:8px}
nav ol{margin:0;padding-left:20px}
nav li{margin:3px 0;font-size:14px}
nav a{color:#0b57d0;text-decoration:none}
nav a:hover{text-decoration:underline}
@media print{
  body{font-size:11px}.wrap{max-width:none;padding:0}
  nav{page-break-after:always}
  h2{page-break-before:auto;page-break-after:avoid}
  pre{font-size:9.5px;page-break-inside:avoid;background:#fff}
}
"""


def to_html(text: str, meta: dict) -> str:
    """
    콘솔 출력을 자체 완결형 HTML 로 변환한다.
    외부 CSS·폰트·스크립트를 전혀 참조하지 않으므로 폐쇄망에서도 그대로 열린다.
    """
    lines, body, i = text.splitlines(), [], 0
    while i < len(lines):
        ln = lines[i]
        if set(ln.strip()) == {"="} and ln.strip():
            title = lines[i + 1].strip() if i + 1 < len(lines) else ""
            body.append(("h2", title))
            i += 3 if (i + 2 < len(lines) and set(lines[i + 2].strip()) == {"="}) else 2
            continue
        if ln.startswith("── "):
            body.append(("h3", ln.replace("─", "").strip()))
            i += 1
            continue
        body.append(("t", ln))
        i += 1

    parts, toc, block, sec = [], [], [], 0

    def flush():
        nonlocal block
        while block and not block[0].strip():
            block.pop(0)
        while block and not block[-1].strip():
            block.pop()
        if not block:
            block = []
            return
        # ⚠ / ❗ / ✅ 로 시작하는 줄은 콜아웃으로 분리
        buf = []
        for ln_ in block:
            st = ln_.strip()
            if st.startswith(("⚠", "❗", "✅", "→", "▸", "ℹ")):
                if buf:
                    parts.append("<pre>" + _html.escape("\n".join(buf)) + "</pre>")
                    buf = []
                cls = "callout ok" if st.startswith("✅") else "callout"
                parts.append(f'<div class="{cls}">{_html.escape(st)}</div>')
            else:
                buf.append(ln_)
        if buf:
            parts.append("<pre>" + _html.escape("\n".join(buf)) + "</pre>")
        block = []

    for kind, val in body:
        if kind == "h2":
            flush()
            sec += 1
            aid = f"s{sec}"
            toc.append(f'<li><a href="#{aid}">{_html.escape(val)}</a></li>')
            parts.append(f'<h2 id="{aid}">{_html.escape(val)}</h2>')
        elif kind == "h3":
            flush()
            parts.append(f"<h3>{_html.escape(val)}</h3>")
        else:
            block.append(val)
    flush()

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTS 챗봇 분석 리포트 {_html.escape(meta['now'])}</title>
<style>{_HTML_CSS}</style></head><body><div class="wrap">
<h1>MTS 챗봇 분석 리포트</h1>
<div class="meta">
<div>생성 {_html.escape(meta['now'])}</div>
<div>입력 {_html.escape(str(meta['src']))}</div>
<div>기준일 · 소스 중단 {_html.escape(str(meta['outage']))}
 · Protector 도입 {_html.escape(str(meta['protector']))}
 · 관측 종료 {_html.escape(str(meta['end']))}</div>
</div>
<nav><b>목차</b><ol>{''.join(toc)}</ol></nav>
{''.join(parts)}
</div></body></html>"""


# ═══════════════════ 출력 재배열 — 보고서 장 순서와 동일하게
#
# 분석은 의존 순서대로 계산하되, 출력은 business_report 의 장 순서로 모아 낸다.
# 데이터는 하나도 버리지 않는다. 매핑되지 않은 항목은 '부록'으로 모인다.

REPORT_CHAPTERS = [
    ("C1", "1. 개요 · 데이터 품질"),
    ("C2", "2. 측정 신뢰도 — 숫자를 믿기 전에"),
    ("C3", "3. 이용자 구성"),
    ("C4", "4. 시간대별 이용 패턴"),
    ("C5", "5. 이탈 유형별 분석"),
    ("C6", "6. 이용 흐름상 중단 지점"),
    ("C7", "7. 기존 가설 검증 결과"),
    ("C75", "7.5 잠재 정보 요구 분석"),
    ("C8", "8. 지속 이용자 특성"),
    ("C9", "9. 우선 대상 선정"),
    ("C11", "11. 개선 과제"),
    ("C12", "12. 기대 효과"),
    ("C13", "13. 데이터 제약 · 협조 요청"),
    ("CX", "부록. 그 밖의 분석"),
    ("CZ", "산출물"),
]

# 원 분석 대단원 → 보고서 장. 서브 이전의 도입 문구가 엉뚱한 장에 붙지 않게 한다.
H_TO_CHAPTER = {
    "0. 스키마": "C1", "0.5": "C1", "1. 커버리지": "C6", "2. A트랙": "C4",
    "3. B트랙": "C5", "4. C트랙": "C11", "5. 사용자 세그먼트": "C3",
    "6. 멀티턴": "C75", "7. 회수": "C11", "8. 세션 단위": "C5",
    "9. 메커니즘": "C7", "10. 획득": "C3", "11. Protector": "C6",
    "12. 어떤 문제": "C5", "13. 사용자가": "C75", "14. 지금": "C13",
    "완료": "CZ",
}

# 서브 섹션 제목의 일부 → 보고서 장. 위에서부터 먼저 맞는 것을 쓴다.
SUB_TO_CHAPTER = [
    ("식별 가능성", "C1"), ("secondary 공기", "C1"),
    ("OTH 폴백", "C2"), ("의도분류 오분류", "C2"), ("질의 왜곡", "C2"),
    ("관련성 프록시", "C2"), ("슬롯 복원율", "C2"), ("환각 위험군", "C2"),
    ("차단 정의 교차", "C2"),
    ("신규 vs 재방문", "C3"), ("세그먼트 도출", "C3"),
    ("시기별 세그먼트", "C3"), ("첫 질의 구성", "C3"),
    ("시장 세션", "C4"),
    ("세션 결과 분포", "C5"), ("의도 포기율", "C5"),
    ("코호트별 리텐션", "C5"), ("평탄화", "C5"),
    ("구성 변화 vs 행동", "C5"), ("잔존 손실", "C5"),
    ("Kaplan-Meier", "C5"), ("자연 사용 주기", "C5"),
    ("대화 중단 귀속", "C5"), ("노출-반응", "C5"),
    ("문제 개수와 이탈", "C5"), ("통합 — 근거 등급", "C5"),
    ("이탈 ≠ 고객", "C5"),
    ("여정 단계별", "C6"), ("실패코드", "C6"), ("여정 전이 행렬", "C6"),
    ("마지막 이탈 지점", "C6"), ("직전 턴 결과", "C6"),
    ("불만·이관", "C6"), ("실패 직후 단계 전이", "C6"),
    ("차단 후 회수 실패", "C6"), ("프로텍터 영향", "C6"),
    ("층1", "C6"), ("층2", "C6"), ("층3", "C6"),
    ("차단 유형 분포", "C6"), ("매칭", "C6"), ("정책 일관성", "C6"),
    ("생존분석", "C7"), ("응답 지연", "C7"), ("의존도 충격", "C7"),
    ("자기검열", "C7"), ("멀티턴은 싱글턴", "C7"),
    ("DiD", "C7"), ("평행추세", "C7"), ("믹스 고정", "C7"),
    ("실질 성공률", "C7"), ("대체 가능성", "C7"),
    ("이어서 묻는 것", "C75"), ("자기완결률", "C75"),
    ("되묻기 × 연쇄", "C75"), ("맥락에 따라", "C75"), ("충족률", "C75"),
    ("응답 구조", "C75"), ("형식 재요청", "C75"),
    ("Activation", "C8"), ("진입 질문별 잔존율", "C8"),
    ("신규 진입 질문", "C8"), ("리텐션 유발", "C8"),
    ("1회성 vs 재방문", "C8"),
    ("세그먼트 × 핵심지표", "C9"), ("수요-공급", "C9"),
    ("비용-편익", "C9"), ("재배치 시뮬레이션", "C9"),
    ("개선 우선순위", "C11"), ("노력 비용", "C11"),
    ("의도별 출력 양·질", "C11"), ("회수 잠재량", "C11"),
    ("증분 효과", "C11"), ("전환 창", "C11"),
    ("회복 상한", "C12"), ("단독 해결 효과", "C12"),
    ("결핍 매트릭스", "C13"), ("분석 불능", "C13"),
    ("불확실 구간", "C13"), ("결측 규모", "C13"),
    ("사람이 채워야", "C13"), ("원본 표본 일괄", "CZ"),
]

_BUF: dict[str, list[str]] = {}
_CUR = ["CX"]
_ORIG = [""]


class _Bucket:
    """현재 장 버퍼로 출력을 모으는 writer."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        _BUF.setdefault(_CUR[0], []).append(s)
        return len(s)

    def flush(self):
        pass

    def __getattr__(self, n):
        return getattr(self.stream, n)


def _chapter_of(title: str) -> str | None:
    for key, ch in SUB_TO_CHAPTER:
        if key in title:
            return ch
    return None


def h(t: str) -> None:
    _ORIG[0] = t
    ch = next((v for k, v in H_TO_CHAPTER.items() if t.startswith(k)),
              _chapter_of(t))
    if ch:
        _CUR[0] = ch
    print(f"\n▪ 원 분석 [{t}]")


def sub(t: str) -> None:
    ch = _chapter_of(t)
    if ch:
        _CUR[0] = ch
    tag = f"  〈원 {_ORIG[0].split('.')[0]}장〉" if _ORIG[0] else ""
    print(f"\n── {t} " + "─" * max(0, 66 - len(t)) + tag)


def show_ex(df, title: str = "원본 예시", n: int = 3) -> None:
    """측정 지점 바로 아래에 실제 데이터 몇 건을 붙인다."""
    if df is None or len(df) == 0 or "안내" in getattr(df, "columns", []):
        return
    print(f"\n  ▸ {title}")
    txt = df.head(n).to_string(index=False, max_colwidth=64)
    print("    " + txt.replace("\n", "\n    "))


def emit_ordered(stream) -> None:
    """버퍼를 보고서 장 순서로 실제 출력에 흘려보낸다. 데이터는 버리지 않는다."""
    used = set()
    for key, title in REPORT_CHAPTERS:
        body = "".join(_BUF.get(key, [])).strip("\n")
        used.add(key)
        if not body:
            continue
        stream.write(f"\n{'='*78}\n  {title}\n{'='*78}\n")
        stream.write(body + "\n")
    leftover = [k for k in _BUF if k not in used]
    if leftover:
        stream.write(f"\n{'='*78}\n  부록. 미분류 항목\n{'='*78}\n")
        for k in leftover:
            stream.write("".join(_BUF[k]))


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ["queries", "orders", "app_views", "app_sessions"]:
        for ext in (".pkl", ".parquet", ".csv"):
            p = data_dir / f"{name}{ext}"
            if p.exists():
                break
        else:
            raise FileNotFoundError(f"{name} 파일 없음: {data_dir}")
        if p.suffix == ".pkl":
            df = pd.read_pickle(p)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, sep=None, engine="python")
        out[name] = sch.normalize(df, name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true", help="합성 데이터로 검증")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out", type=str, default="./out")
    ap.add_argument("--outage", type=str, default="2026-04-15", help="소스 중단일")
    ap.add_argument("--protector", type=str, default="2026-03-01", help="Protector 도입일")
    ap.add_argument("--end", type=str, default="2026-06-30", help="관측 종료일")
    ap.add_argument("--start", type=str, default=None,
                    help="분석 시작일. 파일럿·내부테스트 구간을 잘라냅니다")
    ap.add_argument("--oth-codes", default=None,
                    help="분류 실패 코드(쉼표 구분). 예: OTHER,OTH")
    ap.add_argument("--fallback-tools", default=None,
                    help="폴백 함수명 부분문자열(쉼표 구분). 예: news_and_work")
    ap.add_argument("--report", default="report.html",
                    help="콘솔 출력 저장 파일 (.html / .md / .txt). off 로 끄기. "
                         "쉼표로 여러 개 지정 가능: report.html,report.md")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    FIND: dict = {}
    tee = None
    if args.report and args.report.lower() != "off":
        tee = Tee(sys.stdout)
        sys.stdout = tee
    # 계산은 의존 순서대로, 출력은 보고서 장 순서로. 여기서부터 버퍼에 모은다.
    _real_out = sys.stdout
    sys.stdout = _Bucket(_real_out)

    truth = None
    if args.synth:
        from mts_analysis import synth
        d = synth.generate()
        truth = d.pop("users_truth")
        d = {k: sch.normalize(v, k) for k, v in d.items()}
        d["queries"] = sch.ensure_optional(d["queries"])
    elif args.data:
        d = load(Path(args.data))
        d["queries"] = sch.ensure_optional(d["queries"])
    else:
        ap.error("--synth 또는 --data 중 하나가 필요합니다.")

    if args.start:
        START = pd.Timestamp(args.start)
        n0 = len(d["queries"])
        d["queries"] = d["queries"][d["queries"]["ts"] >= START]
        for k in ("orders", "app_views"):
            if len(d[k]) and "ts" in d[k].columns:
                d[k] = d[k][pd.to_datetime(d[k]["ts"]) >= START]
        if len(d["app_sessions"]) and "date" in d["app_sessions"].columns:
            d["app_sessions"] = d["app_sessions"][
                pd.to_datetime(d["app_sessions"]["date"]) >= START]
        print(f"\n▸ 분석 시작일 {START:%Y-%m-%d} 적용 — "
              f"{n0 - len(d['queries']):,}건 제외 (잔여 {len(d['queries']):,}건)")

    q, orders = d["queries"], d["orders"]
    views, sessions = d["app_views"], d["app_sessions"]
    OUTAGE = pd.Timestamp(args.outage)
    PROT = pd.Timestamp(args.protector)
    END = pd.Timestamp(args.end)

    # ---------------------------------------------------------------- 0. 검증
    h("0. 스키마 검증 및 데이터 품질")
    for w in sch.validate_queries(q):
        print("  •", w)
    print(f"\n  질의 {len(q):,} · 사용자 {q['user_id'].nunique():,} "
          f"· 세션 {q['session_id'].nunique():,}")
    print(f"  주문 {len(orders):,} · 비챗봇조회 {len(views):,} · 앱세션 {len(sessions):,}")

    sub("식별 가능성 점검 — 돌아가는 것과 해석할 수 있는 것은 다릅니다")
    gtab, gchk = GD.run_all(q, policy_date=PROT, event_date=OUTAGE, end_date=END)
    print(gtab.to_string(index=False))
    so = gchk.get("서비스 오픈 시점", {})
    if so and not so["통과"] and not args.start:
        print(f"\n  ⛔ {so['사유']}")
        print("     자르지 않으면 코호트 비교·믹스 반사실 기준시기·shift-share "
              "전반부가 모두 오염됩니다.")

    failed = [k for k, v in gchk.items() if not v["통과"]]
    if failed:
        print(f"\n  ⚠ 실패 {len(failed)}건 — 해당 분석 결과는 아래 표시와 함께 "
              "해석 제한이 붙습니다")

    gap = abs((OUTAGE - PROT).days)
    print(f"\n  ⚠ 교란 점검: 소스 중단일 {OUTAGE:%Y-%m-%d} / Protector 도입일 "
          f"{PROT:%Y-%m-%d} (간격 {gap}일)")
    if gap < 14:
        print("    → 두 시점이 근접. 효과가 교란되어 분리 식별이 어려움. "
              "의도군 DiD로 우회하되 해석에 주의.")

    # ------------------------------------------------------- 0.5 어노테이션 품질
    h("0.5 어노테이션 품질 — 택소노미 진단")

    rv = AN.review_summary(q)
    if "안내" in rv:
        print("  ", rv["안내"])
    else:
        print(f"  needs_review {rv['검토건수']:,}건 ({rv['전체_검토율']:.1%})")
        print(rv["단계별"].to_string())
        print("  [검토율 상위 의도 — 택소노미 경계 문제 후보]")
        print(rv["의도별_상위"].head(6).to_string())

    cf = AN.confidence_profile(q)
    if "안내" in cf:
        print("\n  ", cf["안내"])
    else:
        print(f"\n  평균 확신 {cf['평균확신']:.3f} · "
              f"저확신(<0.7) {cf['저확신율(<0.7)']:.1%}")
        print("  facet별 평균확신:", cf["facet별_평균확신"])
        if cf["저확신여부별_실패율"]:
            print("  저확신 여부별 실패율:", cf["저확신여부별_실패율"])
            print("  → 저확신 구간 실패율이 높으면 어노테이터가 어려워한 발화와 "
                  "모델이 어려워한 발화가 같다는 뜻")
        print("  [저확신 상위 의도]")
        print(cf["의도별_상위"].head(6).to_string())

    co = AN.secondary_cooccurrence(q)
    if "안내" not in co.columns:
        sub("secondary 공기 — 통합·재배치 후보")
        print(co.head(10).to_string(index=False))
        co.to_csv(outdir / "secondary_cooccurrence.csv",
                  index=False, encoding="utf-8-sig")

    # 본 집계에서 검토 대기 건을 분리한다 (섞으면 실패율이 오염됨)
    q_all = q
    q, q_review = AN.split_review(q)
    if len(q_review):
        print(f"\n  ▸ 본 집계에서 검토 대기 {len(q_review):,}건 제외 "
              f"(잔여 {len(q):,}건). 아래 모든 결과는 검토 완료분 기준입니다.")

    # ---------------------------------------------------------------- 1. 커버리지
    h("1. 커버리지 감사 · 실패 귀속")
    ob = "overblock" if q.get("overblock") is not None and q["overblock"].notna().any() else None
    _ev = cov.tool_evidence(q)
    print(f"  기능 존재 증거: {len(_ev['has_tool'])}개 의도에서 툴 호출 이력 확인 "
          "— 미구현(T1)과 라우팅 실패(T2)를 이 기준으로 분리합니다")
    q = cov.derive_fail_codes(q, outage_date=OUTAGE, overblock_flag=ob,
                              evidence=_ev)
    q = cov.flag_c3(q)

    sub("L1 여정 단계별 (퍼널 절단 지점)")
    print(cov.stage_funnel(q).round(3).to_string())

    show_ex(SM.by_category(q, "fail_code", 1), "실패 원인별 실제 사례 (판정 검증용)", 8)
    sub("의도 × 실패코드 (상위 12 의도)")
    cm = cov.coverage_matrix(q)
    print(cm.head(12).round(2).to_string())
    cm.to_csv(outdir / "coverage_matrix.csv", encoding="utf-8-sig")

    sub("개선 우선순위 (질의량 × 실패율 ÷ 해결비용)")
    pr = cov.priority(q, top=12)
    print(pr[["l2_intent", "단계", "fail_code", "질의량", "실패율",
              "담당", "처방", "우선순위"]].round(3).to_string(index=False))
    pr.to_csv(outdir / "priority.csv", index=False, encoding="utf-8-sig")

    sub("차단 후 회수 실패 (C3)")
    n_blk = int(q["outcome"].eq("blocked").sum())
    n_c3 = int(q["fail_code"].eq("C3").sum())
    FIND["차단"] = {"차단건수": n_blk, "회수실패": n_c3,
                    "회수실패율": (n_c3 / n_blk) if n_blk else None}
    if n_blk:
        print(f"  P3 차단 {n_blk:,}건 중 회수 실패 {n_c3:,}건 ({n_c3/n_blk:.1%})")
        print("  → 이 비율이 높으면 이탈 동인은 차단 자체가 아니라 '대체 재료 부재'.")
    else:
        print("  차단 건 없음")

    # ---------------------------------------------------------------- 2. A트랙
    h("2. A트랙 — 집계 분석")

    sub("시장 세션 × 여정 단계 구성비")
    print(A.session_profile(q).round(3).to_string())

    A.session_profile(q).to_csv(outdir / "session_profile.csv", encoding="utf-8-sig")
    A.top_intent_by_session(q, k=5).to_csv(
        outdir / "session_top_intent.csv", encoding="utf-8-sig")

    sub("시장 세션별 상위 의도")
    print(A.top_intent_by_session(q, k=4).to_string())
    A.intraday_profile(q, "l2_intent").to_csv(
        outdir / "intraday_profile.csv", encoding="utf-8-sig")

    sub("믹스 고정 실패율 (심슨의 역설 방어)")
    base_lo = (OUTAGE - pd.Timedelta(days=28)).strftime("%Y-%m-%d")
    base_hi = OUTAGE.strftime("%Y-%m-%d")
    mx = A.mix_adjusted_rate(q, (base_lo, base_hi), freq="W")
    print(mx.round(4).to_string())
    if (mx["괴리"].abs() > .02).any():
        print("  ⚠ 조 실패율과 믹스고정 실패율의 괴리가 큼 — "
              "변화의 상당 부분이 의도 믹스 이동에서 옴. 조 실패율 단독 보고 금지.")
    mx.to_csv(outdir / "mix_adjusted.csv", encoding="utf-8-sig")

    sub("평행추세 검정 (이벤트 스터디)")
    es = A.event_study(q, OUTAGE, span=6)
    if "불가" in es.attrs:
        print("  ⚠", es.attrs["불가"])
    else:
        print(es.round(3).to_string(index=False))
        print("  판정:", A.parallel_trend_verdict(es))
        es.to_csv(outdir / "event_study.csv", index=False, encoding="utf-8-sig")

    sub("DiD — 소스 중단이 의존 의도군 실패율에 미친 효과")
    if not gchk["이벤트 창"]["통과"]:
        print("  ⛔ 사후 구간 부족 — 아래 추정치는 신뢰할 수 없습니다")
    dd = A.did(q, OUTAGE)
    if "불가" in dd:
        print("  ⚠", dd["불가"])
        dd = None
    if dd:
      print(f"  효과 {dd['효과(실패율 pp)']:+.4f} "
          f"(SE {dd['표준오차']:.4f}, p={dd['p값']:.2e})")
      print(f"  95% CI [{dd['95%CI'][0]:+.4f}, {dd['95%CI'][1]:+.4f}] "
            f"· 클러스터 {dd['클러스터수']}")
      if dd["경고"]:
          print("  ⚠", dd["경고"])
    if dd is not None and truth is not None:
        from mts_analysis.synth import TRUE_DID_EFFECT, TRUE_OTH_RATE
        # OTH 폴백이 실패의 일부를 성공으로 덮어쓰므로 관측 가능한 유효 효과는
        # 심어놓은 값보다 (1 - OTH비율) 만큼 작다.
        eff = TRUE_DID_EFFECT * (1 - TRUE_OTH_RATE)
        print(f"  [검증] 유효 참값 {eff:+.4f} "
              f"(심은 값 {TRUE_DID_EFFECT:+.3f} × 폴백 미덮임 {1-TRUE_OTH_RATE:.2f}) → "
              f"{'✅ CI 포함' if dd['95%CI'][0] <= eff <= dd['95%CI'][1] else '❌ 미포함'}")

    # ---------------------------------------------------------------- 3. B트랙
    h("3. B트랙 — 개인 패널 분석")

    sub("구성 변화 vs 행동 변화 (shift-share)")
    if not gchk["이벤트 창"]["통과"]:
        print("  ⛔", gchk["이벤트 창"]["사유"])
    # 창을 데이터 범위로 클램프한다. 사후가 짧으면 '잔존자'가 구조적으로
    # 극소수가 되어 구성 변화가 과대평가된다(창 길이 아티팩트).
    dmax = q["ts"].max()
    post_avail = (min(END, dmax) - OUTAGE).days
    if post_avail < 21:
        half = min((dmax - q["ts"].min()).days // 2, 120)
        cut = q["ts"].min() + pd.Timedelta(days=half)
        t0 = (q["ts"].min().strftime("%Y-%m-%d"), cut.strftime("%Y-%m-%d"))
        t1 = (cut.strftime("%Y-%m-%d"),
              (dmax + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        print(f"  ⛔ 중단일 기준 사후 구간이 {post_avail}일뿐 — "
              f"이벤트 대신 **관측 기간 전후반 비교**로 대체합니다 "
              f"({t0[0]}~{t0[1]} vs {t1[0]}~{t1[1]}).")
        print("     이 결과는 '소스 중단의 효과'가 아니라 '기간 중 구성 변화'입니다.")
    else:
        w = min(42, post_avail)
        t0 = ((OUTAGE - pd.Timedelta(days=w)).strftime("%Y-%m-%d"),
              OUTAGE.strftime("%Y-%m-%d"))
        t1 = (OUTAGE.strftime("%Y-%m-%d"),
              (OUTAGE + pd.Timedelta(days=w)).strftime("%Y-%m-%d"))
    ss = B.shift_share(q, t0, t1, target_stage="EVALUATE")
    print(f"  EVALUATE 비중 {ss.attrs['P0']:.3f} → {ss.attrs['P1']:.3f} "
          f"(Δ {ss.attrs['ΔP']:+.3f})")
    print(f"  잔존 {ss.attrs['n_stay']:,} · 이탈 {ss.attrs['n_exit']:,} "
          f"· 신규 {ss.attrs['n_entry']:,}")
    print(ss.round(4).to_string(index=False))
    print("  판정:", B.shift_share_verdict(ss))
    ss.to_csv(outdir / "shift_share.csv", index=False, encoding="utf-8-sig")

    sub("L1 여정 전이 행렬 (세션 내)")
    tm = B.transition_matrix(q)
    print(tm.round(3).to_string())
    mk = B.markov_order_test(q)
    print(f"  마르코프 검정: p={mk.get('p값', float('nan')):.3g} — {mk.get('판정','')}")
    tm.to_csv(outdir / "transition_matrix.csv", encoding="utf-8-sig")

    sub("생존분석 — 차단 효과 (시간가변 공변량)")
    if not gchk["정책 사전구간"]["통과"]:
        print("  ⛔", gchk["정책 사전구간"]["사유"])
        print("     아래 HR 은 '차단의 정책 효과'가 아니라 "
              "'판단성 질문을 한 사용자와 안 한 사용자의 위험 차이'입니다.")
    sp = B.build_survival_panel(q, END)
    cx_raw = B.cox_block_effect(sp, adjusted=False)
    cx = B.cox_block_effect(sp, adjusted=True)
    FIND["차단생존"] = {"조정HR": cx["HR"], "CI": list(cx["95%CI(HR)"]),
                        "p": cx["p값"], "미조정HR": cx_raw["HR"],
                        "식별가능": bool(gchk["정책 사전구간"]["통과"])}
    print(f"  미조정 HR {cx_raw['HR']:.3f} "
          f"[{cx_raw['95%CI(HR)'][0]:.3f}, {cx_raw['95%CI(HR)'][1]:.3f}]")
    print(f"  조정   HR {cx['HR']:.3f} "
          f"[{cx['95%CI(HR)'][0]:.3f}, {cx['95%CI(HR)'][1]:.3f}] p={cx['p값']:.2e}")
    if not gchk["정책 사전구간"]["통과"]:
        print("     ▸ 이 HR 은 '차단 경험자 vs 미경험자'의 위험비이며, "
              "두 집단은 애초에 다른 질문을 하는 사람들입니다.")
        print("     ▸ HR≈1 을 '차단은 무해하다'로 읽지 마십시오. "
              "정책 효과는 이 데이터로 식별 불가입니다.")
    print("  (차단은 시간가변 공변량 — 불멸시간 편향 방어. "
          "초기 활동량·의도다양성·투자지향도로 비무작위 배정 통제)")
    if abs(cx_raw["HR"] - cx["HR"]) > 0.1:
        print(f"  ⚠ 조정 전후 차이 {cx_raw['HR']-cx['HR']:+.2f} — "
              "차단의 비무작위 배정이 실제로 작동. 미조정 수치 보고 금지")
    if truth is not None:
        import numpy as np
        from mts_analysis.synth import TRUE_BLOCK_LOG_HR
        thr = float(np.exp(TRUE_BLOCK_LOG_HR))
        ok = cx["95%CI(HR)"][0] <= thr <= cx["95%CI(HR)"][1]
        print(f"  [검증] 참 HR {thr:.3f} → {'✅ CI 포함' if ok else '❌ 미포함'}")

    sub("코호트별 리텐션 (가입 주차 기준)")
    cr = RT.cohort_retention(q, END)
    if cr.empty:
        print("  ⚠ 코호트 표본 부족")
    else:
        print(cr.to_string())
        print("  ※ NaN 은 아직 관측 창이 도달하지 않은 시점입니다. 0 으로 읽지 마십시오.")
        cr.to_csv(outdir / "cohort_retention.csv", encoding="utf-8-sig")

    sub("리텐션 커브 평탄화 지점 (PMF 판단 근거)")
    pl = RT.plateau_point(q, END)
    if "안내" not in pl:
        FIND["평탄화"] = {k: v for k, v in pl.items() if k != "커브"}
    if "안내" in pl:
        print("  ", pl["안내"])
    else:
        pts = [d_ for d_ in [1, 7, 14, 30, 45, 60, 90] if d_ in pl["커브"].index]
        print("  커브: " + " · ".join(f"D{d_} {pl['커브'][d_]:.1%}" for d_ in pts))
        print(f"  대상 {pl['대상사용자']:,}명 / 관측 {pl['관측일수']}일")
        print("  판정:", pl["판정"])
        pl["커브"].to_csv(outdir / "retention_curve.csv", encoding="utf-8-sig")

    sub("Activation 역산 — 잔존을 가르는 첫 7일 행동")
    ac = RT.activation_candidates(q, END)
    if "안내" not in ac.columns and len(ac):
        FIND["활성화"] = {**ac.iloc[0].to_dict(),
                          "전체잔존율": ac.attrs.get("전체잔존율")}
    if "안내" in ac.columns:
        print("  ", ac["안내"].iloc[0])
    else:
        print(f"  전체 D30 잔존율 {ac.attrs['전체잔존율']:.1%}")
        print(ac.head(8).to_string(index=False))
        print("  ▸", RT.activation_verdict(ac))
        ac.to_csv(outdir / "activation_candidates.csv",
                  index=False, encoding="utf-8-sig")

    sub("Kaplan-Meier 생존확률")
    _km = B.km_curve(q, END).round(3)
    print(_km.to_string())
    _km.to_csv(outdir / "km_curve.csv", encoding="utf-8-sig")

    sub("자연 사용 주기")
    uc = B.usage_cycle(q)
    print(uc.round(2).to_string(index=False))
    uc.to_csv(outdir / "usage_cycle.csv", index=False, encoding="utf-8-sig")
    FIND["사용주기"] = {r["지표"]: float(r["일"]) for _, r in uc.iterrows()
                        if r["일"] == r["일"]}
    FIND["사용주기권고"] = B.retention_window_advice(uc)
    print("  권고:", B.retention_window_advice(uc))

    # ---------------------------------------------------------------- 4. C트랙
    h("4. C트랙 — 거래 조인 분석")

    sub("정보 → 주문 증분 효과 (개인×종목×일, 3원 FE)")
    vp = C.build_view_order_panel(q, orders, views, horizon_days=5)
    ie = C.incremental_order_effect(vp)
    if "불가" in ie:
        print("  ⚠", ie["불가"])
    else:
     print(f"  단순 차이      {ie['단순차이(pp)']:+.4f}")
     print(f"  FE 증분효과    {ie['증분효과(pp)']:+.4f} "
           f"(SE {ie['표준오차']:.4f}, p={ie['p값']:.2e})")
     print(f"  95% CI         [{ie['95%CI'][0]:+.4f}, {ie['95%CI'][1]:+.4f}]")
     print(f"  단순추정 편향  {ie['편향(단순-FE)']:+.4f}  "
           f"← 이만큼 부풀려짐. 보고 시 FE 추정치를 쓸 것")
     print(f"  관측 {ie['관측수']:,} · 클러스터(사용자) {ie['클러스터(사용자)']:,}")
    if "불가" not in ie and truth is not None:
        from mts_analysis.synth import TRUE_CHATBOT_ORDER_LIFT
        ok = ie["95%CI"][0] <= TRUE_CHATBOT_ORDER_LIFT <= ie["95%CI"][1]
        print(f"  [검증] 참값 {TRUE_CHATBOT_ORDER_LIFT:+.4f} → "
              f"{'✅ CI 포함' if ok else '△ 하한 성격(희석). 아래 창 민감도 참조'}")
    if len(vp):
        vp.to_csv(outdir / "view_order_panel.csv", index=False, encoding="utf-8-sig")

    sub("전환 창(horizon) 민감도 — 추정치가 평평해지는 지점을 채택")
    hs = C.horizon_sensitivity(q, orders, views)
    if "안내" in hs.columns:
        print("  ⚠", hs["안내"].iloc[0])
        flat = hs.iloc[0:0]
    else:
        print(hs.round(4).to_string(index=False))
        flat = hs[hs["직전대비변화"].abs() < 0.005]
    if len(flat):
        print(f"  → {int(flat['창(일)'].iloc[0])}일 이후 안정. 이 값을 본 추정치로 보고")
    print("  주의: 동일 종목 반복조회가 많으면 처치·대조 셀이 결과를 공유해 "
          "추정치가 하한 성격을 띱니다.")
    if "안내" not in hs.columns:
        hs.to_csv(outdir / "horizon_sensitivity.csv", index=False, encoding="utf-8-sig")

    sub("챗봇 이탈 ≠ 고객 이탈")
    tab = C.churn_2x2(q, sessions, END)
    if tab.empty:
        print("  ⚠ app_sessions 데이터 없음 — 챗봇 이탈과 고객 이탈을 구분할 수 없습니다")
    else:
        print(tab.to_string())
        print("  판정:", C.churn_2x2_verdict(tab))
        tab.to_csv(outdir / "churn_2x2.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------------- 5. 세그먼트
    h("5. 사용자 세그먼트 · 진입 · 이탈 · 리텐션 지점")

    sub("세그먼트 도출 (사전 정의 없이 의도 벡터에서)")
    seg = S.segment_users(q)
    if "오류" in seg:
        print("  ", seg["오류"])
    else:
        FIND["세그먼트"] = {"k": seg["k"], "실루엣": seg["실루엣"],
                            "규모": {str(i): float(v) for i, v in seg["규모"].items()}}
        print(f"  k={seg['k']} (실루엣 " +
              ", ".join(f"{k}:{v:.3f}" for k, v in seg["실루엣"].items()) + ")")
        print(seg["프로파일"].to_string())
        print("\n  규모:", ", ".join(f"{i} {v:.1%}" for i, v in seg["규모"].items()))
        st = S.segment_over_time(q, seg)
        if len(st):
            sub("시기별 세그먼트 구성비")
            print(st.to_string())
            st.to_csv(outdir / "segment_over_time.csv", encoding="utf-8-sig")

    sub("신규 진입 질문 → 30일 잔존율")
    eq = S.entry_questions(q, END)
    print(f"  전체 잔존율 {eq.attrs['전체잔존율']:.1%}")
    print("  [첫 경험 결과별]"); print(eq.attrs["첫경험_결과별"].to_string())
    print("  [상위 5]"); print(eq.head(5).to_string())
    print("  [하위 5]"); print(eq.tail(5).to_string())
    eq.to_csv(outdir / "entry_questions.csv", encoding="utf-8-sig")

    sub("마지막 이탈 지점 (종료 위험 lift)")
    ex = S.exit_points(q, END)
    print(f"  기저 종료율 {ex.attrs['기저_종료율']:.3f}")
    print("  [실패코드별]"); print(ex.attrs["실패코드별"].to_string())
    print("  [의도 상위 6]"); print(ex.head(6).to_string())
    ex.to_csv(outdir / "exit_points.csv", encoding="utf-8-sig")

    sub("리텐션 유발 지점 (개인 성향 차감 후 성공 효과)")
    rd = S.retention_drivers(q)
    if "안내" in rd.columns:
        print("  ", rd["안내"].iloc[0])
    else:
     print("  [상위 5]"); print(rd.head(5).to_string())
     print("  [하위 5]"); print(rd.tail(5).to_string())
     rd.to_csv(outdir / "retention_drivers.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------------- 6. 멀티턴
    h("6. 멀티턴 원인 · 형식 재요청 · 불만")

    fu = T.classify_followups(q)
    print(f"  REPEAT 판정 근거: {fu.attrs.get('repeat_basis', '?')}")
    if not gchk["슬롯 충전율"]["통과"]:
        print("  ⛔", gchk["슬롯 충전율"]["사유"])
    n_fu = int(fu["is_followup"].sum())
    print(f"  후속 턴 {n_fu:,} / 전체 {len(fu):,} ({n_fu/len(fu):.1%})")
    print("  유형 분포:", ", ".join(
        f"{k} {v:.1%}" for k, v in fu["turn_kind"].value_counts(normalize=True).items()))
    print("  → REPEAT/FORMAT 은 복구성(실패 신호), REFINE/PIVOT 은 정상 심화")

    show_ex(SM.by_category(fu, "turn_kind", 1),
       "후속 턴 유형별 실제 사례 — 되묻기 판정 검증", 5)
    sub("직전 턴 결과 → 후속 턴 유형")
    print(T.followup_cause_table(fu).to_string())

    sub("'멀티턴은 싱글턴 실패 때문인가' 검정")
    lk = T.single_turn_failure_link(fu)
    FIND["멀티턴"] = {"오즈비": lk["턴수준 오즈비(직전실패→복구성멀티턴)"],
                      "CI": list(lk["95%CI"]), "적합": lk.get("적합 기준"),
                      "유형분포": fu["turn_kind"].value_counts(normalize=True).to_dict()}
    print(f"  의도수준 상관 r={lk['의도수준 상관계수']:.3f}  (참고용)")
    print(f"  턴수준 오즈비 {lk['턴수준 오즈비(직전실패→복구성멀티턴)']:.3f} "
          f"[{lk['95%CI'][0]:.3f}, {lk['95%CI'][1]:.3f}] p={lk['p값']:.2e} "
          f"· 적합: {lk.get('적합 기준', '?')}")
    print("  ⚠", lk["주의"])
    if truth is not None:
        from mts_analysis.synth import TRUE_REPEAT_P_FAIL, TRUE_REPEAT_P_OK
        pf, po = TRUE_REPEAT_P_FAIL + .18, TRUE_REPEAT_P_OK + .05
        exp_or = (pf / (1 - pf)) / (po / (1 - po))
        print(f"  [검증] 심어놓은 오즈비 ≈ {exp_or:.1f} "
              f"(분류 손실로 추정치는 보수적으로 나옵니다)")

    sub("형식 재요청 분리 (상위 6)")
    print(T.format_request_share(fu).head(6).to_string())

    sub("불만·이관 직전 맥락 (lift)")
    cc = T.complaint_context(fu)
    print(cc.head(8).to_string())
    fu.groupby(["l2_intent", "turn_kind"]).size().rename("n").reset_index().to_csv(
        outdir / "followup_kinds.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- 7. 회수·요인
    h("7. 회수 잠재량 · 프로텍터/의도분류/환각 영향 · 출력 품질")

    sub("개선 방향별 회수 잠재량")
    rp = Q.recovery_potential(q)
    if rp.empty or "실패코드" not in rp.columns:
        print("  ⚠ 실패 귀속 건이 없어 회수 잠재량을 계산할 수 없습니다")
    else:
        print(rp[["실패코드", "설명", "담당", "실패질의량", "재방문손실(pp)",
                  "회수비중", "효율(잠재/비용)"]].to_string(index=False))
        print("  ⚠ 인과 추정이 아닌 관측 기반 상한 근사. 우선순위 비교용으로만 사용")
        rp.to_csv(outdir / "recovery_potential.csv", index=False,
                  encoding="utf-8-sig")

    sub("프로텍터 영향 — 차단 자체 vs 회수 실패")
    print(Q.protector_impact(q).to_string())

    sub("의도분류 오분류 영향")
    im = Q.intent_misclassification(q)
    if im.get("라벨공간불일치"):
        print("  ⚠", im["안내"])
        br = im["_bridge"]
        if "안내" in br:
            print("  ", br["안내"])
        else:
            br["기존분류별"].to_csv(outdir / "legacy_bridge.csv",
                                     encoding="utf-8-sig")
            FIND["운영분류"] = {"정합률": br["전체정합률"],
                                "매핑불가": br["매핑불가건수"],
                                "2단키": bool(br.get("2단키 사용"))}
            print(f"\n  크로스워크 정합률 {br['전체정합률']:.1%} "
                  f"(매핑불가 {br['매핑불가건수']:,}건) · "
                  f"2단키 {'사용' if br.get('2단키 사용') else '미사용 — INTENT_CATEGORY1 필요'}")
            if len(br["미매핑코드"]):
                print("  [크로스워크에 없는 운영 코드 — schema.LEGACY_CROSSWALK 에 추가 필요]")
                print("   " + br["미매핑코드"].to_string().replace("\n", "\n   "))
            print("\n  [기존 카테고리별 분해도 — 운영 라우팅 해상도 부족 지점]")
            print(br["기존분류별"].to_string())
            print("\n  [기존 체계가 포괄하지 못하던 신규 의도 — 보이지 않던 수요]")
            print(br["기존체계_미포괄_의도"].to_string())
            print(f"  합계 비중 {br['기존체계_미포괄_의도']['비중'].sum():.1%}")
            print("  →", br["해석"])
            br["기존분류별"].to_csv(outdir / "legacy_bridge.csv", encoding="utf-8-sig")
            br["기존체계_미포괄_의도"].to_csv(
                outdir / "legacy_uncovered.csv", encoding="utf-8-sig")
    elif "안내" in im:
        print("  ", im["안내"])
    else:
        print(f"  운영 분류 정확도 {im['전체정확도']:.1%}"
              + (f" (결측으로 {im['제외행']:,}행 제외)" if im.get("제외행") else ""))
        print(im["오분류여부별_실패율"].to_string())
        print("  [주요 혼동 쌍]"); print(im["주요혼동쌍"].head(6).to_string(index=False))
        im["주요혼동쌍"].to_csv(outdir / "intent_confusion.csv",
                                index=False, encoding="utf-8-sig")

    sub("환각 위험군 (툴 미호출 성공응답 · 근거 미인용)")
    if not gchk["근거인용 분산"]["통과"]:
        print("  ⛔", gchk["근거인용 분산"]["사유"])
    hz = Q.hallucination_risk(q)
    if "안내" in hz:
        print("  ", hz["안내"])
    else:
        print(f"  전체 위험군 비율 {hz['전체_위험군비율']:.1%}")
        print(hz["의도별"].head(6).to_string())
        hz["의도별"].to_csv(outdir / "hallucination_risk.csv", encoding="utf-8-sig")
        FIND["환각위험"] = {"전체위험군비율": hz["전체_위험군비율"],
                            "프록시정밀도": hz.get("프록시_정밀도")}
        if "프록시_정밀도" in hz:
            print(f"  검수 대조 — 프록시 정밀도 {hz['프록시_정밀도']:.1%} "
                  "(위험군 중 실제 환각 비율)")

    sub("의도별 출력 양·질")
    rqual = Q.response_quality(fu)
    print(rqual.head(10).to_string())
    rqual.to_csv(outdir / "response_quality.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------------- 8. 세션·North Star
    h("8. 세션 단위 결과 · North Star · 의도 포기 · 수요-공급 맵")

    sess = SS.session_outcomes(fu)
    sub("세션 결과 분포")
    vc = sess["session_outcome"].value_counts(normalize=True)
    for k, v in vc.items():
        print(f"  {k:<14} {v:6.1%}   {SS.SESSION_OUTCOMES.get(k, '')}")

    gap = SS.turn_vs_session_gap(fu, sess)
    FIND["세션"] = {k: v for k, v in gap.items() if k != "해석"}
    FIND["세션결과분포"] = sess["session_outcome"].value_counts(normalize=True).to_dict()
    print(f"\n  턴 성공률(폴백 포함) {gap['턴 성공률(폴백 포함)']:.1%} vs "
          f"세션 성공률 {gap['세션 성공률']:.1%} (괴리 {gap['괴리']:+.1%})")
    print("  ※ 세션 결과는 분류 실패 후 폴백 응답을 성공으로 세지 않습니다(strict)")
    print(f"  고비용 성공(재질문·다턴) {gap['고비용 성공 비율']:.1%}")
    print("  →", gap["해석"])

    sub("North Star — 주간 태스크 성공 사용자 비율")
    ns = SS.north_star(sess)
    print(ns.tail(10).to_string())
    print("  ※ 개선 전후 비교의 기준선입니다. 나머지 지표는 전부 진단용입니다.")
    ns.to_csv(outdir / "north_star.csv", encoding="utf-8-sig")

    for _oc in ("RESOLVED_HARD", "ABANDONED"):
        _ids = SM.pick_sessions(sess, _oc, 1)
        if _ids:
            print(f"\n  ▸ [{_oc}] 세션 대화 전문 — 결과 판정 검증용")
            print("    " + SM.transcripts(fu, _ids).replace("\n", "\n    "))

    sub("의도별 노력 비용 (CES 행동 프록시)")
    eb = SS.effort_by_intent(sess)
    if "안내" in eb.columns:
        print("  ", eb["안내"].iloc[0])
    else:
        print(eb.head(8).to_string())
        eb.to_csv(outdir / "effort_by_intent.csv", encoding="utf-8-sig")

    sub("의도 포기율 — 이탈의 선행 지표")
    ia = SS.intent_abandonment(q)
    if "안내" in ia.columns:
        print("  ", ia["안내"].iloc[0])
    else:
        print("  [포기효과 상위 — 실패하면 다시 안 묻는 의도]")
        print(ia.head(8).to_string())
        ia.to_csv(outdir / "intent_abandonment.csv", encoding="utf-8-sig")

    ac = SS.abandonment_to_churn(q, END)
    if "안내" in ac:
        print("  ", ac["안내"])
    else:
        print(f"\n  [의도 포기 성향별 사용자 이탈률] "
              f"(분할 기준 재시도율 중앙값 {ac['분할기준(중앙값)']})")
        print(ac["표"].to_string())
        print(f"  이탈률 차이 {ac['이탈률 차이']:+.1%} — {ac['해석']}")

    sub("수요-공급 4사분면")
    dsm = SS.demand_supply_map(q)
    if "안내" in dsm.columns:
        print("  ", dsm["안내"].iloc[0])
    else:
        print(dsm.groupby("사분면").agg(
            의도수=("질의량", "size"), 질의량합=("질의량", "sum"),
            평균성공률=("성공률", "mean")).round(3).to_string())
        sup = SS.suppressed_demand(dsm)
        if len(sup):
            print("\n  ⚠ 수요 억눌림 의심 — 실패량 기준 우선순위에서 놓치는 지점")
            print(sup[["질의량", "사용자수", "성공률", "질의량추세"]].head(8).to_string())
            print("  → 계속 실패해서 아예 묻지 않게 된 것인지, 원래 불필요한 것인지 "
                  "정성 확인이 필요합니다")
        else:
            print("\n  수요 억눌림 의심 의도 없음")
        dsm.to_csv(outdir / "demand_supply_map.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------------- 9. 메커니즘
    h("9. 메커니즘 — 진단에서 결론으로")

    sub("① 자기검열 지수 — 차단이 질문 행동 자체를 줄였는가")
    sc = MX.self_censorship(q)
    if "안내" not in sc:
        FIND["자기검열"] = {k: v for k, v in sc.items() if k != "판정"}
    if "안내" in sc:
        print("  ", sc["안내"])
    else:
        print(f"  대상 {sc['대상사용자']:,}명 · 판단성(P2/P3) 질의 비중 "
              f"{sc['차단전_판단성비중']:.1%} → {sc['차단후_판단성비중']:.1%}")
        print(f"  원시변화 {sc['원시변화']:+.4f} · 대조군 {sc['대조군변화']:+.4f} "
              f"· 보정 {sc['보정변화']:+.4f} (paired p={sc['p값']:.2e})")
        print(f"  5%p 이상 감소한 사용자 {sc['5%p이상 감소한 사용자 비율']:.1%}")
        print("  판정:", sc["판정"])

    sub("② 실질 성공률 — 렌더 성공 ≠ 사용자 성공")
    eff = MX.effective_success(fu)
    es = MX.effective_success_summary(eff)
    if "안내" not in es:
        FIND["실질성공"] = {k: v for k, v in es.items() if k != "해석"}
    if "안내" in es:
        print("  ", es["안내"])
    else:
        print(f"  시스템 성공률 {es['시스템 성공률']:.1%}")
        print(f"  실질 성공률   {es['실질 성공률 하한']:.1%} ~ "
              f"{es['실질 성공률 상한']:.1%} "
              f"(TERMINAL 구간폭 {es['구간폭(TERMINAL 비중)']:.1%})")
        print("  분포:", ", ".join(f"{k} {v:.1%}" for k, v in es["분포"].items()))
        print("  →", es["해석"])
        show_ex(SM.by_category(eff, "eff_kind", 1),
           "재분류별 실제 사례 — 이 판정이 타당한지 확인", 5)
        ebi = MX.effective_by_intent(eff)
        if len(ebi):
            print("\n  [괴리 상위 — 렌더는 됐지만 닿지 않은 의도]")
            print(ebi.head(8).to_string())
            ebi.to_csv(outdir / "effective_success.csv", encoding="utf-8-sig")

    sub("③ 실패 직후 단계 전이 — 세션 내 미시 증거")
    tb = MX.transition_by_outcome(fu)
    if "안내" in tb.columns:
        print("  ", tb["안내"].iloc[0])
    else:
        print("  [실패 후 더 자주 일어나는 전이 상위]")
        print(tb.head(8).to_string())
        tb.to_csv(outdir / "transition_by_outcome.csv", encoding="utf-8-sig")

    sub("④ 의존도 충격 — 소스 중단의 사용자 수준 효과")
    ds = MX.dependency_shock(q, OUTAGE, END)
    if "안내" not in ds:
        FIND["의존도충격"] = {"강도반응": ds["강도반응"], "대상": ds["대상사용자"],
                              "판정": ds["판정"]}
    if "안내" in ds:
        print("  ", ds["안내"])
    else:
        print(f"  대상 {ds['대상사용자']:,}명 · 의존도 5분위")
        print(ds["분위표"].to_string())
        print("  강도-반응:", {k: v for k, v in ds["강도반응"].items()})
        print("  판정:", ds["판정"])
        ds["분위표"].to_csv(outdir / "dependency_shock.csv", encoding="utf-8-sig")

    sub("⑤ 회복 상한 시뮬레이션 — 다 고치면 어디까지 가는가")
    rc = MX.recovery_ceiling(q)
    cols = ["시나리오", "치환건수", "턴성공률", "세션성공률", "무마찰세션률",
            "NorthStar", "Δ세션성공률", "Δ무마찰세션률"]
    print(rc[cols].to_string(index=False))
    if rc.attrs.get("천장경고"):
        print("\n  ⚠ North Star 가 이미 95% 이상입니다 — 천장 효과로 개선을 "
              "추적할 수 없습니다.")
        print("     '주 1회라도 성공한 사용자'는 세션이 여러 번이면 대부분 충족됩니다.")
        print("     추적 지표를 **무마찰세션률(RESOLVED)** 로 바꾸는 것을 권합니다. "
              "재질문 없이 한 번에 해결된 비율이라 개선에 민감합니다.")
    print("  ⚠ 강한 가정 위의 시뮬레이션입니다. '상한의 상한'으로만 읽으십시오.")
    rc.to_csv(outdir / "recovery_ceiling.csv", index=False, encoding="utf-8-sig")

    sub("⑥ 세그먼트 × 핵심지표 — 사용자 축 심슨 방어")
    if "오류" not in seg:
        sx = MX.segment_crosstab(q, seg)
        if "안내" in sx.columns:
            print("  ", sx["안내"].iloc[0])
        else:
            print(sx.to_string())
            sx.to_csv(outdir / "segment_crosstab.csv", encoding="utf-8-sig")
    else:
        print("  세그먼트 미도출")

    sub("⑦ 비용-편익 사분면")
    cb = MX.cost_benefit(q)
    if "안내" in cb.columns:
        print("  ", cb["안내"].iloc[0])
    else:
        print(cb.groupby("사분면").agg(
            의도수=("질의량", "size"), 총비용시간=("총비용", "sum"),
            편익합=("편익", "sum")).round(2).to_string())
        cut = cb[cb["사분면"].str.startswith("★")]
        if len(cut):
            print("\n  ★ 고비용·저편익 — 개선이 아니라 범위 축소 후보")
            print(cut[["질의량", "성공률", "p95응답ms", "총비용", "편익"]]
                  .head(6).to_string())
        cb.to_csv(outdir / "cost_benefit.csv", encoding="utf-8-sig")

    sub("⑨ 질문–응답 관련성 프록시")
    qr = RV.add_relevance(q)
    if qr["rel_coverage"].notna().any():
        print(f"  coverage 중앙 {qr['rel_coverage'].median():.3f} · "
              f"하위20% {qr['rel_coverage'].quantile(.2):.3f} · "
              f"0.1 이하 {qr['rel_coverage'].le(.1).mean():.1%}")
        bi = (qr.groupby("l2_intent")["rel_coverage"].agg(["size", "mean"])
                .rename(columns={"size": "n", "mean": "coverage"}))
        bi = bi[bi["n"] >= 30].sort_values("coverage")
        print("  [관련성 하위 의도 — 질문 내용어가 응답에 닿지 않음]")
        print(bi.head(6).round(3).to_string())
        bi.to_csv(outdir / "relevance_by_intent.csv", encoding="utf-8-sig")
    else:
        print("  ⚠ 질문 원문 또는 응답 원문 없음 — 관련성 계산 불가")
        print("     prep_data.py 재실행 필요 (answer_text 컬럼 추가됨)")

    sub("⑩ 응답 구조 — '단순 데이터 출력'의 영향")
    qs = RV.add_structure(qr)
    if "응답유형" in qs.columns and "st_본문길이" in qs.columns:
        print(qs["응답유형"].value_counts(normalize=True).round(3).to_string())
        qs2 = qs.merge(eff[["query_id", "eff_kind"]], on="query_id", how="left")
        si = RV.structure_impact(qs2, fu)
        if "안내" not in si.columns:
            print()
            print(si.to_string())
            print("  → '표만(해석 없음)' 의 복구성 후속률·훑고넘김률이 높으면 "
                  "구조화 부재가 실제 비용을 만들고 있는 것입니다")
            si.to_csv(outdir / "structure_impact.csv", encoding="utf-8-sig")
    else:
        print("  ⚠ 응답 원문(answer_text) 없음 — 구조 신호 계산 불가")
        print("     원인: prep_data.py 가 구버전입니다. 파일을 최신본으로 교체한 뒤")
        print("           `python3 prep_data.py --src ... --out ./data` 를 다시 실행하십시오.")
        print("           최신본은 실행 시 첫 줄에 'prep_data v1.4.0' 을 출력합니다.")

    sub("⑪ OTH 폴백 진단 — 분류 실패가 성공으로 집계되는가")
    _kw = {}
    if args.oth_codes:
        _kw["oth_codes"] = tuple(x.strip() for x in args.oth_codes.split(",") if x.strip())
    if args.fallback_tools:
        _kw["fallback_tools"] = tuple(
            x.strip() for x in args.fallback_tools.split(",") if x.strip())
    od = RV.oth_fallback_diagnosis(
        qs.merge(fu[["query_id", "turn_kind"]], on="query_id", how="left"), **_kw)
    for k, v in od.items():
        if k in ("해석",):
            continue
        if k.endswith("상위값"):
            print(f"  [{k}]")
            for kk, vv in v.items():
                print(f"    {kk:<28} {vv:,}")
        else:
            print(f"  {k:<26} {v}")
    FIND["폴백"] = {k: v for k, v in od.items()
                    if k != "해석" and not k.endswith("상위값")
                    and not k.startswith("⚠")}
    print("  →", od["해석"])
    _fbm = qs.assign(_c=RV._CODE_RE and None) if False else qs
    _sel = _fbm[_fbm["tool_called"].fillna("").astype(str).str.lower()
                .str.contains("work_and_news|news_and_work", regex=True)]
    if len(_sel):
        show_ex(SM.view(_sel.sample(min(3, len(_sel)), random_state=0)),
           "분류 실패 후 폴백된 실제 사례 — 질문과 답이 맞는지 확인", 3)

    sub("⑫ 질의 왜곡 — 원 질문 vs 툴에 전달된 쿼리")
    qd = RV.query_drift(qs)
    if "안내" not in qd:
        FIND["질의왜곡"] = {k: v for k, v in qd.items() if k != "해석"}
    if "안내" in qd:
        print("  ", qd["안내"])
    else:
        for k, v in qd.items():
            if k != "해석":
                print(f"  {k:<26} {v}")
        print("  →", qd["해석"])

    sub("⑬ 의도별 슬롯 복원율 (툴 인자 기준)")
    sb = RV.slot_by_intent(qs)
    if len(sb):
        print("  [target 복원율 하위 8 — 재질문 판정 정확도도 함께 낮음]")
        print(sb.head(8).to_string())
        sb.to_csv(outdir / "slot_by_intent.csv", encoding="utf-8-sig")
    else:
        print("  ⚠ 슬롯 복원 데이터 없음 — prep_data v1.5.0 이상 필요")

    sub("⑧ 대체 가능성 — 화면을 대체하는가 보완하는가")
    sb = MX.substitutability(q)
    FIND["대체가능성"] = {k: v for k, v in sb.items() if k != "판정"}
    for k, v in sb.items():
        if k != "판정":
            print(f"  {k:<26} {v}")
    print("  판정:", sb["판정"])

    # ---------------------------------------------------------------- 10. 획득
    h("10. 획득 분석 — 왜 떠났나에서 왜 들어오지 않나로")

    if not gchk["패널 가용성"]["통과"]:
        print("  ▸", gchk["패널 가용성"]["사유"])

    sub("신규 vs 재방문 구성")
    nr = AQ.new_vs_returning(q)
    if len(nr):
        print(nr.tail(10).to_string())
        print("  →", AQ.composition_verdict(nr))
        nr.to_csv(outdir / "new_vs_returning.csv", encoding="utf-8-sig")
        FIND["사용자구성"] = {"최근신규비중": float(nr["신규 비중"].tail(3).mean()),
                              "판정": AQ.composition_verdict(nr)}

    sub("신규 사용자의 첫 질의 구성 추이")
    em = AQ.entry_mix_over_time(q)
    if len(em) >= 2:
        print(em.tail(12).to_string())
        em.to_csv(outdir / "entry_mix_over_time.csv", encoding="utf-8-sig")
        sh = AQ.entry_mix_shift(q)
        if "안내" not in sh.columns:
            print("\n  [초기 vs 최근 유입 구성 변화]")
            print(sh.to_string())
            sh.to_csv(outdir / "entry_mix_shift.csv", encoding="utf-8-sig")
    else:
        print("  ⚠ 구간 부족")

    sub("진입 질문별 잔존율 (Wilson CI · 유의성)")
    er = AQ.entry_retention(q, END)
    if "안내" in er.columns:
        print("  ", er["안내"].iloc[0], "→ l1_stage 로 재시도")
        er = AQ.entry_retention(q, END, level="l1_stage", min_n=30)
    if "안내" in er.columns:
        print("  ", er["안내"].iloc[0])
    else:
        print(f"  전체 잔존율 {er.attrs['전체잔존율']:.1%} "
              f"(대상 {er.attrs['대상사용자']:,}명)")
        print("  [상위 6]"); print(er.head(6).to_string(index=False))
        print("  [하위 6]"); print(er.tail(6).to_string(index=False))
        n_sig = int((er["유의"] == "★").sum())
        print(f"  → 전체 대비 유의한 의도 {n_sig}/{len(er)}개. "
              "★ 없는 차이는 근거로 쓰지 마십시오")
        er.to_csv(outdir / "entry_retention.csv", index=False, encoding="utf-8-sig")

    sub("유입 구성 변화가 만든 잔존 손실 (반사실)")
    mc = AQ.mix_counterfactual(q, END)
    if "안내" in mc.columns:
        print("  ", mc["안내"].iloc[0])
    else:
        print(f"  기준 시기: {mc.attrs['기준시기']}")
        print(mc.tail(10).to_string())
        loss = float(mc["믹스손실"].tail(3).mean())
        obs_drop = float(mc["관측잔존"].iloc[0] - mc["관측잔존"].iloc[-1])
        share = abs(loss) / obs_drop if obs_drop > 0 else float("nan")
        print(f"  최근 3개월 평균 믹스손실 {loss:+.4f} · "
              f"관측 잔존 하락 {obs_drop:.4f} · 설명력 {share:.0%}")
        if share != share:
            print("  → 관측 잔존이 하락하지 않아 판정 보류")
        elif share >= .5:
            print("  → 하락의 절반 이상이 유입 구성 변화로 설명됩니다. "
                  "개선 대상은 응답 품질이 아니라 획득 채널입니다.")
        elif share >= .2:
            print("  → 유입 구성이 일부(20~50%) 기여합니다. 획득과 품질 양쪽 필요.")
        else:
            print(f"  → ⚠ 구성으로 설명되는 건 {share:.0%}뿐입니다. "
                  "**같은 유형의 사용자가 예전보다 덜 남고 있습니다** — "
                  "획득이 아니라 경험 자체가 나빠졌을 가능성이 큽니다.")
        mc.to_csv(outdir / "mix_counterfactual.csv", encoding="utf-8-sig")

    sub("1회성 vs 재방문 — 어떤 진입이 두 번째를 만드는가")
    op = AQ.oneshot_profile(q)
    if "안내" in op.columns:
        op = AQ.oneshot_profile(q, level="l1_stage", min_n=30)
    if "안내" in op.columns:
        print("  ", op["안내"].iloc[0])
    else:
        print(f"  전체 재방문율 {op.attrs['전체재방문율']:.1%}")
        print(op.head(6).to_string())
        op.to_csv(outdir / "oneshot_profile.csv", encoding="utf-8-sig")

    sub("응답 지연의 영향 — 경로를 먼저 나눈 뒤")
    ld = LT.add_outcomes(q, fu, sess)
    if "tool_steps" not in q.columns or q["tool_steps"].isna().all():
        print("  ⛔ tool_steps 없음 — 호출 스텝수를 통제하지 못합니다. "
              "prep_data v1.5.0 이상으로 재실행하면 '속도 문제인가 계획 "
              "복잡도 문제인가'를 가를 수 있습니다.")
    print("  [경로별 지연과 결과] 지연은 경로가 다르면 의미가 다릅니다")
    ps = LT.path_summary(ld)
    print(ps.to_string())
    ps.to_csv(outdir / "latency_by_path.csv", encoding="utf-8-sig")

    sh = LT.path_share_of_latency(ld)
    if sh:
        print("\n  [지연 분산을 무엇이 설명하는가]")
        for k, v in sh.items():
            print(f"    {k:<16} {v}")
        st_exp = sh.get("호출 스텝수 설명력")
        if st_exp is not None and st_exp == st_exp and st_exp > 0.3:
            print("    → 스텝 수가 지연 분산의 상당 부분을 설명합니다. "
                  "속도 문제가 아니라 **계획 복잡도 문제**입니다.")

    print("\n  [정상 응답 안에서만 — 질문유형·스텝수 흡수 후]")
    le = LT.latency_effect(ld)
    if "안내" in le.columns:
        print("  ", le["안내"].iloc[0])
    else:
        print(f"  기준: {le.attrs.get('기준')}")
        print(le.to_string())
        le.to_csv(outdir / "latency_effect.csv", encoding="utf-8-sig")

    print("\n  [시간대 × 지연 구간 — 다음 턴 발생률]")
    lbs = LT.latency_by_session(ld)
    if "안내" in lbs.columns:
        print("  ", lbs["안내"].iloc[0])
    else:
        print(lbs.to_string())
        lbs.to_csv(outdir / "latency_by_session.csv", encoding="utf-8-sig")

    print("\n  [인내 한계 — 몇 초부터 문제인가]")
    pt = LT.patience_threshold(ld, by="시간대")
    if "안내" in pt.columns:
        pt = LT.patience_threshold(ld)
    if "안내" in pt.columns:
        print("  ", pt["안내"].iloc[0])
    else:
        print(pt.to_string(index=False))
        print("  판정:", LT.threshold_verdict(pt))
        pt.to_csv(outdir / "latency_threshold.csv", index=False,
                  encoding="utf-8-sig")
        FIND["지연"] = {"판정": LT.threshold_verdict(pt),
                        "꺾임": pt["꺾이는 지점(초)"].dropna().tolist(),
                        "최대낙차": float(pt["낙차"].max())}

    for _pth in ("차단", "폴백"):
        fr = LT.fast_rejection(ld, _pth)
        if "안내" in fr.columns:
            continue
        print(f"\n  [{_pth} — 너무 빨리 끝나는 것이 문제인가]")
        print(fr.to_string())
        tr = fr.attrs.get("추세")
        if tr is not None and tr > 0.03:
            print(f"    → 느릴수록 대화가 이어집니다({tr:+.1%}p). "
                  "빠른 거절이 성의 없게 느껴질 가능성 — 속도가 아니라 "
                  "응답 내용의 문제입니다.")
        fr.to_csv(outdir / f"latency_fast_{_pth}.csv", encoding="utf-8-sig")
    if sh:
        FIND["지연분산"] = {k: v for k, v in sh.items() if k != "스텝별 중앙지연"}

    sub("진입 추천질문 재배치 시뮬레이션")
    ra = AQ.entry_reallocation(q, END)
    if "안내" in ra:
        ra = AQ.entry_reallocation(q, END, level="l1_stage", min_n=30)
    if "안내" in ra:
        print("  ", ra["안내"])
    else:
        for k, v in ra.items():
            if k != "주의":
                print(f"  {k:<18} {v}")
        print("  ⚠", ra["주의"])

    # ------------------------------------------------------ 11. Protector 여정
    h("11. Protector 차단 이후 여정")

    sub("차단 정의 교차 검증")
    bd = PR.block_definitions(q)
    for k, v in bd.items():
        if not k.startswith("_") and k != "판정":
            print(f"  {k:<18} {v}")
    if "판정" in bd:
        print("  →", bd["판정"])
    FIND["차단정의"] = {k: v for k, v in bd.items() if not k.startswith("_")}

    defs = ["func"] + (["flag"] if bd.get("P표기 사용가능") else [])
    for how in defs:
        blk = PR.blocked_series(q, how)
        tag = {"func": "함수 기반", "flag": "P 표기 기준"}[how]
        if blk.sum() < 30:
            print(f"\n  [{tag}] 차단 {int(blk.sum())}건 — 분석 생략")
            continue
        print(f"\n{'─'*20} [{tag}] 차단 {int(blk.sum()):,}건 {'─'*20}")

        q2 = q.copy()
        q2["차단유형"] = PR.block_types(q2, blk)
        fu2 = T.classify_followups(q2)

        sub(f"[{tag}] 차단 유형 분포")
        vc = q2.loc[blk, "차단유형"].value_counts(normalize=True)
        print("  " + ", ".join(f"{k} {v:.1%}" for k, v in vc.items()))

        sub(f"[{tag}] 층1 — 차단 직후 다음 턴")
        br = PR.immediate_reaction(fu2, blk)
        if "안내" in br.columns:
            print("  ", br["안내"].iloc[0])
        else:
            print(PR.reaction_summary(br).to_string())
            print("\n  [차단 유형별]")
            print(PR.reaction_summary(br, by="차단유형").to_string())
            byi = PR.reaction_summary(br, by="l2_intent")
            if "막다른길지수" in byi.columns:
                print("\n  [막다른 길 상위 의도 — 우회 못 하고 이탈]")
                print(byi.head(6).to_string())
                byi.to_csv(outdir / f"protector_reaction_{how}.csv",
                           encoding="utf-8-sig")
            if how == "func":
                FIND["차단반응"] = PR.reaction_summary(br)["비율"].to_dict()

        sub(f"[{tag}] 층2 — 세션 영향")
        de = PR.dead_end(fu2, blk)
        if "안내" not in de.columns:
            des = PR.dead_end_summary(de)
            print(des.to_string())
            des.to_csv(outdir / f"protector_deadend_{how}.csv", encoding="utf-8-sig")
            if how == "func" and len(des):
                FIND["막다른길"] = des["막다른길비율"].to_dict()
        ss2 = PR.session_shift(fu2, blk)
        if "안내" in ss2:
            print("  ", ss2["안내"])
        else:
            print(f"  차단 전 {ss2['차단 전']}")
            print(f"  차단 후 {ss2['차단 후']}")
            print(f"  변화   {ss2['변화']}  (대상 {ss2['대상세션']:,}세션)")
            print("  →", ss2["해석"])
            if how == "func":
                FIND["세션영향"] = ss2["변화"]

        sub(f"[{tag}] 층3 — 이후 세션 (차단 시점 기준)")
        es2 = PR.session_event_study(q2, blk)
        if "안내" in es2.columns:
            print("  ", es2["안내"].iloc[0])
        else:
            print(es2.to_string())
            print("  ※ rel<0 구간이 평평해야 이후 변화를 차단 탓으로 볼 수 있습니다")
            es2.to_csv(outdir / f"protector_eventstudy_{how}.csv",
                       encoding="utf-8-sig")

        sub(f"[{tag}] 매칭 — 같은 P3 인데 차단된 건 vs 통과한 건")
        mp = PR.matched_p3(q2, blk)
        if "안내" in mp:
            print("  ", mp["안내"])
            if "차단" in mp:
                print(f"     (P3 전체 — 차단 {mp['차단']:,} / 통과 {mp['통과']:,})")
        else:
            print(f"  매칭 {mp['매칭표본']:,}건 · {mp['층수']}개 층 "
                  f"(차단 {mp['차단']:,} / 통과 {mp['통과']:,})")
            print("\n  [균형 검정]")
            print(PR.matched_balance(mp["_matched"]).to_string())
            mo = PR.matched_outcome(mp["_matched"], fu2, sess)
            print("\n  [결과 비교 — 층 가중]")
            print(mo.to_string())
            mo.to_csv(outdir / f"protector_matched_{how}.csv", encoding="utf-8-sig")
            if how == "func":
                FIND["매칭결과"] = mo["차이"].to_dict()

    sub("정책 일관성 — 같은 P3 의 차단률 편차")
    pc = PR.policy_consistency(q, PR.blocked_series(q, "func"))
    if "안내" in pc.columns:
        print("  ", pc["안내"].iloc[0])
    else:
        print(f"  P3 중 통과 {pc.attrs['통과건수']:,}건 · "
              f"월별 차단률 변동폭 {pc.attrs.get('월별_변동폭', float('nan')):.3f}")
        print(pc.attrs["월별"].to_string())
        print("\n  [의도별 차단률 — 낮은 순]")
        print(pc.head(6).to_string())
        pc.to_csv(outdir / "protector_consistency.csv", encoding="utf-8-sig")
        FIND["정책일관성"] = {"통과건수": pc.attrs["통과건수"],
                              "월변동폭": pc.attrs.get("월별_변동폭")}

    # ------------------------------------------------- 12. 문제별 이탈 기여도
    h("12. 어떤 문제가 이탈에 얼마나 영향을 주는가")
    print("  ※ 단일 분해는 불가능합니다(이탈이 세 층이고, 문제가 겹치며, "
          "반사실이 없음).")
    print("     성격이 다른 네 가지 답을 각각 냅니다.")

    sub("A. 대화 중단 귀속 — 세션을 끝낸 '마지막 실패'로")
    att = AT.session_attribution(fu, sess)
    if "안내" in att.columns:
        print("  ", att["안내"].iloc[0])
    else:
        print(f"  실패로 끝난 세션 {att.attrs['실패세션수']:,} / "
              f"전체 {att.attrs['전체세션수']:,}")
        print(att[["세션수", "기여율", "종료위험배수", "겪은세션비율",
                   "담당", "처방"]].to_string())
        print("\n  ▸", AT.headline(att))
        att.to_csv(outdir / "attribution_session.csv", encoding="utf-8-sig")
        FIND["기여도"] = att["기여율"].to_dict()
        FIND["기여도요약"] = AT.headline(att)

    sub("B. 노출-반응 — 겪은 사용자 vs 안 겪은 사용자")
    exr = AT.exposure_response(q, END)
    if "안내" in exr.columns:
        print("  ", exr["안내"].iloc[0])
    else:
        print(exr.to_string(index=False))
        print("  ⚠", AT.exposure_caution())
        exr.to_csv(outdir / "attribution_exposure.csv", index=False,
                   encoding="utf-8-sig")

    sub("C. 단독 해결 효과 — 각 문제를 하나씩만 없앴을 때")
    fix = AT.single_fix(q)
    print(fix.to_string(index=False))
    print("  ⚠ '고치면 그 질의가 성공한다'는 가정 위의 상한입니다. "
          "문제 간 비교용으로만 쓰십시오.")
    fix.to_csv(outdir / "attribution_singlefix.csv", index=False,
               encoding="utf-8-sig")

    sub("D. 겪은 문제 개수와 이탈률")
    pcc = AT.problem_count_churn(q, END)
    if len(pcc):
        print(pcc.to_string())
        print("  ⚠", pcc.attrs.get("주의", ""))
        pcc.to_csv(outdir / "attribution_count.csv", encoding="utf-8-sig")

    sub("통합 — 근거 등급과 함께")
    _mb = FIND.get("매칭결과") or {}
    comb = AT.combined(att, exr, fix,
                       {"차이": _mb.get("대화 종료율")} if _mb else None)
    if "안내" not in comb.columns:
        print(comb.to_string(index=False))
        comb.to_csv(outdir / "attribution_combined.csv", index=False,
                    encoding="utf-8-sig")
        print("\n  ▸ 근거 등급이 '상관'인 행은 '이 문제가 이탈을 만든다'가 아니라 "
              "'함께 나타난다'로 읽으십시오.")

    # ------------------------------------------------- 13. 근본 의도
    h("13. 사용자가 실제로 알고 싶었던 것")
    print("  ※ '질문의 단어가 답변에 있는가'는 표면 관련성입니다.")
    print("     여기서는 **어떤 질문 뒤에 무엇이 따라오는지**로 근본 의도를 역산합니다.")

    prep = ND.prepare(fu)
    n_deep = int((prep["_ok"] & prep["_deepen"]).sum())
    print(f"  심화 후속(성공 후 · 같은 대상 · 되묻기 아님) {n_deep:,}건")

    sub("표면 질문 뒤에 실제로 이어서 묻는 것")
    un = ND.underlying_need(prep)
    if "안내" in un.columns:
        print("  ", un["안내"].iloc[0])
    else:
        print(un[["표면 질문", "이어서 묻는 것", "건수"]].to_string(index=False))
        print("  → 괄호 안 배수는 기저율 대비입니다. 2배면 그 질문 고유의 후속입니다.")
        un.to_csv(outdir / "underlying_need.csv", index=False, encoding="utf-8-sig")

    show_ex(SM.followup_pairs(prep, 5), "질문 → 이어서 물은 것 (근본 의도 역산 근거)", 5)
    sub("자기완결률 — 한 번에 끝나는가")
    scn = ND.self_contained(prep)
    if "안내" in scn.columns:
        print("  ", scn["안내"].iloc[0])
    else:
        print("  [자기완결률 하위 — 답은 맞는데 더 묻게 만드는 질문]")
        print(scn.head(8).to_string())
        scn.to_csv(outdir / "self_contained.csv", encoding="utf-8-sig")

        sub("되묻기 × 연쇄 4사분면")
        qd = ND.need_quadrant(scn)
        if "안내" not in qd.columns:
            print(qd.groupby("구분").agg(
                의도수=("건수", "size"), 평균되묻기=("되묻기율", "mean"),
                평균연쇄=("연쇄깊이", "mean")).round(3).to_string())
            star = qd[qd["구분"].str.startswith("★")]
            if len(star):
                print("\n  ★ 한 번에 못 채움 — 답은 맞는데 세 번 더 묻게 만드는 지점")
                print(star[["질문", "자기완결률", "되묻기율", "연쇄깊이"]]
                      .head(6).to_string(index=False))
                print("  → 실패 지표에는 잡히지 않습니다. 응답 템플릿 과제입니다.")
            qd.to_csv(outdir / "need_quadrant.csv", encoding="utf-8-sig")
            FIND["한번에못채움"] = int(len(star))

    sub("맥락에 따라 원하는 것이 다른가")
    cc = ND.context_conditional(prep)
    if "안내" in cc.columns:
        print("  ", cc["안내"].iloc[0])
    else:
        print(cc.head(10).to_string(index=False))
        print("  → 같은 질문이라도 직전 단계가 다르면 이어서 묻는 것이 달라집니다. "
              "맥락별 응답 구성의 근거입니다.")
        cc.to_csv(outdir / "context_conditional.csv", index=False,
                  encoding="utf-8-sig")

    sub("충족률 — 응답에 필요한 정보가 담겼는가")
    ncfg_path = outdir / "needs_config.json"
    ncfg = ND.ensure_needs_config(ncfg_path, scn, un)
    ful = ND.fulfillment(q, ncfg)
    if "안내" in ful.columns:
        print("  ", ful["안내"].iloc[0])
        print(f"     ▸ {ncfg_path.name} 에 의도별 권장 항목과 키워드를 채우면 "
              "다음 실행부터 계산됩니다.")
        print("       (위 후속 분포를 근거로 초안을 만들어 두었습니다)")
    else:
        print(ful[["질문", "건수", "충족률"]].to_string(index=False))
        ful.to_csv(outdir / "fulfillment.csv", index=False, encoding="utf-8-sig")
        fv = ND.fulfillment_validation(q, prep, ful)
        if "안내" not in fv:
            print(f"\n  [검증] 충족률↔자기완결률 {fv['충족률↔자기완결률']:+.3f} · "
                  f"충족률↔되묻기율 {fv['충족률↔되묻기율']:+.3f} (n={fv['n']})")
            print("  →", fv["해석"])

    # ---------------------------------------------------------------- 14. 결핍
    h("14. 지금 데이터로 답할 수 없는 것")

    _pm = {}
    if args.data:
        _pmp = Path(args.data) / "prep_meta.json"
        if _pmp.exists():
            try:
                _pm = json.loads(_pmp.read_text(encoding="utf-8"))
            except Exception:
                pass

    cfg_path = outdir / "gaps_config.json"
    cfg = GP.ensure_config(cfg_path)
    auto = GP.detect(q, {**d, **({"cohort": cr} if "cr" in dir() else {})}, gchk, _pm)
    merged = GP.merge(auto, cfg)

    sub("결핍 매트릭스")
    show = merged[["상태", "질문", "필요", "대안", "한계", "확보시",
                   "난이도", "담당", "일정"]]
    print(show.to_string(index=False))
    merged.to_csv(outdir / "data_gaps.csv", index=False, encoding="utf-8-sig")

    sub("분석 불능 비율")
    ba = GP.blocked_analyses(gchk, {**d, "cohort": 1})
    print(f"  설계한 점검·분석 {ba['전체']}개 중 {ba['불능']}개가 "
          f"데이터 제약으로 결론에 이르지 못했습니다 ({ba['불능비율']:.0%})")
    print(ba["표"][~ba["표"]["가능"]][["분석", "사유"]].to_string(index=False))

    sub("불확실 구간 — 결핍이 만드는 '모르는 폭'")
    ub = GP.uncertainty_bands(FIND)
    if len(ub):
        print(ub.to_string(index=False))
        ub.to_csv(outdir / "uncertainty_bands.csv", index=False, encoding="utf-8-sig")
    else:
        print("  계산 가능한 구간이 없습니다")

    sub("결측 규모")
    print(GP.missing_scale(q).to_string(index=False))
    wl = GP.window_loss({"cohort": pd.read_csv(outdir / "cohort_retention.csv")
                         if (outdir / "cohort_retention.csv").exists() else None})
    if wl:
        print(f"\n  관측 창 미도달: 코호트 표의 {wl['미관측 셀 비율']:.1%} 셀이 "
              f"아직 판정 불가 ({wl['일부라도 미관측인 코호트']}/{wl['전체 코호트']} 코호트)")

    st = GP.config_status(cfg, merged)
    FIND["결핍"] = {"불능비율": ba["불능비율"], "결핍항목수": st["결핍항목"],
                    "미기입": st["미기입"]}
    sub("사람이 채워야 할 것")
    if st["미기입"] > 0 or st["오판사례"] == 0:
        print(f"  ▸ {cfg_path.name} 을 열어 아래를 채우면 보고서에 반영됩니다.")
        print(f"     · 결핍 {st['결핍항목']}개 중 {st['미기입']}개에 "
              "난이도·담당·일정 미기입")
        print(f"     · 오판사례 {st['오판사례']}건 기록됨 "
              "(데이터가 없어 잘못 볼 뻔했던 사례)")
        print("     파일이 없으면 방금 템플릿을 만들어 두었습니다.")
    else:
        print(f"  ✅ 결핍 {st['결핍항목']}개 전부 기입 완료 · "
              f"오판사례 {st['오판사례']}건")

    h("완료")
    print(f"  산출물 → {outdir.resolve()}")

    sub("원본 표본 일괄 저장")
    try:
        _saved = SM.dump_all(qs if "qs" in dir() else q, fu, sess, outdir,
                             eff if "eff" in dir() else None)
        print(f"  out/samples/ 에 {len(_saved)}개 파일 저장 — "
              "각 측정 지점의 원본 데이터입니다")
        for _k, _v in _saved.items():
            print(f"    · {_k}.csv ({_v}건)")
        print("  세션_대화전문.txt 은 결과 판정(해결/이탈)을 눈으로 확인할 때 쓰십시오")
    except Exception as _e:
        print(f"  ⚠ 표본 저장 실패: {_e}")

    # 버퍼 방출 — 보고서 장 순서
    sys.stdout = _real_out
    emit_ordered(sys.stdout)

    try:
        (outdir / "findings.json").write_text(
            json.dumps(FIND, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"  분석 요약 → {(outdir / 'findings.json').name} ({len(FIND)}항목)")
    except Exception as e:
        print(f"  ⚠ findings.json 저장 실패: {e}")

    if tee is not None:
        sys.stdout = tee.stream
        raw = tee.buf.getvalue()
        meta = {"now": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "src": args.data or "(합성 데이터)", "out": str(outdir.resolve()),
                "outage": args.outage, "protector": args.protector, "end": args.end}
        for name in [x.strip() for x in args.report.split(",") if x.strip()]:
            rp = outdir / name
            if rp.suffix == ".html":
                content = to_html(raw, meta)
            elif rp.suffix == ".md":
                content = to_markdown(raw, meta)
            else:
                content = raw
            rp.write_text(content, encoding="utf-8")
            print(f"  리포트 → {rp.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())