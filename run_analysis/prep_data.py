"""
실데이터 → 분석 스키마 어댑터.

    python prep_data.py --src labeled_all.csv --out ./data

원본 컬럼:
  ID, APP_NAME, APP_ID, VERSION, ELAPSED_TIME, CHAT_REQ_DATE, CHAT_RES_DATE,
  CHAT_USER_ID, INTENT_CATEGORY1, INTENT_CATEGORY2, QUESTION, ANSWER,
  FUNCTIONS, MSG_ID, CREATED_AT
  + 재어노테이션 컬럼 (l1_stage, l2_intent, f1~f6, answerable, outcome 등)

세션 컬럼이 원본에 없으므로 사용자별 무활동 간격으로 파생합니다.
"""
from __future__ import annotations

__version__ = "1.9.0"   # FUNCTIONS 인자에서 슬롯 복원

import argparse
import ast
import json
from pathlib import Path

import re

import numpy as np
import pandas as pd

# 원본 → 표준 직접 매핑
DIRECT = {
    "MSG_ID": "query_id",
    "CHAT_USER_ID": "user_id",
    "INTENT_CATEGORY1": "intent_pred_group",  # 운영 분류 그룹 (BASIC 등 충돌 해소에 필수)
    "INTENT_CATEGORY2": "intent_pred",   # 운영 의도분류 (재어노테이션 정답과 대조)
    "QUESTION": "query_text",
}

# 재어노테이션 프롬프트 출력 → 표준 스키마
ANNOTATION_ALIASES: dict[str, str] = {
    "stage": "l1_stage",
    "primary": "l2_intent",
    "target": "f1_target_type",
    "tense": "f2_tense",
    "personalization": "f3_personal",
    "compliance": "f4_compliance",
    "response_type": "f5_response",
    "turn_type": "f6_turn",
}

# Protector 차단 표기 컬럼 후보 (값이 "P" 인 열)
PROTECTOR_COL_HINTS = ("PROTECTOR", "protector", "PROTECT", "BLOCK", "BLOCKED",
                       "차단", "프로텍터", "FLAG", "P")


def detect_protector_col(df: pd.DataFrame) -> str | None:
    """
    'P' 로 차단을 표기한 열을 찾는다.
    이름으로 먼저 찾고, 없으면 값 패턴(P 또는 공백만으로 구성)으로 탐색한다.
    """
    for c in df.columns:
        if any(h.lower() == str(c).strip().lower() for h in PROTECTOR_COL_HINTS):
            return c
    for c in df.columns:
        v = df[c].dropna().astype(str).str.strip().str.upper()
        if len(v) == 0:
            continue
        uniq = set(v.unique())
        if uniq and uniq <= {"P", "", "Y", "N", "0", "1", "TRUE", "FALSE"} \
                and "P" in uniq and 0 < (v == "P").mean() < 0.6:
            return c
    return None


# facets 가 중첩 JSON 한 컬럼에 들어있는 경우 그 컬럼명 후보
FACET_CONTAINERS = ("facets", "FACETS", "facet")

# ANSWER 는 {"id": ..., "type": "bot", "text": "..."} 형태의 JSON.
# 정상 렌더 응답은 text 안에 HTML 태그(<div class=...>)를 포함하고,
# 실패·차단 응답은 태그 없이 순수 문자열만 들어온다.
HTML_TAG_RE = re.compile(
    r"<\s*(div|span|p|table|tbody|thead|tr|td|th|ul|ol|li|br|hr|a|b|i|em|strong"
    r"|img|section|article|h[1-6]|button|input|iframe|svg|canvas|pre|code)"
    r"(\s[^<>]*)?/?\s*>", re.IGNORECASE)

# 차단 판별 보조 문구. f4_compliance 로 대부분 걸리지만 이중 확인용.
BLOCK_PATTERNS = (
    "투자 판단", "투자판단", "투자 권유", "투자권유", "책임지지",
    "판단을 드릴 수", "권유해 드릴 수", "조언해 드릴 수", "추천해 드릴 수",
    "제공해 드릴 수 없", "答변드릴 수 없",
)


def extract_answer_text(v) -> str:
    """ANSWER JSON 에서 bot 발화 text 만 뽑는다. 형식이 깨져도 죽지 않게."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, dict):
        items = [v]
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        sv = str(v).strip()
        if not sv:
            return ""
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(sv)
                break
            except Exception:
                continue
        if parsed is None:
            return sv                      # JSON 이 아니면 원문 그대로
        items = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    texts = []
    for it in items:
        if isinstance(it, dict):
            if it.get("type") not in (None, "bot"):
                continue
            texts.append(str(it.get("text") or it.get("content") or ""))
        else:
            texts.append(str(it))
    return "\n".join(t for t in texts if t)


def has_html(text: pd.Series) -> pd.Series:
    return text.fillna("").str.contains(HTML_TAG_RE, regex=True)


def flatten_facets(df: pd.DataFrame) -> pd.DataFrame:
    """중첩 JSON facets 컬럼을 평탄화한다."""
    col = next((c for c in FACET_CONTAINERS if c in df.columns), None)
    if col is None:
        return df
    def _load(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str) and v.strip():
            for loader in (json.loads, ast.literal_eval):
                try:
                    r = loader(v)
                    if isinstance(r, dict):
                        return r
                except Exception:
                    pass
        return {}
    exp = pd.json_normalize(df[col].map(_load)).set_index(df.index)
    # facets 안의 키가 위치 기반 리스트인 경우도 방어
    if exp.empty and df[col].map(lambda v: isinstance(v, (list, tuple))).any():
        keys = ["target", "tense", "personalization", "compliance",
                "response_type", "turn_type"]
        exp = pd.DataFrame(df[col].map(
            lambda v: dict(zip(keys, v)) if isinstance(v, (list, tuple)) else {}
        ).tolist(), index=df.index)
    new = [c for c in exp.columns if c not in df.columns]
    print(f"  facets 평탄화: {col} → {list(exp.columns)}")
    return df.join(exp[new])


def derive_signals(df: pd.DataFrame) -> pd.DataFrame:
    """ANSWER/FUNCTIONS 에서 판정 신호를 뽑는다."""
    out = pd.DataFrame(index=df.index)
    out["answer_text"] = (df["ANSWER"].map(extract_answer_text)
                          if "ANSWER" in df.columns else "")
    out["has_html"] = has_html(out["answer_text"])
    out["has_func"] = (df["FUNCTIONS"].map(parse_functions).notna()
                       if "FUNCTIONS" in df.columns else False)
    return out


def derive_outcome(df: pd.DataFrame, sig: pd.DataFrame) -> pd.Series:
    """
    HTML 렌더 여부로 정상/비정상을 가르고, 비정상을 차단/실패로 나눈다.

      has_html                        → success   (정상 렌더)
      ~has_html & (P3 or 거절문구)     → blocked
      ~has_html & 그 외                → fail

    전제: 이 챗봇에는 '툴 없이 정상 응답'이 존재하지 않는다(사용자 확인).
    """
    txt = sig["answer_text"]
    blk_txt = txt.str.contains("|".join(BLOCK_PATTERNS), regex=True, na=False)
    if "f4_compliance" in df.columns:
        blk_p3 = df["f4_compliance"].astype(str).eq("P3")
    else:
        blk_p3 = pd.Series(False, index=df.index)
    blocked = (~sig["has_html"]) & (blk_p3 | blk_txt)
    return pd.Series(np.select([sig["has_html"], blocked],
                               ["success", "blocked"], default="fail"),
                     index=df.index)


def derive_answerable(df: pd.DataFrame, sig: pd.DataFrame) -> pd.Series:
    """
    FUNCTIONS × HTML 조합으로 실패 원인을 한 단계 더 쪼갠다.

      툴 호출 O + 렌더 X → 툴은 돌았는데 결과가 없음  → no_source (데이터 결손)
      툴 호출 X + 렌더 X → 툴 자체가 안 불림          → no_tool  (라우팅·미구현)

    ★ 여전히 근사입니다. no_source 안에서 D1(미보유)/D2(중단)/D3(커버리지)는
      갈리지 않고, no_tool 안에서 T1(미구현)/T2(오라우팅)도 갈리지 않습니다.
      정확한 귀속은 어노테이션의 answerable 필드가 있어야 합니다.
    """
    out = pd.Series("unknown", index=df.index, dtype=object)
    oc = df["outcome"]
    out[oc.eq("success")] = "yes"
    out[oc.eq("blocked")] = "blocked"

    fail = oc.eq("fail")
    # 인증 필요 의도의 실패는 인증 게이트 쪽으로 먼저 귀속
    if "f3_personal" in df.columns:
        need_auth = df["f3_personal"].astype(str).eq("account_required")
        out[fail & need_auth] = "no_auth"
    rest = fail & out.eq("unknown")
    out[rest & sig["has_func"]] = "no_source"
    out[rest & ~sig["has_func"]] = "no_tool"
    # followup 턴의 툴 미호출은 맥락 상속 실패일 가능성이 큼
    if "f6_turn" in df.columns:
        fu = df["f6_turn"].astype(str).eq("followup")
        out[fail & fu & out.eq("no_tool")] = "no_slot"
    return out


def parse_kdatetime(sr: pd.Series) -> pd.Series:
    """
    한국어 오전/오후 표기 파싱: "2025-11-04 오후 5:10:47"

    pandas 는 오전/오후를 못 읽으므로 AM/PM 으로 치환한 뒤 12시간제로 파싱한다.
    (오후 12시 → 12:00, 오전 12시 → 00:00 으로 올바르게 처리됨)
    """
    t = sr.astype(str).str.strip()
    t = (t.str.replace("오전", "AM", regex=False)
          .str.replace("오후", "PM", regex=False)
          .str.replace(r"\s+", " ", regex=True))
    out = pd.to_datetime(t, format="%Y-%m-%d %p %I:%M:%S", errors="coerce")
    # 시각이 앞, AM/PM 이 뒤에 오는 변형도 시도
    miss = out.isna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(
            t[miss], format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
    # 그래도 남으면 일반 파서 (ISO, 슬래시 등)
    miss = out.isna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(t[miss], errors="coerce")
    return out


# 인자가 담기는 컨테이너 키 후보 (실로그: inputs)
ARG_CONTAINER_KEYS = ("inputs", "arguments", "args", "parameters",
                      "params", "input", "argument")
# 인자가 아닌 메타 필드
_META_KEYS = {"step", "name", "brief", "id", "type", "order", "index",
              "tool", "func", "function", "desc", "description", "reason"}

# 툴 인자명 → 슬롯. 함수 명세 기준. 실제 키가 다르면 여기만 고치십시오.
ARG_TO_SLOT = {
    "target": ("company_name", "index_name", "company", "stock_name",
               "sector_name", "ticker", "code"),
    "period": ("base_d", "end_d", "start_d", "base_q", "base_year",
               "start_dt", "end_dt", "base_dt", "from_dt", "to_dt",
               "start_date", "end_date"),
    "metric": ("metric", "metrics"),
    "sort":   ("order", "sort", "order_by"),
    "count":  ("top_n", "top_k", "limit", "count", "n"),
    "query":  ("query", "q", "search_query"),
}
_SLOT_OF_ARG = {a: sl for sl, args in ARG_TO_SLOT.items() for a in args}


def parse_function_calls(v) -> list[dict]:
    """
    FUNCTIONS → [{"name": ..., "args": {...}}] 로 정규화한다.

    형식이 제각각이라 방어적으로 처리한다.
      · [{"name": "...", "arguments": {...}}]
      · [{"name": "...", "arguments": "{...}"}]      (문자열 JSON)
      · [{"function": {"name": ..., "arguments": ...}}]
      · {"name": ..., "parameters": {...}}
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple)):
        items = list(v)
    elif isinstance(v, dict):
        items = [v]
    else:
        sv = str(v).strip()
        if sv in ("", "[]", "{}", "null", "None", "-", "nan"):
            return []
        items = None
        for loader in (json.loads, ast.literal_eval):
            try:
                items = loader(sv)
                break
            except Exception:
                continue
        if items is None:
            return [{"name": t.strip().strip("'\""), "args": {}}
                    for t in sv.replace("|", ",").split(",") if t.strip()]
        if isinstance(items, dict):
            items = [items]

    out = []
    for it in items or []:
        if not isinstance(it, dict):
            nm = str(it).strip().strip("'\"")
            if nm:
                out.append({"name": nm, "args": {}})
            continue
        if isinstance(it.get("function"), dict):      # 중첩 형식
            it = it["function"]
        name = str(it.get("name") or it.get("tool") or it.get("func") or "").strip()
        raw = None
        for k in ARG_CONTAINER_KEYS:
            if it.get(k) is not None:
                raw = it[k]
                break
        if raw is None:
            # 알려진 키가 없으면 dict 값을 가진 첫 비메타 필드를 인자로 본다
            for k, v2 in it.items():
                if k.lower() not in _META_KEYS and isinstance(v2, dict):
                    raw = v2
                    break
        if raw is None:
            raw = {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                try:
                    raw = ast.literal_eval(raw)
                except Exception:
                    raw = {}
        if not isinstance(raw, dict):
            raw = {}
        if name:
            out.append({"name": name, "args": raw,
                        "step": it.get("step"), "brief": it.get("brief")})
    return out


def slots_from_calls(calls: list[dict]) -> dict:
    """툴 호출 인자에서 L3 슬롯을 복원한다."""
    slots = {k: [] for k in ("target", "period", "metric", "sort", "count", "query")}
    for c in calls:
        for k, v in (c.get("args") or {}).items():
            sl = _SLOT_OF_ARG.get(str(k).strip().lower())
            if sl is None or v is None:
                continue
            vals = v if isinstance(v, (list, tuple)) else [v]
            for x in vals:
                xs = str(x).strip()
                if xs and xs.lower() not in ("none", "null", "nan", ""):
                    slots[sl].append(xs)
    return {k: list(dict.fromkeys(v)) for k, v in slots.items()}


def parse_functions(v):
    """FUNCTIONS 컬럼 → 호출된 툴 이름. 형식이 제각각이라 방어적으로 처리."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple)):
        items = list(v)
    elif isinstance(v, str):
        s = v.strip()
        if s in ("", "[]", "{}", "null", "None", "-"):
            return None
        try:
            items = json.loads(s)
        except Exception:
            try:
                items = ast.literal_eval(s)
            except Exception:
                items = [x for x in s.replace("|", ",").split(",") if x.strip()]
    else:
        return None
    if isinstance(items, dict):
        items = [items]
    if not items:
        return None
    names = []
    for it in items:
        if isinstance(it, dict):
            names.append(str(it.get("name") or it.get("function")
                              or it.get("tool") or next(iter(it.values()), "")))
        else:
            names.append(str(it).strip().strip("'\""))
    names = [n for n in names if n]
    return "|".join(names) if names else None


def parse_list(v):
    """slot_target / source_expected → 리스트."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    s = str(v).strip()
    if s in ("", "[]", "nan", "None", "-"):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            r = loader(s)
            if isinstance(r, (list, tuple)):
                return [str(x) for x in r]
        except Exception:
            pass
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()]


def derive_sessions(df: pd.DataFrame, gap_min: int = 30) -> pd.Series:
    """
    사용자별 무활동 간격 기준 세션 분할.

    30분은 웹 분석의 관례입니다. MTS 챗봇은 장중 연속 사용이 길 수 있으니
    실제 턴 간격 분포를 보고 조정하십시오 (아래 진단 출력 참조).
    """
    d = df.sort_values(["user_id", "ts"])
    gap = d.groupby("user_id")["ts"].diff().dt.total_seconds()
    new = (gap.isna() | (gap > gap_min * 60)).astype(int)
    seq = new.groupby(d["user_id"]).cumsum()
    sid = d["user_id"].astype(str) + "-S" + seq.astype(int).astype(str)
    return sid.reindex(df.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--sep", default=None, help="구분자. 생략 시 자동 감지")
    ap.add_argument("--session-gap", type=int, default=30, help="세션 분할 간격(분)")
    ap.add_argument("--ts-col", default="CHAT_REQ_DATE", help="질의 시각 컬럼")
    ap.add_argument("--elapsed-unit", default="ms", choices=["ms", "s"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"prep_data v{__version__}")
    raw = pd.read_csv(args.src, sep=args.sep or None,
                      engine="python" if args.sep is None else "c")
    print(f"원본 {len(raw):,}행 · {len(raw.columns)}컬럼")

    df = flatten_facets(raw).rename(columns={**DIRECT, **ANNOTATION_ALIASES}).copy()
    # 평탄화가 facets.target 처럼 접두어를 남긴 경우 한 번 더 매핑
    df = df.rename(columns={f"facets.{k}": v for k, v in ANNOTATION_ALIASES.items()})

    # --- 시각
    ts_col = args.ts_col
    if ts_col not in df.columns:
        for alt in ("CHAT_REQ_DATE", "CAT_REQ_DATE", "CREATED_AT", "CHAT_RES_DATE"):
            if alt in df.columns:
                ts_col = alt
                print(f"  ℹ ts 컬럼 자동 선택: {ts_col}")
                break
        else:
            print(f"  ❌ 시각 컬럼 없음. --ts-col 로 지정하십시오. "
                  f"후보: {list(df.columns)[:15]}")
            return 1
    df["ts"] = parse_kdatetime(df[ts_col])
    # 결측이면 다른 시각 컬럼으로 보충 (원본 결측이 대부분)
    for alt in ("CREATED_AT", "CHAT_RES_DATE", "CHAT_REQ_DATE", "CAT_REQ_DATE"):
        if alt in df.columns and df["ts"].isna().any():
            fill = parse_kdatetime(df[alt])
            n_before = int(df["ts"].isna().sum())
            df["ts"] = df["ts"].fillna(fill)
            recovered = n_before - int(df["ts"].isna().sum())
            if recovered:
                print(f"  ℹ {alt} 로 {recovered:,}행 보충")
    bad = int(df["ts"].isna().sum())
    if bad:
        ex = df.loc[df["ts"].isna(), ts_col].astype(str).head(3).tolist()
        print(f"  ⚠ ts 확보 실패 {bad:,}행 ({bad/len(df):.1%}) — 제외. 예: {ex}")
        df = df[df["ts"].notna()]
    print(f"  기간 {df['ts'].min():%Y-%m-%d} ~ {df['ts'].max():%Y-%m-%d}")

    # --- 응답시간
    if "ELAPSED_TIME" in df.columns:
        e = pd.to_numeric(df["ELAPSED_TIME"], errors="coerce")
        df["latency_ms"] = e * (1000 if args.elapsed_unit == "s" else 1)
        med = df["latency_ms"].median()
        print(f"  ELAPSED_TIME 중앙값 → {med:,.0f}ms "
              f"({'단위 확인 필요' if med < 50 or med > 60000 else '타당'})")
    else:
        df["latency_ms"] = np.nan

    # --- 세션 파생
    df["session_id"] = derive_sessions(df, args.session_gap)
    g = df.sort_values(["user_id", "ts"]).groupby("user_id")["ts"].diff().dt.total_seconds()
    print(f"  턴 간격 중앙값 {g.median():,.0f}초 · p90 {g.quantile(.9):,.0f}초")
    print(f"  세션 {df['session_id'].nunique():,}개 "
          f"(세션당 평균 {len(df)/df['session_id'].nunique():.2f}턴, "
          f"gap={args.session_gap}분 기준)")

    # --- 응답 · 툴
    # response_len 은 JSON 전체가 아니라 추출한 본문 기준이어야 함
    if "FUNCTIONS" in df.columns:
        calls = df["FUNCTIONS"].map(parse_function_calls)
        df["tool_called"] = calls.map(
            lambda cs: "|".join(c["name"] for c in cs) or None)
        df["tool_args"] = calls.map(
            lambda cs: json.dumps({c["name"]: c["args"] for c in cs},
                                  ensure_ascii=False) if cs else None)
        # 멀티스텝 계획: 한 응답에 몇 개의 툴을 연쇄 호출했는가
        df["tool_steps"] = calls.map(len)
        df["tool_brief"] = calls.map(
            lambda cs: " / ".join(str(c["brief"]) for c in cs if c.get("brief"))
            or None)
        print(f"  툴 호출 있는 응답 {df['tool_called'].notna().mean():.1%}")

        # --- 인자 → 슬롯 복원
        sl = calls.map(slots_from_calls)
        df["slot_target"] = sl.map(lambda d_: d_["target"])
        df["slot_period"] = sl.map(lambda d_: "~".join(d_["period"]) or None)
        df["slot_metric"] = sl.map(lambda d_: "|".join(d_["metric"]) or None)
        df["slot_sort"] = sl.map(lambda d_: (d_["sort"] or [None])[0])
        df["slot_count"] = sl.map(lambda d_: (d_["count"] or [None])[0])
        df["tool_query"] = sl.map(lambda d_: (d_["query"] or [None])[0])

        keyc = pd.Series([k for cs in calls for c in cs
                          for k in (c.get("args") or {})]).value_counts()
        print(f"  멀티스텝: 평균 {df['tool_steps'].mean():.2f}개 호출 · "
              f"2개 이상 {df['tool_steps'].ge(2).mean():.1%}")
        print(f"  슬롯 복원: target {df['slot_target'].map(len).gt(0).mean():.1%} · "
              f"period {df['slot_period'].notna().mean():.1%} · "
              f"metric {df['slot_metric'].notna().mean():.1%} · "
              f"query {df['tool_query'].notna().mean():.1%}")
        if len(keyc):
            unmapped = [k for k in keyc.index if str(k).lower() not in _SLOT_OF_ARG]
            print(f"  [툴 인자 키 상위] "
                  + ", ".join(f"{k}({v:,})" for k, v in keyc.head(10).items()))
            if unmapped:
                print(f"  ℹ 슬롯 미매핑 인자: {unmapped[:8]} "
                      "— 필요하면 ARG_TO_SLOT 에 추가하십시오")
        else:
            ex = df["FUNCTIONS"].dropna().astype(str).head(2).tolist()
            print("  ⚠ 툴 인자를 하나도 파싱하지 못했습니다. FUNCTIONS 원문 샘플:")
            for e in ex:
                print(f"     {e[:160]}")

    # --- 리스트형 컬럼
    for c in ("slot_target", "source_expected"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: v if isinstance(v, list) else parse_list(v))
        else:
            df[c] = [[] for _ in range(len(df))]

    # --- Protector 차단 표기 (사용자 제공) 인식
    pcol = detect_protector_col(raw)
    if pcol is not None:
        v = raw[pcol].fillna("").astype(str).str.strip().str.upper()
        df["protector_flag"] = v.isin(["P", "Y", "1", "TRUE"]).to_numpy()
        print(f"\n  Protector 표기 열 '{pcol}' 인식 — "
              f"차단 {df['protector_flag'].mean():.1%} "
              f"({int(df['protector_flag'].sum()):,}건)")
    else:
        df["protector_flag"] = pd.NA
        print("\n  ℹ Protector 표기 열을 찾지 못했습니다 — 함수 기반 파생만 사용합니다")

    # --- 신호 추출 → outcome / answerable 파생
    sig = derive_signals(df)
    print(f"\n  ANSWER 파싱: 텍스트 추출 {(sig['answer_text'].str.len()>0).mean():.1%} "
          f"· HTML 렌더 {sig['has_html'].mean():.1%} · 툴 호출 {sig['has_func'].mean():.1%}")
    xt = pd.crosstab(sig["has_func"], sig["has_html"])
    xt.index = ["툴X", "툴O"][:len(xt)]
    xt.columns = ["렌더X", "렌더O"][:len(xt.columns)]
    print("  [FUNCTIONS × HTML 교차]")
    print("   " + xt.to_string().replace("\n", "\n   "))
    if sig["has_html"].mean() < 0.2 or sig["has_html"].mean() > 0.98:
        print("   ⚠ HTML 비율이 극단적입니다. HTML_TAG_RE 가 실제 마크업과 맞는지 확인하십시오.")
        ex = sig.loc[~sig["has_html"], "answer_text"].head(2).tolist()
        print("   [렌더X 샘플] " + " / ".join(x[:70] for x in ex))

    df["answer_text"] = df["ANSWER"] if "ANSWER" in df.columns else sig["answer_text"]
    df["response_len"] = sig["answer_text"].str.len()
    df["cited"] = sig["has_func"]           # 툴 결과에 근거한 응답인지

    if "outcome" not in df.columns:
        df["outcome"] = derive_outcome(df, sig)
        vc = df["outcome"].value_counts(normalize=True)
        print("\n  outcome 파생 (HTML 렌더 기준):",
              ", ".join(f"{k} {v:.1%}" for k, v in vc.items()))
        for lbl in ("blocked", "fail"):
            sm = sig.loc[df["outcome"].eq(lbl), "answer_text"].head(2).tolist()
            if sm:
                print(f"    [{lbl} 샘플] " + " / ".join(x[:60] for x in sm))

    _derived = {"outcome_derived": "outcome" not in df.columns,
                "answerable_derived": "answerable" not in df.columns}
    if "answerable" not in df.columns:
        df["answerable"] = derive_answerable(df, sig)
        vc = df["answerable"].value_counts(normalize=True)
        print("\n  answerable 파생 (FUNCTIONS×HTML 조합):",
              ", ".join(f"{k} {v:.1%}" for k, v in vc.items()))
        print("    ℹ no_source = 툴은 돌았으나 결과 없음 / no_tool = 툴 미호출")
        print("       D1·D2·D3 세부 구분과 T1·T2 구분은 어노테이션 answerable 필드 필요")

    # --- 어노테이션 부가 필드 보존
    for src, dst in [("needs_review", "needs_review"), ("confidence", "confidence"),
                     ("secondary", "secondary")]:
        if src in df.columns:
            if dst == "secondary":
                df[dst] = df[src].map(parse_list)
            elif dst == "needs_review":
                df[dst] = df[src].map(
                    lambda v: str(v).strip().lower() in ("true", "1", "y", "yes"))
            else:
                df[dst] = pd.to_numeric(df[src], errors="coerce")

    # --- 필수 컬럼 점검 (재어노테이션에서 와야 하는 것들)
    for c in ("l1_stage", "l2_intent"):
        if c in df.columns:
            n_na = int(df[c].isna().sum())
            if n_na:
                print(f"  ⚠ {c} 결측 {n_na:,}행 ({n_na/len(df):.1%}) — "
                      "어노테이션 실패분. 의도별 분석에서 제외됩니다")

    need = ["l1_stage", "l2_intent", "f1_target_type", "f2_tense", "f3_personal",
            "f4_compliance", "f5_response", "f6_turn", "answerable", "outcome"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"\n  ❌ 재어노테이션 컬럼 누락: {miss}")
        print("     ANNOTATION_ALIASES 에 실제 컬럼명을 매핑하십시오.")
        return 1

    if "query_id" not in df.columns:
        df["query_id"] = df.index.astype(str)
    if "tool_expected" not in df.columns:
        df["tool_expected"] = pd.NA
    if "sample_stratum" not in df.columns:
        df["sample_stratum"] = "all"
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
        print("  ℹ sample_weight 없음 → 1.0. 층화 표집했다면 반드시 채우십시오.")

    keep = ["query_id", "session_id", "user_id", "ts", "latency_ms",
            "sample_stratum", "sample_weight", "slot_target", "source_expected",
            "tool_expected"] + need
    opt = [c for c in ["intent_pred", "tool_called", "response_len", "cited",
                       "csat", "intent_pred_group", "query_text", "answer_text",
                       "overblock",
                       "halluc_audit", "needs_review", "confidence", "secondary",
                       "tool_args", "tool_query", "slot_period", "slot_metric",
                       "slot_sort", "slot_count", "tool_steps", "tool_brief",
                       "protector_flag"]
           if c in df.columns]
    cols = list(dict.fromkeys(keep + opt))          # 순서 유지 중복 제거
    saved = df.loc[:, ~df.columns.duplicated()][cols]
    assert "answer_text" in saved.columns, "answer_text 누락 — 버전 확인 필요"
    assert not saved.columns.duplicated().any(), "중복 컬럼 발생"
    saved.to_pickle(out / "queries.pkl")
    print(f"   보존된 선택 컬럼: {', '.join(opt)}")
    import json as _json
    _derived.update({
        "protector_col": pcol,
        "slot_filled": float(df["slot_target"].map(
            lambda v: isinstance(v, (list, tuple)) and len(v) > 0).mean()),
        "rows": int(len(df)), "version": __version__})
    (out / "prep_meta.json").write_text(
        _json.dumps(_derived, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ {out/'queries.pkl'} 저장 ({len(df):,}행)")
    print("   리스트 컬럼 보존을 위해 pickle 로 저장합니다.")

    for name, cols in [("orders", ["user_id", "ticker", "ts", "order_amt", "filled"]),
                       ("app_views", ["user_id", "ticker", "ts", "channel"]),
                       ("app_sessions", ["user_id", "date"])]:
        p = out / f"{name}.pkl"
        if not p.exists():
            pd.DataFrame(columns=cols).to_pickle(p)
            print(f"   {p.name} 빈 스텁 생성 (C트랙 비활성)")

    print(f"\n다음: python run_analysis.py --data {out} --out ./out \\")
    print(f"        --outage YYYY-MM-DD --protector YYYY-MM-DD "
          f"--end {df['ts'].max():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
