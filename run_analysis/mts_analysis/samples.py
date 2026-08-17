"""
측정 지점별 원본 데이터 샘플.

집계 숫자만으로는 '그 판정이 맞는지' 확인할 수 없다.
각 측정 지점에서 실제 질문·응답·툴 호출을 함께 보여 주면,
설계가 의도대로 동작하는지 눈으로 검증할 수 있다.

  · 실패 귀속 코드가 정말 그 원인인가
  · 폴백으로 잡힌 건이 정말 무관한 답인가
  · 되묻기로 분류된 턴이 정말 재질문인가
  · '한 번에 못 채운' 질문이 정말 그런가

각 지점의 표본은 out/samples/*.csv 로도 저장된다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .relevance import strip_html
from .schema import label_ko

BASE_COLS = ["ts", "session_id", "user_id", "l2_intent", "f4_compliance",
             "outcome", "fail_code", "tool_called", "tool_steps"]


def _short(v, n: int = 90) -> str:
    t = strip_html(str(v)) if isinstance(v, str) else ""
    t = " ".join(t.split())
    return t[:n] + ("…" if len(t) > n else "")


def view(q: pd.DataFrame, idx=None, extra: list[str] | None = None,
         qlen: int = 70, alen: int = 110) -> pd.DataFrame:
    """원본 행을 사람이 읽을 수 있는 형태로."""
    d = q.loc[idx] if idx is not None else q
    out = pd.DataFrame(index=d.index)
    out["시각"] = pd.to_datetime(d["ts"]).dt.strftime("%m-%d %H:%M")
    out["질문"] = (d["query_text"].map(lambda v: _short(v, qlen))
                   if "query_text" in d.columns else "")
    out["응답"] = (d["answer_text"].map(lambda v: _short(v, alen))
                   if "answer_text" in d.columns else "")
    out["의도"] = d["l2_intent"].map(label_ko)
    if "f4_compliance" in d.columns:
        out["등급"] = d["f4_compliance"]
    out["결과"] = d["outcome"]
    if "fail_code" in d.columns:
        out["원인"] = d["fail_code"].fillna("")
    if "tool_called" in d.columns:
        out["툴"] = d["tool_called"].fillna("(없음)").map(lambda v: _short(v, 40))
    for c in (extra or []):
        if c in d.columns:
            out[c] = d[c]
    return out


def by_category(q: pd.DataFrame, col: str, n_per: int = 3,
                extra: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    """범주별로 n_per 건씩 뽑는다. 판정이 범주 정의와 맞는지 확인용."""
    if col not in q.columns:
        return pd.DataFrame({"안내": [f"{col} 컬럼 없음"]})
    rng = np.random.default_rng(seed)
    parts = []
    for k, g in q[q[col].notna()].groupby(col):
        take = min(n_per, len(g))
        idx = rng.choice(g.index, take, replace=False)
        v = view(q, idx, extra)
        v.insert(0, "구분", str(k))
        parts.append(v)
    if not parts:
        return pd.DataFrame({"안내": ["표본 없음"]})
    return pd.concat(parts).reset_index(drop=True)


def extremes(q: pd.DataFrame, col: str, n: int = 3,
             extra: list[str] | None = None) -> pd.DataFrame:
    """수치 컬럼의 양극단. 지표가 실제로 그 현상을 잡는지 확인용."""
    if col not in q.columns:
        return pd.DataFrame({"안내": [f"{col} 없음"]})
    d = q[q[col].notna()]
    if d.empty:
        return pd.DataFrame({"안내": ["값 없음"]})
    lo = view(q, d.nsmallest(n, col).index, (extra or []) + [col])
    lo.insert(0, "구분", f"{col} 하위")
    hi = view(q, d.nlargest(n, col).index, (extra or []) + [col])
    hi.insert(0, "구분", f"{col} 상위")
    return pd.concat([lo, hi]).reset_index(drop=True)


def transcripts(fu: pd.DataFrame, session_ids, max_turns: int = 8) -> str:
    """세션 대화를 그대로 재현. 흐름 판정(세션 결과·되묻기) 검증에 가장 유용."""
    lines = []
    for sid in session_ids:
        g = fu[fu["session_id"].eq(sid)].sort_values("ts")
        if g.empty:
            continue
        lines.append(f"── 세션 {sid} ({len(g)}턴) " + "─" * 40)
        for i, (_, r) in enumerate(g.head(max_turns).iterrows(), 1):
            kind_val = r.get("turn_kind")
            kind = "" if pd.isna(kind_val) else str(kind_val)
            mark = f" [{kind}]" if kind and kind != "<NA>" else ""
            lines.append(
                f" {i}. Q {_short(r.get('query_text'), 64)}"
                f" ({label_ko(r['l2_intent'])}{mark})")
            lines.append(
                f" A {_short(r.get('answer_text'), 96)}"
                f" → {r['outcome']}"
                + (f"/{r['fail_code']}" if pd.notna(r.get("fail_code")) else ""))
        if len(g) > max_turns:
            lines.append(f" … 이하 {len(g)-max_turns}턴 생략")
        lines.append("")
    return "\n".join(lines)


def pick_sessions(sess: pd.DataFrame, outcome: str, n: int = 2,
                  seed: int = 0) -> list:
    d = sess[sess["session_outcome"].eq(outcome)]
    if d.empty:
        return []
    rng = np.random.default_rng(seed)
    return list(rng.choice(d.index, min(n, len(d)), replace=False))


# ═══════════════════════════ 일괄 저장

def dump_all(q: pd.DataFrame, fu: pd.DataFrame, sess: pd.DataFrame,
             out_dir: Path, eff: pd.DataFrame | None = None,
             n_per: int = 5) -> dict:
    """모든 측정 지점의 표본을 out/samples/ 로 저장."""
    sd = out_dir / "samples"
    sd.mkdir(parents=True, exist_ok=True)
    saved = {}

    def _w(name, df):
        if df is None or len(df) == 0 or "안내" in getattr(df, "columns", []):
            return
        df.to_csv(sd / f"{name}.csv", index=False, encoding="utf-8-sig")
        saved[name] = len(df)

    _w("실패귀속코드별", by_category(q, "fail_code", n_per))
    _w("컴플라이언스등급별", by_category(q, "f4_compliance", n_per))
    _w("여정단계별", by_category(q, "l1_stage", n_per))
    if "응답유형" in q.columns:
        _w("응답유형별", by_category(q, "응답유형", n_per,
                                     extra=["st_수치밀도", "st_해석문장수"]))
    if "차단유형" in q.columns:
        _w("차단유형별", by_category(q, "차단유형", n_per))
    if "rel_coverage" in q.columns:
        _w("관련성_양극단", extremes(q, "rel_coverage", n_per,
                                     extra=["rel_missed"]))
    if "response_len" in q.columns:
        _w("응답길이_양극단", extremes(q, "response_len", n_per))
    if "latency_ms" in q.columns:
        _w("지연_양극단", extremes(q, "latency_ms", n_per))
    if "turn_kind" in fu.columns:
        _w("후속턴유형별", by_category(fu, "turn_kind", n_per))
    if eff is not None and "eff_kind" in eff.columns:
        _w("실질성공재분류별", by_category(eff, "eff_kind", n_per))
    if "intent_pred" in q.columns:
        mis = q[q["intent_pred"].notna() & q["l2_intent"].notna()]
        mis = mis[mis["intent_pred"].astype(str) != mis["l2_intent"].astype(str)]
        if len(mis):
            v = view(mis.sample(min(n_per * 3, len(mis)), random_state=0),
                     extra=["intent_pred"])
            _w("운영분류_불일치", v)

    # 세션 대화 전문
    tx = []
    for oc in ("RESOLVED", "RESOLVED_HARD", "ABANDONED", "DEFLECTED"):
        ids = pick_sessions(sess, oc, 2)
        if ids:
            tx.append(f"\n{'='*70}\n[{oc}] 세션 예시\n{'='*70}")
            tx.append(transcripts(fu, ids))
    if tx:
        (sd / "세션_대화전문.txt").write_text("\n".join(tx), encoding="utf-8")
        saved["세션_대화전문"] = len(tx)
    return saved


def followup_pairs(prep: pd.DataFrame, n: int = 8, seed: int = 0) -> pd.DataFrame:
    """
    '표면 질문 → 이어서 묻는 것' 쌍의 실제 예시.
    근본 의도 역산이 억지가 아닌지 확인하는 가장 직접적인 증거.
    """
    d = prep[prep["_ok"] & prep["_deepen"]]
    if d.empty:
        return pd.DataFrame({"안내": ["심화 후속 표본 없음"]})
    rng = np.random.default_rng(seed)
    idx = rng.choice(d.index, min(n, len(d)), replace=False)
    g = d.loc[idx]
    return pd.DataFrame({
        "처음 질문": [_short(v, 52) for v in g.get("query_text", "")],
        "→ 의도": [label_ko(i) for i in g["l2_intent"]],
        "받은 답": [_short(v, 60) for v in g.get("answer_text", "")],
        "이어서 물은 것": [label_ko(i) for i in g["_n_intent"]],
    }).reset_index(drop=True)
