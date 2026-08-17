"""
사용자가 실제로 알고 싶었던 것.

'삼성전자 주가 알려줘'에 70,100원이라고 답하면 관련성은 만점이다.
그런데 사용자가 정말 알고 싶었던 것은 대개 '왜 그렇게 움직였나',
'앞으로 어떨까', '내가 산 가격보다 오른 건가'다.

표면 질의(surface query)와 근본 의도(underlying need)는 다르다.
지금까지의 관련성·구조 지표는 전자만 재고 있었다.

여기서는 **어떤 질문 뒤에 무엇이 따라오는지**로 근본 의도를 역산한다.
후속 질문이 곧 '첫 질문이 채워주지 못한 것'이다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import INTENT_TO_STAGE, label_ko
from .turns import BAD_KINDS


def _slot_key(v) -> str:
    if isinstance(v, (list, tuple, set)):
        return "|".join(sorted(str(x) for x in v))
    return "" if pd.isna(v) else str(v)


def prepare(fu: pd.DataFrame, gap_min: int = 15,
            same_target_only: bool = True) -> pd.DataFrame:
    """
    후속 분석용 전처리.

    ★ 세 가지를 걸러야 '심화'만 남는다.
      · 실패 후 후속은 복구 시도이지 심화가 아니다 → 성공 건만
      · 대상이 바뀌면 새 질문이다 → 같은 slot 유지 건만 (슬롯 있을 때)
      · 되묻기(REPEAT/FORMAT)는 심화가 아니다 → 제외
    """
    d = fu.sort_values(["session_id", "ts"]).copy()
    g = d.groupby("session_id")
    d["_slot"] = d["slot_target"].map(_slot_key)
    d["_n_intent"] = g["l2_intent"].shift(-1)
    d["_n_stage"] = g["l1_stage"].shift(-1)
    d["_n_slot"] = g["_slot"].shift(-1)
    d["_n_kind"] = g["turn_kind"].shift(-1)
    d["_n_ts"] = g["ts"].shift(-1)
    d["_p_stage"] = g["l1_stage"].shift(1)
    d["_ok"] = d["outcome"].eq("success").fillna(False)

    within = (d["_n_ts"] - d["ts"]).dt.total_seconds() <= gap_min * 60
    d["_has_next"] = d["_n_intent"].notna() & within.fillna(False)
    d["_deepen"] = (d["_has_next"]
                    & ~d["_n_kind"].isin(BAD_KINDS).fillna(False)
                    & d["_n_intent"].ne(d["l2_intent"]))
    if same_target_only:
        slot_ok = d["_slot"].eq(d["_n_slot"]) | d["_slot"].eq("")
        d["_deepen"] &= slot_ok.fillna(True)
    return d


# ═══════════════════════════ 1. 후속 의도 분포와 lift

def followup_matrix(prep: pd.DataFrame, min_n: int = 60,
                    top: int = 4) -> pd.DataFrame:
    """
    성공한 질문 뒤에 따라온 다음 질문. 기저율 대비 lift 로 본다.

    lift 를 쓰는 이유: 어떤 의도든 뉴스 조회가 흔하면 그건 특성이 아니다.
    P(다음=Y | 현재=X) ÷ P(다음=Y) 여야 X 고유의 후속이 드러난다.
    """
    d = prep[prep["_ok"] & prep["_deepen"]]
    if len(d) < min_n:
        return pd.DataFrame({"안내": ["심화 후속 표본이 부족합니다"]})
    base = d["_n_intent"].value_counts(normalize=True)
    rows = []
    for src, g in d.groupby("l2_intent"):
        if len(g) < min_n:
            continue
        share = g["_n_intent"].value_counts(normalize=True)
        for tgt, p in share.head(top).items():
            rows.append({"질문": src, "이어서 묻는 것": tgt, "비율": float(p),
                         "lift": float(p / base.get(tgt, np.nan)),
                         "건수": int(len(g))})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"안내": ["집계 가능한 의도가 없습니다"]})
    return out.sort_values(["질문", "lift"], ascending=[True, False]).round(3)


def underlying_need(prep: pd.DataFrame, min_n: int = 60,
                    min_lift: float = 1.3, top: int = 3) -> pd.DataFrame:
    """
    의도별 '실제로 알고 싶었던 것' 요약.
    lift 가 높은 후속만 남겨 한 줄로 묶는다.
    """
    fm = followup_matrix(prep, min_n, top=8)
    if "안내" in fm.columns:
        return fm
    sig = fm[fm["lift"] >= min_lift]
    rows = []
    for src, g in sig.groupby("질문"):
        g = g.nlargest(top, "lift")
        rows.append({
            "질문": src,
            "표면 질문": label_ko(src),
            "이어서 묻는 것": " · ".join(
                f"{label_ko(t)}({p:.0%}, {l:.1f}배)"
                for t, p, l in zip(g["이어서 묻는 것"], g["비율"], g["lift"])),
            "건수": int(g["건수"].iloc[0]),
        })
    out = pd.DataFrame(rows)
    return out if len(out) else pd.DataFrame({"안내": ["유의한 후속 패턴 없음"]})


# ═══════════════════════════ 2. 자기완결률과 연쇄 깊이

def self_contained(prep: pd.DataFrame, min_n: int = 60) -> pd.DataFrame:
    """
    자기완결률 = 성공한 뒤 더 묻지 않고 끝난 비율
    연쇄 깊이   = 그 질문에서 시작해 이어진 심화 질문 수

    ★ 되묻기(실패 복구)와 다르다. 이건 '답은 맞았는데 부족해서 더 묻는' 것이다.
    """
    d = prep[prep["_ok"]].copy()
    if len(d) < min_n:
        return pd.DataFrame({"안내": ["표본 부족"]})
    d["_self"] = ~d["_deepen"]
    d["_bad"] = d["_n_kind"].isin(BAD_KINDS).fillna(False)

    # 연쇄 깊이: 세션 내에서 심화가 연속으로 이어진 길이
    dd = prep.sort_values(["session_id", "ts"]).copy()
    dd["_chain"] = 0
    chain = []
    cur = 0
    prev_sid = None
    for sid, dp in zip(dd["session_id"], dd["_deepen"].fillna(False)):
        if sid != prev_sid:
            cur = 0
            prev_sid = sid
        cur = cur + 1 if dp else 0
        chain.append(cur)
    dd["_chain_len"] = chain
    depth = (dd.groupby(["session_id", "l2_intent"])["_chain_len"].max()
               .groupby("l2_intent").mean().rename("연쇄깊이"))

    g = d.groupby("l2_intent").agg(
        건수=("_self", "size"), 자기완결률=("_self", "mean"),
        되묻기율=("_bad", "mean"))
    g = g[g["건수"] >= min_n].join(depth)
    g.insert(0, "질문", [label_ko(i) for i in g.index])
    return g.sort_values("자기완결률").round(3)


def need_quadrant(sc: pd.DataFrame) -> pd.DataFrame:
    """
    되묻기(실패) × 연쇄(심화) 4사분면.

      되묻기 적음 + 연쇄 짧음 → 잘 답함
      되묻기 적음 + 연쇄 김   → ★ 한 번에 못 채움 (응답 설계 과제)
      되묻기 많음 + 연쇄 짧음 → 답을 못 함
      되묻기 많음 + 연쇄 김   → 아예 붕괴
    """
    if "안내" in sc.columns or "연쇄깊이" not in sc.columns:
        return sc
    d = sc.dropna(subset=["연쇄깊이"]).copy()
    if d.empty:
        return pd.DataFrame({"안내": ["연쇄 깊이 산출 불가"]})
    bm, cm = d["되묻기율"].median(), d["연쇄깊이"].median()
    d["구분"] = np.select(
        [(d["되묻기율"] < bm) & (d["연쇄깊이"] >= cm),
         (d["되묻기율"] >= bm) & (d["연쇄깊이"] >= cm),
         (d["되묻기율"] >= bm)],
        ["★한 번에 못 채움", "붕괴", "답을 못 함"], default="잘 답함")
    return d.sort_values(["구분", "연쇄깊이"], ascending=[True, False]).round(3)


# ═══════════════════════════ 3. 맥락 조건부

def context_conditional(prep: pd.DataFrame, intent: str | None = None,
                        min_n: int = 40, top: int = 3) -> pd.DataFrame:
    """
    같은 질문이라도 **여정 어디에서 나왔느냐**에 따라 원하는 것이 다르다.
    앞 단계별로 후속 분포를 나눠 본다.

    이 결과는 곧 '응답을 맥락에 따라 다르게 구성해야 한다'는 근거가 된다.
    """
    d = prep[prep["_ok"] & prep["_deepen"] & prep["_p_stage"].notna()]
    if intent:
        d = d[d["l2_intent"].eq(intent)]
    if len(d) < min_n:
        return pd.DataFrame({"안내": ["맥락 조건부 표본이 부족합니다"]})
    rows = []
    for (src, ps), g in d.groupby(["l2_intent", "_p_stage"]):
        if len(g) < min_n:
            continue
        share = g["_n_intent"].value_counts(normalize=True).head(top)
        rows.append({
            "질문": label_ko(src), "직전 단계": label_ko(ps), "건수": len(g),
            "이어서 묻는 것": " · ".join(f"{label_ko(t)}({p:.0%})"
                                          for t, p in share.items())})
    out = pd.DataFrame(rows)
    return out if len(out) else pd.DataFrame({"안내": ["조건부 셀 표본 부족"]})


# ═══════════════════════════ 4. 충족률 (설정 기반)

NEEDS_TEMPLATE_NOTE = [
    "의도별로 '응당 함께 나가야 할 정보'를 정의하면 충족률이 계산됩니다.",
    "권장항목: 항목명과 응답에서 찾을 키워드(정규식 가능)를 적으십시오.",
    "아래 seed 는 실제 후속 질문 분포에서 자동 제안된 것입니다.",
    "필요 없는 항목은 지우고, 빠진 항목은 추가하십시오.",
]


def ensure_needs_config(path: Path, sc: pd.DataFrame,
                        need: pd.DataFrame) -> dict:
    """
    설정 파일이 없으면 후속 분포를 근거로 초안을 만들어 둔다.
    사람이 채우면 다음 실행부터 충족률이 계산된다.
    """
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    seed = {"_안내": NEEDS_TEMPLATE_NOTE, "권장항목": {}}
    src = need if "질문" in getattr(need, "columns", []) else None
    if src is not None:
        for _, r in src.head(15).iterrows():
            seed["권장항목"][str(r["질문"])] = {
                "_표면질문": str(r.get("표면 질문", "")),
                "_관측된후속": str(r.get("이어서 묻는 것", "")),
                "항목": [{"이름": "", "키워드": ""}],
            }
    path.write_text(json.dumps(seed, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return {}


def fulfillment(q: pd.DataFrame, cfg: dict, min_n: int = 30) -> pd.DataFrame:
    """
    응답에 권장 항목이 실제로 담겼는지. 설정이 비어 있으면 건너뛴다.
    """
    spec = (cfg or {}).get("권장항목", {})
    usable = {k: v for k, v in spec.items()
              if any(str(i.get("키워드", "")).strip()
                     for i in v.get("항목", []))}
    if not usable:
        return pd.DataFrame({"안내": [
            "needs_config.json 의 권장항목 키워드가 비어 있습니다 — "
            "채우면 충족률이 계산됩니다"]})
    txt = (q["answer_text"] if "answer_text" in q.columns
           else q.get("ANSWER", pd.Series("", index=q.index))).fillna("").astype(str)
    rows = []
    for intent, v in usable.items():
        sel = q["l2_intent"].eq(intent) & q["outcome"].eq("success")
        if sel.sum() < min_n:
            continue
        items = [i for i in v["항목"] if str(i.get("키워드", "")).strip()]
        hit = np.zeros(int(sel.sum()))
        detail = {}
        sub = txt[sel]
        for it in items:
            m = sub.str.contains(str(it["키워드"]), regex=True, na=False)
            hit += m.to_numpy(dtype=float)
            detail[it.get("이름") or it["키워드"]] = round(float(m.mean()), 3)
        rows.append({"질문": label_ko(intent), "건수": int(sel.sum()),
                     "충족률": float((hit / len(items)).mean()),
                     "항목별": detail})
    out = pd.DataFrame(rows)
    return out.sort_values("충족률").round(3) if len(out) else pd.DataFrame(
        {"안내": ["대상 표본 부족"]})


def fulfillment_validation(q: pd.DataFrame, prep: pd.DataFrame,
                           ful: pd.DataFrame) -> dict:
    """
    충족률이 실제로 결과를 예측하는지 검증.
    관계가 없으면 '정보를 더 준다고 만족이 늘지 않는다'는 뜻이며,
    그 경우 양이 아니라 형식의 문제다.
    """
    if "안내" in ful.columns or len(ful) < 4:
        return {"안내": "검증할 충족률 표본이 부족합니다"}
    sc = self_contained(prep)
    if "안내" in sc.columns:
        return {"안내": "자기완결률 산출 불가"}
    m = ful.merge(sc.reset_index()[["질문", "자기완결률", "되묻기율"]],
                  on="질문", how="inner")
    if len(m) < 4:
        return {"안내": "교집합 표본 부족"}
    return {"n": len(m),
            "충족률↔자기완결률": round(float(m["충족률"].corr(m["자기완결률"])), 3),
            "충족률↔되묻기율": round(float(m["충족률"].corr(m["되묻기율"])), 3),
            "해석": ("자기완결률과 양(+), 되묻기율과 음(−)의 관계가 나오면 "
                     "충족률을 응답 품질 지표로 쓸 수 있습니다. "
                     "관계가 없으면 양이 아니라 형식의 문제입니다.")}
