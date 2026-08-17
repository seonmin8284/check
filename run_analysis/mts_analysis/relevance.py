"""
질문–응답 관련성 프록시 · 응답 구조 신호.

형태소 분석기 없이 동작한다(폐쇄망 전제). 조사·어미를 규칙으로 떼고
내용어를 뽑은 뒤, 질문 내용어가 응답에 얼마나 닿았는지를 본다.

★ 이 지표는 '관련성의 근사'이지 판정이 아니다.
  1000건 인적 평가로 캘리브레이션한 뒤에야 전체 로그에 확장할 수 있다.
  (calibration.py 참조)
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# 조사·어미 — 긴 것부터 떼야 한다
_JOSA = sorted([
    "이라고", "라고는", "에서는", "에게서", "으로는", "이라는", "께서는",
    "에서도", "에게도", "부터는", "까지는", "보다는", "처럼도",
    "에서", "에게", "으로", "라고", "부터", "까지", "보다", "처럼", "한테",
    "이나", "이란", "이며", "이고", "하고", "에는", "에도", "께서", "만큼",
    "조차", "마저", "이든", "든지", "이랑", "라도", "밖에",
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만",
    "로", "랑", "야", "아", "여", "께", "든", "나",
], key=len, reverse=True)

# 내용 없는 흔한 어휘
_STOP = {
    "알려줘", "알려주세요", "뭐야", "뭔가요", "어때", "어떻게", "어디", "언제",
    "얼마", "무엇", "그리고", "그런데", "하지만", "관련", "정보", "내용", "대해",
    "대한", "해줘", "해주세요", "있나", "있어", "인가요", "입니까", "인지",
    "좀", "저기", "그거", "이거", "저거", "요즘", "지금", "오늘", "제발",
}

_HANGUL = re.compile(r"[가-힣]+")
_ALNUM = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d[\d,.]*%?")
_TAG = re.compile(r"<[^>]+>")
_SENT = re.compile(r"[.!?。\n]+")


def _strip_josa(w: str) -> str:
    for j in _JOSA:
        if len(w) > len(j) + 1 and w.endswith(j):
            return w[: -len(j)]
    return w


def content_tokens(text: str, min_len: int = 2) -> set[str]:
    """한글 어절에서 조사를 떼고 내용어 후보를 뽑는다. 영문·숫자는 그대로."""
    if not isinstance(text, str) or not text.strip():
        return set()
    t = unicodedata.normalize("NFKC", text)
    out: set[str] = set()
    for w in _HANGUL.findall(t):
        w = _strip_josa(w)
        if len(w) >= min_len and w not in _STOP:
            out.add(w)
    for w in _ALNUM.findall(t):
        if len(w) >= min_len:
            out.add(w.lower())
    return out


def char_ngrams(text: str, n: int = 2) -> set[str]:
    if not isinstance(text, str):
        return set()
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", _TAG.sub(" ", text)).strip()


def relevance_scores(question: str, answer_html: str) -> dict:
    """
    질문–응답 관련성 프록시 3종.

      coverage  : 질문 내용어 중 응답에 등장한 비율 (재현율 성격) ← 주 지표
      jaccard   : 내용어 집합 자카드
      bigram    : 문자 bigram 자카드 (형태 변형에 강함)

    무관한 응답(예: 분류 실패 후 뉴스 벡터DB 결과)은 coverage 가 낮게 나온다.
    """
    a_txt = strip_html(answer_html)
    qt, at = content_tokens(question), content_tokens(a_txt)
    if not qt:
        return {"coverage": np.nan, "jaccard": np.nan, "bigram": np.nan,
                "q_terms": 0, "hit_terms": 0, "missed": ""}
    hit = qt & at
    union = qt | at
    qb, ab = char_ngrams(question), char_ngrams(a_txt)
    return {
        "coverage": len(hit) / len(qt),
        "jaccard": len(hit) / len(union) if union else np.nan,
        "bigram": len(qb & ab) / len(qb | ab) if (qb | ab) else np.nan,
        "q_terms": len(qt),
        "hit_terms": len(hit),
        "missed": "|".join(sorted(qt - at)[:8]),
    }


def add_relevance(q: pd.DataFrame, question_col: str = "query_text",
                  answer_col: str = "answer_text") -> pd.DataFrame:
    """관련성 프록시 컬럼을 붙인다. answer_text 가 없으면 ANSWER 원문을 쓴다."""
    d = q.copy()
    if answer_col not in d.columns or d[answer_col].isna().all():
        answer_col = ("ANSWER" if "ANSWER" in d.columns
                      and not d["ANSWER"].isna().all() else None)
    if (question_col not in d.columns or d[question_col].isna().all()
            or answer_col is None):
        d[["rel_coverage", "rel_jaccard", "rel_bigram"]] = np.nan
        return d
    rows = [relevance_scores(qq, aa)
            for qq, aa in zip(d[question_col].fillna(""), d[answer_col].fillna(""))]
    r = pd.DataFrame(rows, index=d.index)
    d["rel_coverage"] = r["coverage"]
    d["rel_jaccard"] = r["jaccard"]
    d["rel_bigram"] = r["bigram"]
    d["rel_q_terms"] = r["q_terms"]
    d["rel_missed"] = r["missed"]
    return d


# ------------------------------------------------------- 응답 구조 신호

def structure_signals(answer_html: str) -> dict:
    """
    '단순 데이터 출력'인지 '해석이 붙은 답'인지 구조로 판별한다.

    가설: 표는 있는데 요약 문장이 없는 응답이 가장 나쁘다.
          사용자가 스스로 읽고 판단해야 하기 때문.
    """
    if not isinstance(answer_html, str):
        answer_html = ""
    h = answer_html
    txt = strip_html(h)
    toks = txt.split()
    n_tok = max(len(toks), 1)
    n_num = sum(bool(re.search(r"\d", t)) for t in toks)

    sents = [s.strip() for s in _SENT.split(txt) if len(s.strip()) >= 8]
    # 해석 문장 = 숫자가 없는 완결 문장 (수치 나열이 아닌 서술)
    interp = [s for s in sents if not re.search(r"\d", s)]
    lead = sents[0] if sents else ""

    return {
        "표": len(re.findall(r"<\s*table", h, re.I)),
        "행수": len(re.findall(r"<\s*tr", h, re.I)),
        "목록항목": len(re.findall(r"<\s*li", h, re.I)),
        "제목태그": len(re.findall(r"<\s*h[1-6]", h, re.I)),
        "강조": len(re.findall(r"<\s*(b|strong|em)\b", h, re.I)),
        "본문길이": len(txt),
        "수치밀도": n_num / n_tok,
        "문장수": len(sents),
        "해석문장수": len(interp),
        "요약선행문": bool(lead and not re.search(r"\d", lead) and len(lead) >= 15),
    }


def add_structure(q: pd.DataFrame, answer_col: str = "answer_text") -> pd.DataFrame:
    d = q.copy()
    if answer_col not in d.columns or d[answer_col].isna().all():
        answer_col = "ANSWER"
    if answer_col not in d.columns or d[answer_col].isna().all():
        return d
    s = pd.DataFrame([structure_signals(a) for a in d[answer_col].fillna("")],
                     index=d.index)
    for c in s.columns:
        d[f"st_{c}"] = s[c]
    # 유형 분류: 데이터 나열 vs 해석 포함
    d["응답유형"] = np.select(
        [(d["st_표"] > 0) & ~d["st_요약선행문"],
         (d["st_표"] > 0) & d["st_요약선행문"],
         (d["st_해석문장수"] >= 2)],
        ["표만(해석 없음)", "표+요약", "서술형"], default="단문·기타")
    return d


def structure_impact(q: pd.DataFrame, fu: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    응답 유형별로 후속 행동이 어떻게 달라지는가.
    인과는 아니지만 '구조화 부재의 영향'을 보여주는 가장 직접적인 근거.
    """
    d = q.copy()
    if "응답유형" not in d.columns:
        return pd.DataFrame({"안내": ["add_structure 를 먼저 적용하십시오"]})
    if fu is not None and "turn_kind" in fu.columns:
        d = d.merge(fu[["query_id", "turn_kind"]], on="query_id", how="left")
    d = d[d["outcome"].eq("success")]
    if d.empty:
        return pd.DataFrame({"안내": ["성공 응답 없음"]})

    agg = {"n": ("응답유형", "size"),
           "본문길이_중앙": ("st_본문길이", "median"),
           "수치밀도": ("st_수치밀도", "mean")}
    if "turn_kind" in d.columns:
        d["_bad_next"] = d["turn_kind"].isin(["REPEAT", "FORMAT"]).fillna(False)
        agg["복구성후속률"] = ("_bad_next", "mean")
    if "eff_kind" in d.columns:
        d["_eff"] = d["eff_kind"].eq("EFFECTIVE")
        d["_skim"] = d["eff_kind"].eq("SKIMMED")
        agg["실질성공률"] = ("_eff", "mean")
        agg["훑고넘김률"] = ("_skim", "mean")
    if "rel_coverage" in d.columns:
        agg["관련성_coverage"] = ("rel_coverage", "mean")
    return d.groupby("응답유형").agg(**agg).round(3)


# ------------------------------------------------------- OTH 폴백 진단

# 분류 실패 코드 — 실로그에서 확인된 값 포함. 실제 코드가 다르면 인자로 넘기십시오.
OTH_CODES = ("OTHER", "OTH", "OTHERS", "ETC", "기타",
             "UNKNOWN", "NONE", "NA", "OOD", "FALLBACK")
# 폴백 함수명 — 부분일치로 검사한다. 어순이 바뀌어도 잡히도록 양쪽 다 둔다.
FALLBACK_TOOLS = ("work_and_news", "news_and_work",
                  "get_work_and_news", "get_news_and_work")


def oth_fallback_diagnosis(q: pd.DataFrame, oth_codes=OTH_CODES,
                           fallback_tools=FALLBACK_TOOLS) -> dict:
    """
    분류 실패(OTH) → 폴백 함수 강제 호출 구조의 실제 규모와 피해를 잰다.

    이 구조에서는 분류 실패가 '실패'로 관측되지 않는다.
    무조건 응답이 나오고 렌더도 되므로 outcome=success 로 집계된다.
    → 지금까지의 성공률은 이 비중만큼 부풀려져 있다.
    """
    d = q.copy()
    is_oth = pd.Series(False, index=d.index)
    if "intent_pred" in d.columns:
        is_oth = d["intent_pred"].astype(str).str.upper().isin(
            [c.upper() for c in oth_codes])
    is_fb = pd.Series(False, index=d.index)
    if "tool_called" in d.columns:
        tc = d["tool_called"].fillna("").astype(str).str.lower()
        is_fb = tc.apply(lambda s: any(f in s for f in fallback_tools))
    d["_oth"] = is_oth
    d["_fb"] = is_fb
    d["_silent_ok"] = (is_oth | is_fb) & d["outcome"].eq("success")

    res = {"OTH 비중": round(float(is_oth.mean()), 4),
           "폴백툴 호출 비중": round(float(is_fb.mean()), 4),
           "조용한 성공 비중": round(float(d["_silent_ok"].mean()), 4)}

    # ★ 아무것도 안 걸리면 코드값이 기본 목록과 다른 것이다.
    #   추측하지 말고 실제 값 분포를 보여준다.
    if not is_oth.any() and "intent_pred" in d.columns:
        top = d["intent_pred"].dropna().astype(str).value_counts().head(12)
        res["⚠ 분류실패코드 미탐지"] = (
            "intent_pred 값이 기본 목록과 일치하지 않습니다. "
            "아래에서 분류 실패에 해당하는 코드를 골라 oth_codes 로 넘기십시오.")
        res["intent_pred 상위값"] = top.to_dict()
    if not is_fb.any() and "tool_called" in d.columns:
        top = d["tool_called"].dropna().astype(str).value_counts().head(12)
        res["⚠ 폴백툴 미탐지"] = (
            "tool_called 에서 폴백 함수를 찾지 못했습니다. "
            "아래 함수명 중 폴백에 해당하는 것을 fallback_tools 로 넘기십시오.")
        res["tool_called 상위값"] = top.to_dict()

    if "rel_coverage" in d.columns and d["rel_coverage"].notna().any():
        grp = d.groupby(d["_oth"] | d["_fb"])["rel_coverage"]
        res["관련성 coverage (폴백)"] = round(float(grp.mean().get(True, np.nan)), 4)
        res["관련성 coverage (정상)"] = round(float(grp.mean().get(False, np.nan)), 4)
        res["coverage 격차"] = round(
            res["관련성 coverage (정상)"] - res["관련성 coverage (폴백)"], 4)

    if "turn_kind" in d.columns:
        b = d.groupby(d["_oth"] | d["_fb"])["turn_kind"].apply(
            lambda x: float(x.isin(["REPEAT", "FORMAT"]).mean()))
        res["복구성 후속률 (폴백/정상)"] = (round(float(b.get(True, np.nan)), 4),
                                            round(float(b.get(False, np.nan)), 4))

    adj = None
    if res["조용한 성공 비중"] > 0:
        raw = float(d["outcome"].eq("success").mean())
        adj = raw - res["조용한 성공 비중"]
        res["보고 성공률"] = round(raw, 4)
        res["폴백 제외 성공률(하한)"] = round(adj, 4)

    res["해석"] = (
        "폴백 강제 호출 구조에서는 분류 실패가 성공으로 집계됩니다. "
        "'폴백 제외 성공률'은 폴백 건을 전부 실패로 본 하한이고, 실제 값은 "
        "그 사이 어딘가입니다 — 1000건 인적 평가로 좁히십시오.")
    return res


# ------------------------------------------------------- 질의 왜곡 (재작성 쿼리)

def query_drift(q: pd.DataFrame, oth_codes=OTH_CODES,
                fallback_tools=FALLBACK_TOOLS) -> dict:
    """
    사용자 원 질문 vs 툴에 실제로 전달된 쿼리(tool_query)의 어휘 보존율.

    폴백 함수는 쿼리를 재작성해 벡터DB를 조회한다. 그 재작성 과정에서
    질문의 핵심어가 탈락하면, 이후 응답이 무관해지는 것은 예정된 결과다.
    **관련성 실패의 원인이 검색 단계인지 생성 단계인지**를 가르는 지점.
    """
    if "tool_query" not in q.columns or q["tool_query"].isna().all():
        return {"안내": "tool_query 없음 — prep_data v1.5.0 이상으로 재실행하십시오"}
    if "query_text" not in q.columns or q["query_text"].isna().all():
        return {"안내": "query_text 없음"}

    d = q[q["tool_query"].notna() & q["query_text"].notna()].copy()
    if d.empty:
        return {"안내": "대조 가능한 행 없음"}

    d["보존율"] = [relevance_scores(a, b)["coverage"]
                   for a, b in zip(d["query_text"], d["tool_query"])]

    tc = d["tool_called"].fillna("").astype(str).str.lower()
    is_fb = tc.apply(lambda s_: any(f in s_ for f in fallback_tools))
    if "intent_pred" in d.columns:
        is_fb = is_fb | d["intent_pred"].astype(str).str.upper().isin(
            [c.upper() for c in oth_codes])

    res = {"대상건수": int(len(d)),
           "전체 보존율(중앙)": round(float(d["보존율"].median()), 4),
           "보존율 0.5 미만 비중": round(float(d["보존율"].lt(.5).mean()), 4)}
    if is_fb.any() and (~is_fb).any():
        res["보존율 (폴백)"] = round(float(d.loc[is_fb, "보존율"].mean()), 4)
        res["보존율 (정상)"] = round(float(d.loc[~is_fb, "보존율"].mean()), 4)
        res["격차"] = round(res["보존율 (정상)"] - res["보존율 (폴백)"], 4)

    if "rel_coverage" in d.columns and d["rel_coverage"].notna().any():
        # 검색 단계 손실 vs 생성 단계 손실 분해
        keep = d["보존율"]
        final = d["rel_coverage"]
        res["질문→쿼리 손실(검색 단계)"] = round(float(1 - keep.mean()), 4)
        res["쿼리→응답 손실(생성 단계)"] = round(
            float((keep - final).clip(lower=0).mean()), 4)

    res["해석"] = ("보존율이 낮으면 무관 응답의 책임은 생성이 아니라 "
                   "**쿼리 재작성**에 있습니다. 그 경우 모델 교체로는 해결되지 않고 "
                   "재작성 로직을 손봐야 합니다.")
    return res


def slot_by_intent(q: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """
    의도별 슬롯 복원율. 툴 호출 인자에서 대상·기간이 실제로 잡히는지 본다.
    target 이 안 잡히는 의도는 재질문(REPEAT) 판정 정확도도 함께 떨어진다.
    """
    d = q.copy()
    d["_tgt"] = d["slot_target"].map(
        lambda v: isinstance(v, (list, tuple)) and len(v) > 0)
    agg = {"n": ("_tgt", "size"), "target복원율": ("_tgt", "mean")}
    for c, lbl in [("slot_period", "period"), ("slot_metric", "metric"),
                   ("tool_query", "query")]:
        if c in d.columns:
            d[f"_{lbl}"] = d[c].notna()
            agg[f"{lbl}복원율"] = (f"_{lbl}", "mean")
    g = d.groupby("l2_intent").agg(**agg)
    return g[g["n"] >= min_n].sort_values("target복원율").round(3)
