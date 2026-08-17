"""
데이터 결핍 진단.

'데이터가 부족합니다'는 분석의 변명으로 들린다. 같은 내용을
"이 질문에 답하려면 무엇이 필요하고, 확보하면 무엇이 가능해지는가"로
바꾸면 의사결정 정보가 된다.

자동으로 판정하는 것 — 무엇이 없는가, 그래서 어떤 분석이 막혔는가,
                     지금 숫자의 불확실 구간은 얼마인가
사람이 채우는 것   — 확보 난이도·담당·일정, 이번에 틀릴 뻔했던 사례
                     (gaps_config.json 에 적으면 보고서에 반영된다)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# 결핍의 층위 — 성격이 다르면 해결 주체가 다르다
LAYERS = {
    "window": "관측 창 — 시간이 지나면 해결",
    "collect": "미수집 — 지금 로그에 남기지 않음",
    "join": "미연결 — 있는데 조인이 안 됨",
    "label": "미라벨 — 사람이 판정해야 함",
}

# 알려진 결핍 항목. 자동 판정 결과와 사용자 설정이 여기에 병합된다.
GAP_CATALOG = [
    dict(id="entry_path", 층="collect",
         질문="재방문율 하락의 나머지 몫은 무엇 때문인가",
         필요="진입 경로(배너·홈·검색 등) 기록",
         대안="유입 의도 구성 변화",
         한계="하락의 일부만 설명됨",
         확보시="하락 원인 규명, 획득 채널별 사용자 질 비교"),
    dict(id="app_session", 층="join",
         질문="사용자가 챗봇만 떠난 건가 서비스를 떠난 건가",
         필요="앱 전체 세션 로그 (user_id, date)",
         대안="챗봇 로그만",
         한계="챗봇 이탈을 고객 이탈로 과대 해석할 위험",
         확보시="이탈 심각도 판정, 개선 우선순위 재조정"),
    dict(id="orders", 층="join",
         질문="챗봇이 실제 거래를 늘리는가",
         필요="주문·체결 내역, 비챗봇 종목 조회 로그",
         대안="없음 (분석 자체가 빠짐)",
         한계="사업 기여를 전혀 제시할 수 없음",
         확보시="챗봇의 거래 기여를 금액으로 제시"),
    dict(id="answerable", 층="label",
         질문="실패 원인이 자료인가 기능인가 모델인가",
         필요="'답할 수 있었어야 하는가' 라벨",
         대안="함수 호출·렌더 구조로 추정",
         한계="자료·기능 세부 구분 불가, 상당수가 모델 오류로 몰림",
         확보시="개선 투자처 확정"),
    dict(id="slot", 층="label",
         질문="되묻기가 실제로 얼마나 되는가",
         필요="질문 대상(종목) 라벨",
         대안="질문 원문 유사도",
         한계="되묻기 판정 정확도 저하",
         확보시="되묻기 지표 확정, 거래 전환 분석 가능"),
    dict(id="quality_audit", 층="label",
         질문="답변이 실제로 질문에 맞는가",
         필요="표본 1,000건 인적 검수",
         대안="질문 핵심어가 답변에 담기는 비율",
         한계="분류 실패 응답을 잘 가려내지 못함",
         확보시="성공률 실측, 자동 지표 보정"),
    dict(id="cs_contact", 층="join",
         질문="안내형 답변이 실제로 문제를 해결했는가",
         필요="고객센터 유입 로그 (24시간 조인)",
         대안="챗봇 내 종료 여부",
         한계="잘못 안내해도 성공으로 집계",
         확보시="안내 품질 실측"),
    dict(id="release_log", 층="collect",
         질문="언제부터 무엇이 바뀌어 지표가 꺾였는가",
         필요="배포·프롬프트 변경 이력 (날짜, 변경 내용)",
         대안="지표 시계열의 변곡점 추정",
         한계="원인 후보를 좁힐 수 없음",
         확보시="지표 변화와 변경 이력 대조"),
    dict(id="protector_flag", 층="label",
         질문="차단이 사용자 여정을 어떻게 바꾸는가",
         필요="Protector 발동 여부 표기",
         대안="함수 호출·렌더 구조로 파생",
         한계="차단과 일반 실패가 섞일 수 있음",
         확보시="차단 이후 여정을 정확히 추적"),
    dict(id="overblock", 층="label",
         질문="차단하지 않아도 될 것을 차단하고 있는가",
         필요="차단 건 사후 검수 결과",
         대안="없음",
         한계="과차단 규모를 알 수 없음",
         확보시="정책 임계값 조정 근거"),
]


# ═══════════════════════════════════ 자동 판정

def detect(q: pd.DataFrame, tables: dict, gchk: dict | None = None,
           prep_meta: dict | None = None) -> pd.DataFrame:
    """각 결핍 항목이 지금 확보되어 있는지 자동 판정한다."""
    pm = prep_meta or {}
    have = {}

    def _nonempty(name):
        t = tables.get(name)
        return t is not None and len(t) > 0

    have["orders"] = _nonempty("orders") and _nonempty("app_views")
    have["app_session"] = _nonempty("app_sessions")
    have["entry_path"] = "entry_point" in q.columns and q["entry_point"].notna().any()
    have["answerable"] = not bool(pm.get("answerable_derived", True))
    have["slot"] = float(q["slot_target"].map(
        lambda v: isinstance(v, (list, tuple)) and len(v) > 0).mean()) >= 0.05
    have["quality_audit"] = "human_relevance" in q.columns
    have["cs_contact"] = "cs_contact" in q.columns
    have["release_log"] = False          # 외부 문서라 로그로 판정 불가
    have["protector_flag"] = ("protector_flag" in q.columns
                              and q["protector_flag"].notna().any())
    have["overblock"] = ("overblock" in q.columns and q["overblock"].notna().any())

    rows = []
    for g in GAP_CATALOG:
        ok = bool(have.get(g["id"], False))
        rows.append({**g, "상태": "확보" if ok else "결핍",
                     "층위": LAYERS.get(g["층"], g["층"])})
    return pd.DataFrame(rows)


def blocked_analyses(gchk: dict | None, tables: dict) -> dict:
    """설계한 분석 중 몇 개가 데이터 제약으로 결론에 이르지 못했는가."""
    items = []
    if gchk:
        for k, v in gchk.items():
            items.append({"분석": k, "가능": bool(v.get("통과", True)),
                          "사유": v.get("사유", "")})
    for name, label in [("view_order_panel", "정보→주문 전환 효과"),
                        ("churn_2x2", "챗봇 이탈 vs 고객 이탈")]:
        items.append({"분석": label, "가능": name in tables,
                      "사유": "" if name in tables else "필요한 로그가 연결되지 않음"})
    df = pd.DataFrame(items)
    n = len(df)
    blocked = int((~df["가능"]).sum()) if n else 0
    return {"표": df, "전체": n, "불능": blocked,
            "불능비율": blocked / n if n else np.nan}


def uncertainty_bands(F: dict | None) -> pd.DataFrame:
    """
    결핍이 만드는 '모르는 폭'을 숫자로. 이 표가 가장 설득력이 크다.
    """
    F = F or {}
    rows = []
    fb = F.get("폴백", {})
    if fb.get("보고 성공률") is not None:
        lo = fb.get("폴백 제외 성공률(하한)")
        hi = fb.get("보고 성공률")
        rows.append({"지표": "응답 성공률", "하한": lo, "상한": hi,
                     "폭": (hi - lo) if (lo is not None and hi is not None) else np.nan,
                     "좁히는 방법": "표본 1,000건 인적 검수"})
    es = F.get("실질성공", {})
    if es.get("실질 성공률 하한") is not None:
        rows.append({"지표": "사용자 관점 성공률",
                     "하한": es["실질 성공률 하한"], "상한": es["실질 성공률 상한"],
                     "폭": es.get("구간폭(TERMINAL 비중)"),
                     "좁히는 방법": "고객센터 유입 조인 또는 만족도 수집"})
    cx = F.get("차단생존", {})
    if cx.get("CI"):
        rows.append({"지표": "차단의 이탈 위험비", "하한": cx["CI"][0],
                     "상한": cx["CI"][1],
                     "폭": cx["CI"][1] - cx["CI"][0],
                     "좁히는 방법": "차단 표기 라벨 + P3 통과 건 확보"})
    out = pd.DataFrame(rows)
    return out.round(4) if len(out) else out


def missing_scale(q: pd.DataFrame) -> pd.DataFrame:
    """결측·미충전 규모."""
    rows = []
    for col, label in [("l2_intent", "의도 라벨"), ("l1_stage", "여정 단계"),
                       ("f4_compliance", "컴플라이언스 등급"),
                       ("answerable", "답변 가능성 라벨"),
                       ("query_text", "질문 원문"), ("answer_text", "응답 원문")]:
        if col in q.columns:
            rows.append({"항목": label, "결측률": float(q[col].isna().mean())})
    rows.append({"항목": "질문 대상(슬롯) 충전율",
                 "결측률": 1 - float(q["slot_target"].map(
                     lambda v: isinstance(v, (list, tuple)) and len(v) > 0).mean())})
    return pd.DataFrame(rows).round(4)


def window_loss(tables: dict) -> dict:
    """관측 창이 도달하지 않아 판정에서 빠진 규모(코호트 표의 미관측 셀)."""
    c = tables.get("cohort")
    if c is None or len(c) == 0:
        return {}
    cols = [x for x in c.columns if str(x).startswith("D")]
    if not cols:
        return {}
    na = c[cols].isna()
    key = c.columns[0]
    未 = c.loc[na.any(axis=1), key].astype(str).tolist()
    return {"미관측 셀 비율": round(float(na.to_numpy().mean()), 4),
            "일부라도 미관측인 코호트": len(未),
            "전체 코호트": len(c),
            "해당 코호트": 未[-5:]}


# ═══════════════════════════════════ 사용자 입력 병합

TEMPLATE = {
    "_안내": [
        "이 파일을 채우면 보고서에 그대로 반영됩니다.",
        "결핍항목: 자동 판정된 항목에 담당·난이도·일정을 덧붙입니다.",
        "  난이도는 낮음 / 중간 / 높음 중 하나로 적으십시오.",
        "오판사례: 데이터가 부족해 처음에 잘못 볼 뻔했던 사례입니다.",
        "  추상적 경고보다 실제 사례가 훨씬 잘 전달됩니다.",
        "추가결핍: 목록에 없는 결핍을 직접 추가할 때 씁니다.",
    ],
    "결핍항목": {
        "entry_path": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "app_session": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "orders": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "answerable": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "slot": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "quality_audit": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "cs_contact": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "release_log": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "protector_flag": {"난이도": "", "담당": "", "일정": "", "메모": ""},
        "overblock": {"난이도": "", "담당": "", "일정": "", "메모": ""},
    },
    "오판사례": [
        {"처음결론": "", "보완후": "", "부족했던것": ""},
    ],
    "추가결핍": [],
}


def ensure_config(path: Path) -> dict:
    """설정 파일이 없으면 템플릿을 만들고, 있으면 읽는다."""
    if not path.exists():
        path.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def merge(auto: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """자동 판정 결과에 사용자 입력을 병합."""
    d = auto.copy()
    for c in ("난이도", "담당", "일정", "메모"):
        d[c] = ""
    user = (cfg or {}).get("결핍항목", {})
    for i, r in d.iterrows():
        u = user.get(r["id"], {})
        for c in ("난이도", "담당", "일정", "메모"):
            if u.get(c):
                d.at[i, c] = str(u[c])
    extra = (cfg or {}).get("추가결핍", [])
    if extra:
        d = pd.concat([d, pd.DataFrame(extra)], ignore_index=True)
    # 결핍 → 확보 순, 난이도 낮은 순
    order = {"낮음": 0, "중간": 1, "높음": 2, "": 3}
    d["_o"] = d["난이도"].map(lambda v: order.get(str(v).strip(), 3))
    d = d.sort_values(["상태", "_o"], ascending=[True, True]).drop(columns="_o")
    return d.reset_index(drop=True)


def misjudgments(cfg: dict) -> pd.DataFrame:
    rows = [r for r in (cfg or {}).get("오판사례", [])
            if any(str(v).strip() for v in r.values())]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def config_status(cfg: dict, merged: pd.DataFrame) -> dict:
    """사용자가 채워야 할 것이 얼마나 남았는지."""
    gaps = merged[merged["상태"].eq("결핍")]
    filled = int((gaps["난이도"].astype(str).str.strip() != "").sum())
    return {"결핍항목": int(len(gaps)), "난이도기입": filled,
            "오판사례": int(len(misjudgments(cfg))),
            "미기입": int(len(gaps) - filled)}
