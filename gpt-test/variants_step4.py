"""STEP 4 개선 3종 — make_variant.py 가 가져다 쓴다.

STEP 4 는 답 변경의 51% 를 만들고(전파율 60%) 프롬프트의 33% 를 쓴다.
파스 층위에서도 `goal 개수 80%` → `goal 분류 52%` 로 계단이 꺾인다 —
몇 조각으로 쪼갤지는 정하는데 라벨을 못 정한다. 절삭은 이미 세 번 다 실패했다
(cent/ccon/cex). 그래서 줄이는 게 아니라 구조를 바꾸는 세 가지를 시험한다.
"""

import re

FACET_SCOPE = """First fix `domain`, then choose `facet` **only from that domain's list
below**. A facet outside the list is a classification error.

  issuer         profile ipo price flow short fundamentals valuation
                 estimate target_price scoring news disclosure sector_map
                 screening fx knowledge regulation howto
  market         price flow short valuation screening sector_map ipo fx
                 knowledge regulation howto news
                 (no fundamentals / estimate / target_price / disclosure /
                  scoring / profile — 시장 단위로는 그 데이터가 없다)
  internal       howto ipo regulation knowledge fx disclosure
  finance_legal  knowledge regulation ipo howto

`none` is available in every domain.

"""

NONE_OLD = (
    "- A goal whose type is `assessment` or `recommendation` is normally "
    "`none`,\n"
    "  because it consumes other goals rather than requesting a facet of its "
    "own.\n"
    "  Give it a real facet only when the user explicitly asks for that facet."
)

NONE_NEW = (
    "- `none` is a last resort. Even a judgment goal is *about* something —\n"
    "  give it the facet of the evidence the judgment rests on. \"향후 실적\n"
    "  전망\" is `estimate`; \"목표주가 하향 가능성\" is `target_price`;\n"
    "  \"주가 전망\" is `price`. Use `none` only when no facet fits at all.\n"
    "  Judgment vs lookup is already carried by `type`, not by the facet."
)


def facet_2stage(secs, find):
    """① facet 을 도메인별로 좁힌다.

    지금은 19개를 한 층위에 늘어놓고 고르게 한다. route.TABLE 은 이미
    (domain, facet) 으로 인덱싱되므로 도메인마다 유효한 facet 이 정해져 있다.
    internal 은 6개, finance_legal 은 4개뿐인데 19개를 다 보여주고 있었다.
    선택지를 줄이면 분류가 덜 흔들린다는 가설.
    """
    i = find(secs, "STEP 4")
    body = secs[i][1]
    m = re.search(r"### FACET\n\n", body)
    if not m:
        raise KeyError("### FACET")
    secs[i] = (secs[i][0], body[:m.end()] + FACET_SCOPE + body[m.end():])
    return secs


def no_none(secs, find):
    """② facet=none 을 걷어내고 판단 여부는 type 이 전담하게 한다.

    `none` 은 "판단 목표"라는 뜻인데 그건 type=assessment 가 이미 말한다.
    같은 정보를 두 필드가 나눠 가지면 둘이 어긋날 수 있다 — type=query 인데
    facet=none 같은 조합이 실제로 UNMAPPED 를 만든다. facet 은 늘 "무엇에
    관한 정보인가"만 답하게 한다.

    주의: route.py 의 CV_RULES 두 개가 facet=none 에 걸려 있다
    (internal/none→manual, market/none→knowledge). none 이 줄면 그 규칙이
    안 터지므로, 이 변형은 프롬프트 이득과 규칙 손실이 상쇄될 수 있다.
    """
    i = find(secs, "STEP 4")
    body = secs[i][1]
    if NONE_OLD not in body:
        raise KeyError("none 규칙 문단을 못 찾았다")
    secs[i] = (secs[i][0], body.replace(NONE_OLD, NONE_NEW))
    return secs


def renumber(secs, find):
    """③ STEP 번호를 읽는 순서대로 다시 매긴다.

    pcls1 로 CLASSIFY 를 앞으로 옮기면서 번호가 4→0→1→2→3→5→6 이 됐다.
    pcls1 의 이득이 "분류를 먼저 읽어서"인지 "번호가 뒤섞여서"인지 갈리지
    않는데, 번호만 바로잡은 변형을 재면 갈린다.
    """
    n = [0]

    def fix(_m):
        n[0] += 1
        return f"## STEP {n[0]} —"

    return [
        (re.sub(r"## STEP \d+ —", fix, h) if h else h, b) for h, b in secs
    ]
