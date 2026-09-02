"""run_csv_d.py 에서 프롬프트 변형본을 찍어낸다.

D 의 프롬프트를 손으로 복사해 5벌 만들면 원본이 바뀔 때마다 5벌이 어긋난다.
그래서 변환으로 생성한다 — 원본은 하나고 변형은 규칙이다.

변형 종류:

  위치   섹션 순서만 바꾼다. 내용은 그대로라 토큰이 거의 안 변한다.
         분류(STEP 4)가 재현성 파괴 1위인데 프롬프트 중간에 묻혀 있다.
         앞·뒤로 옮겨 위치 효과가 있는지 본다.

  절삭   블록을 줄인다. 근거는 var_source.py 의 전파율 —
         STEP 3 은 10행 갈려 답이 0건 바뀌었고, STEP 2 는 79행 갈려
         19% 만 전파된다. 값어치보다 자리를 많이 쓰는 쪽부터 자른다.

실행:
    .venv/Scripts/python.exe make_variant.py            # 전부 생성
    .venv/Scripts/python.exe make_variant.py --list     # 목록만
"""

import argparse
import re
import sys

SRC = "run_csv_d.py"


def prompt_of(src: str) -> str:
    return max(re.findall(r'"""(.*?)"""', src, re.S), key=len)


def split_sections(p: str) -> list[tuple[str, str]]:
    """[(헤더, 본문)] — 첫 원소의 헤더는 '' (서두)."""
    parts = re.split(r"(?m)^(## .+)$", p)
    out = [("", parts[0])]
    for i in range(1, len(parts), 2):
        out.append((parts[i], parts[i + 1] if i + 1 < len(parts) else ""))
    return out


def rebuild(secs) -> str:
    return "".join(h + b if h else b for h, b in secs)


def find(secs, key):
    for i, (h, _) in enumerate(secs):
        if key in h:
            return i
    raise KeyError(key)


# ── 변형 정의 ────────────────────────────────────────────

def v_classify_first(secs):
    """STEP 4 CLASSIFY 를 STEP 0 앞으로. 분류 기준을 먼저 읽힌다."""
    i = find(secs, "STEP 4")
    j = find(secs, "STEP 0")
    s = secs.pop(i)
    secs.insert(j, s)
    return secs


def v_classify_last(secs):
    """STEP 4 CLASSIFY 를 예시 직전(맨 뒤쪽)으로. 최신성 효과를 본다."""
    i = find(secs, "STEP 4")
    s = secs.pop(i)
    j = find(secs, "BOUNDARY")
    secs.insert(j, s)
    return secs


def v_examples_first(secs):
    """경계 예시를 STEP 정의 앞으로. 규칙보다 사례를 먼저 읽힌다."""
    i = find(secs, "BOUNDARY")
    s = secs.pop(i)
    j = find(secs, "STEP 0")
    secs.insert(j, s)
    return secs


def _strip_glosses(body: str) -> str:
    """'타입명  한글설명' 목록을 이름만 남긴 한 줄로 접는다.

    스키마가 strict enum 이라 이름은 이미 강제된다. 프롬프트가 더하는 것은
    '언제 쓰는가'인데, 그 설명이 실제로 값을 하는지 시험하는 절삭이다.
    """
    lines = body.split("\n")
    names, keep, in_list = [], [], False
    for l in lines:
        m = re.match(r"^([a-z_]{3,20})\s{2,}\S", l)
        if m:
            names.append(m.group(1))
            in_list = True
            continue
        if in_list and not l.strip():
            if names:
                keep.append(", ".join(names))
                names = []
            in_list = False
        keep.append(l)
    if names:
        keep.append(", ".join(names))
    return "\n".join(keep)


def v_cut_constraints(secs):
    """STEP 3 — 제약 타입 설명 제거. 갈려도 답이 0건 바뀐 블록."""
    i = find(secs, "STEP 3")
    secs[i] = (secs[i][0], _strip_glosses(secs[i][1]))
    return secs


def v_cut_entities(secs):
    """STEP 2 — 엔티티 타입 설명 제거. 79행 갈리는데 전파율 19%."""
    i = find(secs, "STEP 2")
    secs[i] = (secs[i][0], _strip_glosses(secs[i][1]))
    return secs


def v_cut_examples(secs):
    """경계 예시를 앞 4개만 남긴다 (프롬프트의 29% 를 쓰는 블록)."""
    i = find(secs, "BOUNDARY")
    body = secs[i][1]
    blocks = body.split("\n---\n")
    secs[i] = (secs[i][0], "\n---\n".join(blocks[:5]))
    return secs


VARIANTS = {
    "pcls1": ("위치: CLASSIFY 를 맨 앞으로", v_classify_first),
    "pcls2": ("위치: CLASSIFY 를 예시 직전으로", v_classify_last),
    "pex":   ("위치: 경계 예시를 STEP 앞으로", v_examples_first),
    "ccon":  ("절삭: STEP 3 제약 압축", v_cut_constraints),
    "cent":  ("절삭: STEP 2 엔티티 압축", v_cut_entities),
    "cex":   ("절삭: 경계 예시 8→4", v_cut_examples),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for tag, (desc, _) in VARIANTS.items():
            print(f"  {tag:<7} {desc}")
        return 0

    src = open(SRC, encoding="utf-8").read()
    base = prompt_of(src)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        ntok = lambda s: len(enc.encode(s))
    except ImportError:
        ntok = lambda s: len(s) // 3

    b = ntok(base)
    print(f"기준 D 프롬프트 {b} tok\n")
    print(f"  {'tag':<7}{'tok':>6}{'Δ':>7}  설명")
    for tag, (desc, fn) in VARIANTS.items():
        secs = fn(split_sections(base))
        new = rebuild(secs)
        out = src.replace(base, new)
        out = out.replace("work_out_d.csv", f"work_out_{tag}.csv")
        path = f"run_csv_{tag}.py"
        open(path, "w", encoding="utf-8").write(out)
        t = ntok(new)
        print(f"  {tag:<7}{t:>6}{t-b:>+7}  {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
