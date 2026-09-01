"""골든 67행을 dev/holdout 으로 갈라 **동결**한다.

지금까지 capability graph 의 규칙도, 그 규칙의 선택도, 프롬프트 few-shot 도
전부 골든 67행 전체를 보면서 만들어졌다. 그래서 어떤 숫자도 "새 데이터에서
이만큼 나온다"를 뜻하지 못한다. 여기서 끊는다.

holdout 은 앞으로의 어떤 최적화에서도 **읽지 않는다.** 규칙 작성, 규칙 선택,
프롬프트 수정, few-shot 고르기 — 전부 dev 만 보고 한다. holdout 은 마지막에
한 번 열어 보고, 그 뒤로는 오염된 것으로 간주해 새로 잘라야 한다.

한 번 만들어지면 덮어쓰지 않는다. 분할이 흔들리면 동결이 아니다.

실행:
    .venv/Scripts/python.exe freeze_split.py          # 없으면 생성
    .venv/Scripts/python.exe freeze_split.py --show   # 현황만
"""

import argparse
import csv
import hashlib
import os
import random
import sys
from collections import defaultdict

# 분할 라벨은 셋이다. holdout 을 한 번 열면 그 뒤로는 dev 와 다를 바 없으므로,
# "아직 안 연 것"과 "열어버린 것"을 이름으로 갈라둔다. 뭉뚱그리면 다음 라운드에
# 소진된 표본을 깨끗한 검증셋으로 착각하게 된다.
#
#   dev     적합에 쓴다. 마음껏 본다.
#   burned  holdout 이었으나 개봉됐다. 이제 dev 취급 — 검증 근거로 못 쓴다.
#   sealed  아직 안 열었다. 유일하게 일반화 성능을 말할 수 있는 표본.
DEV, BURNED, SEALED = "dev", "burned", "sealed"

# 골든이 늘어날 때 새 source 를 어디로 보낼지. 기존 배정은 절대 안 건드린다.
#   ext_edge — 그래프·프롬프트·앙상블 결정 어디에도 노출된 적이 없다.
NEW_SOURCE_SPLIT = {"ext_edge": SEALED}

GOLDEN = "golden_labels.csv"
SPLIT = "split_frozen.csv"
SEED = 20260901
HOLDOUT_FRAC = 1 / 3


def load_keys():
    rows = []
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["source"], int(r["idx"]), r["query"]))
    return rows


def make_split(rows):
    """source 층화. 각 source 에서 1/3 을 holdout 으로."""
    rng = random.Random(SEED)
    by_src = defaultdict(list)
    for src, idx, q in rows:
        by_src[src].append((src, idx, q))

    split = {}
    for src in sorted(by_src):
        items = sorted(by_src[src], key=lambda x: x[1])
        rng.shuffle(items)
        n_hold = max(1, round(len(items) * HOLDOUT_FRAC))
        for i, (s, idx, q) in enumerate(items):
            split[(s, idx)] = "holdout" if i < n_hold else "dev"
    return split


def write_split(rows, split):
    with open(SPLIT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "idx", "split", "query"])
        for src, idx, q in rows:
            w.writerow([src, idx, split[(src, idx)], q])


def load_split(path=SPLIT):
    """{(source, idx): 'dev'|'holdout'}"""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source"], int(r["idx"]))] = r["split"]
    return out


def dev_keys(path=SPLIT):
    """적합에 써도 되는 행. 개봉된 holdout(burned)도 여기 포함된다."""
    return {k for k, v in load_split(path).items() if v in (DEV, BURNED)}


def fit_only_keys(path=SPLIT):
    """burned 를 빼고 원래 dev 만. 기존 결과와 대조할 때 쓴다."""
    return {k for k, v in load_split(path).items() if v == DEV}


def burned_keys(path=SPLIT):
    return {k for k, v in load_split(path).items() if v == BURNED}


def sealed_keys(path=SPLIT):
    """아직 안 연 행. 여기서 잰 값만 일반화 성능이라 부를 수 있다."""
    return {k for k, v in load_split(path).items() if v == SEALED}


def holdout_keys(path=SPLIT):
    raise RuntimeError(
        "holdout_keys() 는 폐기됐다. burned(개봉됨) 와 sealed(봉인) 를 "
        "구분해서 sealed_keys() 또는 burned_keys() 를 써라."
    )


def golden_digest():
    """골든 파일이 분할 이후 바뀌었는지 보는 지문."""
    h = hashlib.sha256()
    with open(GOLDEN, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def report(split, rows):
    by = defaultdict(lambda: [0, 0])
    for src, idx, _ in rows:
        by[src][split[(src, idx)] == "holdout"] += 1
    print(f"{'source':<12}{'dev':>5}{'holdout':>9}")
    td = th = 0
    for src in sorted(by):
        d, h = by[src]
        td += d
        th += h
        print(f"{src:<12}{d:>5}{h:>9}")
    print(f"{'합계':<12}{td:>5}{th:>9}   (holdout {th/(td+th):.0%})")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true")
    p.add_argument("--extend", action="store_true",
                   help="골든에 새로 생긴 행만 분할에 덧붙인다 (기존 배정 불변)")
    args = p.parse_args()

    rows = load_keys()

    if os.path.exists(SPLIT):
        split = load_split()
        missing = [(s, i) for s, i, _ in rows if (s, i) not in split]
        print(f"[FROZEN] {SPLIT} 이미 존재 — 기존 배정은 덮어쓰지 않는다.")

        if missing and args.extend:
            rng = random.Random(SEED)
            by_src = defaultdict(list)
            for s, i in missing:
                by_src[s].append((s, i))
            for src in sorted(by_src):
                items = sorted(by_src[src], key=lambda x: x[1])
                forced = NEW_SOURCE_SPLIT.get(src)
                if forced:
                    for k in items:
                        split[k] = forced
                    print(f"[EXTEND] {src} {len(items)}행 → 전량 {forced} "
                          f"(NEW_SOURCE_SPLIT 지정)")
                else:
                    # 지정이 없는 source 는 dev 로만 넣는다. holdout 을 사후에
                    # 늘리면 이미 본 데이터가 섞여 봉인이 깨진다.
                    for k in items:
                        split[k] = "dev"
                    print(f"[EXTEND] {src} {len(items)}행 → 전량 dev (기본값)")
            write_split(rows, split)
            print(f"[EXTEND] {SPLIT} 갱신 — 기존 {len(rows)-len(missing)}행 배정 불변\n")
        elif missing:
            print(f"[WARN] 골든에 분할이 없는 행 {len(missing)}개: {missing[:5]}")
            print("       --extend 를 주면 덧붙인다.")

        report(split, [r for r in rows if (r[0], r[1]) in split])
        print(f"\n골든 지문 {golden_digest()}")
        return 0

    if args.show:
        print(f"[NONE] {SPLIT} 없음")
        return 1

    split = make_split(rows)
    write_split(rows, split)
    print(f"[FREEZE] {SPLIT} 생성 (seed={SEED})")
    report(split, rows)
    print(f"\n골든 지문 {golden_digest()}")
    print("\n앞으로 dev 만 보고 최적화한다. holdout 은 마지막 1회만 연다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
