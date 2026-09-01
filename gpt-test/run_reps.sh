#!/usr/bin/env bash
# arm 내 분산 측정용 반복 실행.
#
# 기존 산출물(<src>_out.csv, <src>_out_b.csv)을 rep1 으로 보고 rep2/rep3 을 낸다.
# 두 스크립트 모두 마지막 편집 이후에 rep1 이 생성됐으므로 같은 프롬프트다.
#
#   rep2 -> <src>_out_r2.csv   / <src>_out_b_r2.csv
#   rep3 -> <src>_out_r3.csv   / <src>_out_b_r3.csv
set -u

PY=.venv/Scripts/python.exe
SRCS="work invest ext_ipo ext_tax ext_div ext_fx ext_index ext_basis"

for rep in 2 3; do
  for src in $SRCS; do
    for arm in A B; do
      if [ "$arm" = "A" ]; then
        script=run_csv.py;   out="${src}_out_r${rep}.csv"
      else
        script=run_csv_b.py; out="${src}_out_b_r${rep}.csv"
      fi
      if [ -s "$out" ]; then
        echo "[SKIP] $out 이미 있음"
        continue
      fi
      echo "[RUN ] rep$rep $arm $src -> $out"
      "$PY" "$script" --input "${src}.csv" --output "$out" >/dev/null 2>&1 \
        || echo "[FAIL] $out"
    done
  done
done
echo "[DONE]"
