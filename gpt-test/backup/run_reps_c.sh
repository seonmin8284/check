#!/usr/bin/env bash
# arm C 반복 실행. 기존 <src>_out_c.csv 를 rep1 로 보고 rep2/rep3 을 낸다.
set -u
PY=.venv/Scripts/python.exe
SRCS="work invest ext_ipo ext_tax ext_div ext_fx ext_index ext_basis"
for rep in 2 3; do
  for src in $SRCS; do
    out="${src}_out_c_r${rep}.csv"
    [ -s "$out" ] && { echo "[SKIP] $out"; continue; }
    echo "[RUN ] rep$rep C $src -> $out"
    "$PY" run_csv_c.py --input "${src}.csv" --output "$out" >/dev/null 2>&1 \
      || echo "[FAIL] $out"
  done
done
echo "[DONE]"
