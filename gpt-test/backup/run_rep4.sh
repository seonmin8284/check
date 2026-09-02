#!/usr/bin/env bash
# rep4 — rep1 을 뺀 깨끗한 3회(rep2,3,4)로 다수결을 성립시키기 위한 추가 실행.
set -u
PY=.venv/Scripts/python.exe
SRCS="work invest ext_ipo ext_tax ext_div ext_fx ext_index ext_basis"
for src in $SRCS; do
  for arm in A B; do
    if [ "$arm" = "A" ]; then script=run_csv.py; out="${src}_out_r4.csv"
    else script=run_csv_b.py; out="${src}_out_b_r4.csv"; fi
    [ -s "$out" ] && { echo "[SKIP] $out"; continue; }
    echo "[RUN ] rep4 $arm $src"
    "$PY" "$script" --input "${src}.csv" --output "$out" >/dev/null 2>&1 || echo "[FAIL] $out"
  done
done
echo "[DONE]"
