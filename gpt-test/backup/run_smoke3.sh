#!/usr/bin/env bash
# 후보 3종 x 3회. rep1 은 기존 smoke_out_<tag>.csv.
set -u
PY=.venv/Scripts/python.exe
for rep in 2 3; do
  for t in d pcls1 cex; do
    out="smoke_out_${t}_r${rep}.csv"
    [ -s "$out" ] && { echo "[SKIP] $out"; continue; }
    "$PY" "run_csv_${t}.py" --input smoke.csv --output "$out" >/dev/null 2>&1 \
      && echo "[OK  ] $t rep$rep" || echo "[FAIL] $t rep$rep"
  done
done
echo "[DONE]"
