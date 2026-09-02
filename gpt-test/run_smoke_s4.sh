#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
for rep in 1 2 3; do
  for t in s4a s4b s4c; do
    out="smoke_out_${t}_r${rep}.csv"
    [ -s "$out" ] && { echo "[SKIP] $out"; continue; }
    "$PY" "run_csv_${t}.py" --input smoke.csv --output "$out" >/dev/null 2>&1 \
      && echo "[OK  ] $t rep$rep" || echo "[FAIL] $t rep$rep"
  done
done
echo "[DONE]"
