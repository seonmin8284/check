#!/usr/bin/env bash
# 변형 스모크 — 27행 층화 표본. 기준선 D 포함 7종.
set -u
PY=.venv/Scripts/python.exe
for t in d pcls1 pcls2 pex ccon cent cex; do
  out="smoke_out_${t}.csv"
  [ -s "$out" ] && { echo "[SKIP] $out"; continue; }
  "$PY" "run_csv_${t}.py" --input smoke.csv --output "$out" >/dev/null 2>&1 \
    && echo "[OK  ] $t" || echo "[FAIL] $t"
done
echo "[DONE]"
