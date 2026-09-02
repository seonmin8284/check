#!/usr/bin/env bash
# D 반복 실행 — 재현성 측정용. rep1 은 기존 <src>_out_d.csv.
set -u
PY=.venv/Scripts/python.exe
run() {
  [ -s "$2" ] && { echo "[SKIP] $2"; return; }
  "$PY" run_csv_d.py --input "$1" --output "$2" >/dev/null 2>&1 || echo "[FAIL] $2"
}
for rep in 2 3; do
  run "work copy.csv"   "work_out_d_r${rep}.csv"
  run "invest copy.csv" "invest_out_d_r${rep}.csv"
  for src in ext_ipo ext_tax ext_div ext_fx ext_index ext_basis ext_edge; do
    run "${src}.csv" "${src}_out_d_r${rep}.csv"
  done
  echo "[rep$rep DONE]"
done
echo "[DONE]"
