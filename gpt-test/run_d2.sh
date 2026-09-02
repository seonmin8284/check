#!/usr/bin/env bash
# ② 프롬프트(facet=none 제거) 로 273행 x 3회. 파일명은 _out_e*.csv.
# 기존 _out_d*.csv 는 ② 이전 프롬프트의 산출물이라 덮어쓰지 않는다.
set -u
PY=.venv/Scripts/python.exe
run() {
  [ -s "$2" ] && { echo "[SKIP] $2"; return; }
  "$PY" run_csv_d.py --input "$1" --output "$2" >/dev/null 2>&1 || echo "[FAIL] $2"
}
for rep in 1 2 3; do
  sfx=""; [ "$rep" != "1" ] && sfx="_r${rep}"
  run "work copy.csv"   "work_out_e${sfx}.csv"
  run "invest copy.csv" "invest_out_e${sfx}.csv"
  for src in ext_ipo ext_tax ext_div ext_fx ext_index ext_basis ext_edge; do
    run "${src}.csv" "${src}_out_e${sfx}.csv"
  done
  echo "[rep$rep DONE]"
done
echo "[DONE]"
