#!/usr/bin/env bash
# D안(= B안에서 intent 제거) 전체 실행.
#
# work/invest 는 골든이 100행씩이다. work.csv/invest.csv 는 13행짜리
# 부분집합이므로 반드시 100행본("work copy.csv")을 써야 한다 — 여기서
# 한 번 틀려서 87행씩 빠진 채로 돌린 적이 있다.
set -u
PY=.venv/Scripts/python.exe
run() {
  [ -s "$2" ] && { echo "[SKIP] $2"; return; }
  echo "[RUN ] $2  <- $1"
  "$PY" run_csv_d.py --input "$1" --output "$2" >/dev/null 2>&1 || echo "[FAIL] $2"
}
run "work copy.csv"   work_out_d.csv
run "invest copy.csv" invest_out_d.csv
for src in ext_ipo ext_tax ext_div ext_fx ext_index ext_basis ext_edge; do
  run "${src}.csv" "${src}_out_d.csv"
done
echo "[DONE]"
