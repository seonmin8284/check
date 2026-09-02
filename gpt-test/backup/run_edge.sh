#!/usr/bin/env bash
# ext_edge(32행) 파스 생성. rep1 은 만들지 않는다 — CLEAN_REPS=(2,3,4) 규약에
# 맞춰 rep2~4 로 낸다. ext_edge 는 그래프·프롬프트 어디에도 노출된 적이 없어
# 오염이 없지만, 파일명 규약을 기존과 맞춰야 하네스가 그대로 돈다.
set -u
PY=.venv/Scripts/python.exe
run() {  # $1=script $2=out
  [ -s "$2" ] && { echo "[SKIP] $2"; return; }
  echo "[RUN ] $2"
  "$PY" "$1" --input ext_edge.csv --output "$2" >/dev/null 2>&1 || echo "[FAIL] $2"
}
for rep in 2 3 4; do
  run run_csv.py   "ext_edge_out_r${rep}.csv"
  run run_csv_b.py "ext_edge_out_b_r${rep}.csv"
done
for rep in 2 3; do
  run run_csv_c.py "ext_edge_out_c_r${rep}.csv"
done
echo "[DONE]"
