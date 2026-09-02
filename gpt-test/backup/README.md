# backup/ — 결론이 난 뒤 물러난 것들

여기 있는 것은 **틀린 것이 아니라 끝난 것**이다. 결론을 만든 실험과 그 산출물이고,
결론 자체는 [../AB_RESULT.md](../AB_RESULT.md) 에 §1~§15 로 남아 있다.

지우지 않은 이유는 셋이다. 결론의 근거가 여기 있고, 골든이 더 늘면 다시 돌려야
할 것이 있고, 무엇보다 **이 프로젝트에서 결론이 세 번 뒤집혔다** — 되돌아갈
자리를 남겨두는 편이 낫다.

## 현행에 남은 것 (`../`)

| 파일 | 역할 |
|---|---|
| `route.py` | 라우터. 3층 그래프 + CV 규칙 5개 |
| `run_csv_d.py` | 채택된 프롬프트 (arm D + CLASSIFY 앞으로 + facet=none 억제) |
| `route_intent_d.py` | goal 에서 의도 코드 유도 (LLM 호출 0) |
| `ensemble_predict.py` | 배포용 투표 병합 (기본 임계 = 합집합) |
| `optimize_route.py` | 규칙 자동 생성 + 5-fold CV 선택 |
| `freeze_split.py` | dev/burned/sealed 분할 관리 |
| `var_source.py` · `parse_repro.py` | 재현성 진단 |
| `refit_b.py` | 이름은 옛 실험 것이지만 **공용 헬퍼**(골든/분할/파스 로딩, f1)라 남겼다 |
| `*_out_e*.csv` | 현행 arm 파스 273행 × 3회 |
| `run_d2.sh` | 그 파스를 만드는 러너 |

## 여기 들어온 것

### 기각된 arm

| | 왜 물러났나 |
|---|---|
| `run_csv.py` (A) | 미노출 116행에서 B 에 뒤짐(0.5887 vs 0.6187). 초기 우세는 그래프가 A rep1 에 맞춰진 산물이었다 — rep1 편향 진단 8/8 |
| `run_csv_b.py` (B) | D 로 대체. intent 를 걷어내 토큰 −14%, 성능 동등 |
| `run_csv_c.py` (C) | intent 분류 스키마. 완벽한 의도로도 천장이 0.843 |
| `route_intent.py` | C 의 라우터. `ensemble_predict.py` 가 `_c.csv` 를 읽을 때 아직 참조한다 |

### 기각된 프롬프트 변형

`run_csv_{pcls1,pcls2,pex,ccon,cent,cex,s4a,s4b,s4c,s4bc}.py` 와 생성기
`make_variant.py` · `variants_step4.py`.

채택된 둘(`pcls1`, `s4b`)은 이미 `run_csv_d.py` 에 반영돼 있다. 나머지는 스모크
에서 방향이 나빴다 — 절삭 셋(`ccon`/`cent`/`cex`)은 전부 손해였고,
`s4a`(facet 도메인별)는 분류 재현성을 14→7 로 반토막냈다.

### 소임을 다한 분석 스크립트

| | |
|---|---|
| `ab_fit.py` | arm 별 그래프 적합. "각 arm 에 제 그래프를 준다" 원칙의 출처 |
| `rep_var.py` · `ensemble.py` · `ensemble_sweep.py` | 반복 분산·앙상블 예산 탐색 |
| `refit_graph.py` · `validate_refit.py` · `validate_holdout.py` | dev/sealed 기반 재적합과 검증. `optimize_route.py` 의 CV 방식으로 대체됐다 |
| `compare_bd.py` | B vs D 2×2 (파스 × 그래프) |
| `score_ab.py` | 초기 A/B 채점기 |

### 옛 파스 산출물 (148개)

`*_out.csv`(A) · `*_out_b*.csv`(B) · `*_out_c*.csv`(C) · `*_out_d*.csv`(D 구판) ·
`*_out_r*.csv` · `smoke_out_*.csv`.

`_out_d*` 는 STEP4 ② 이전 프롬프트라 파스 분포가 다르다(facet=none 13% vs 8.5%).
그 위에서 규칙을 고르면 지금 쓰지 않는 분포에 맞추게 되므로 뺐다.

### 기타

- `golden_labels_200.csv` — 사용자 라벨 200행. 현행 골든 273행에 병합됨
- `golden_labels.pre200.bak.csv` — 병합 전 99행본. 커밋된 적이 없어 git 으로 복원 불가라 보존
- `work.csv` · `invest.csv` — 13행 부분집합. 100행본(`../work copy.csv`)이 정본
- `arm_test/` — 초기 실험 디렉터리

## 다시 쓰려면

여기 스크립트들은 **프로젝트 루트 기준 상대경로**로 산출물을 찾는다.
`backup/` 안에서 실행하면 파일을 못 찾는다. 루트로 복사해 쓰거나 경로를 고쳐라.

옛 arm 을 다시 재려면 산출물도 같이 루트로 올려야 한다 — 예를 들어
`rep_var.py` 는 `<src>_out.csv` 부터 `<src>_out_b_r4.csv` 까지를 찾는다.
