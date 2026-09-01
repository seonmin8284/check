# gpt-5-mini: 의도분류 + capability graph vs. 단건 fat-prompt

두 아키텍처를 같은 평가셋(42케이스)으로 비교한다.

| | Arm A | Arm B |
|---|---|---|
| LLM 역할 | 의도분류 + 슬롯추출만 | 분류 + 조회 + 결과 생성 |
| 계산 주체 | 결정적 capability graph (파이썬) | LLM |
| 시스템 프롬프트 | 의도 목록만 | 의도 목록 + 전체 룩업 테이블 |

## 파일

- `capabilities.py` — 순수 함수 + 타입 포트, 의도→실행계획(DAG)
- `graph.py` — 결정적 실행기. 상류 간선이 슬롯보다 우선해 배선된다
- `dataset.py` — 라벨링된 42케이스. 정답값은 gold 라벨을 그래프에 통과시켜 생성
- `arms.py` — 두 arm 구현 (둘 다 structured output 으로 intent 를 강제 → 분류 성능 비교 가능)
- `benchmark.py` — 실행 + 지연/정확도/토큰 리포트

## 실행

```bash
.venv/Scripts/python.exe benchmark.py --limit 5   # 스모크
.venv/Scripts/python.exe benchmark.py             # 전체
.venv/Scripts/python.exe benchmark.py --repeat 3  # 지연 안정화
```

`results.json` 에 케이스별 원자료가 저장된다.

## 최종 문장을 LLM 이 쓰게 하기 (`--compose`)

기본 모드는 결과값(파이썬 값 / JSON)까지만 낸다. `--compose` 를 켜면 자연어 작성 단계가 붙는다.

- Arm A: 그래프가 확정한 값을 근거로 **2번째 API 호출**이 문장을 쓴다 (`arms.COMPOSE_SYSTEM`).
  작성 프롬프트는 "주어진 값만 쓰고 새로 계산하지 마라"로 잠가 둔다.
- Arm B: 설계상 단건 호출이므로 같은 응답에 `answer` 필드가 하나 늘어날 뿐이다.

```bash
.venv/Scripts/python.exe compose_demo.py --case 22   # 단건 단계별 관찰
.venv/Scripts/python.exe benchmark.py --compose      # 전체 비교
```

## 측정 결과 (2026-08-31, v2 taxonomy 43의도 / main 53케이스 + edge 6케이스)

| 셋 | effort | arm | 의도정확도 | 결과정확도 | 평균 | p95 | in_tok | out_tok | 비용 |
|---|---|---|---|---|---|---|---|---|---|
| main | low | A | 98.1% | 98.1% | 3.23s | 9.65s | 1360 | 122 | $0.031 |
| main | low | B | 98.1% | 98.1% | 4.56s | 8.27s | 2770 | 293 | $0.068 |
| main | **minimal** | A | **100.0%** | **98.1%** | 2.36s | 8.49s | 1360 | 55 | $0.024 |
| main | **minimal** | B | 98.1% | **52.8%** | 2.24s | 6.60s | 2770 | 52 | $0.042 |
| edge | low | A | 83.3% | 83.3% | 3.03s | 4.68s | 1362 | 162 | $0.004 |
| edge | low | B | 83.3% | 83.3% | 4.38s | 8.79s | 2772 | 275 | $0.008 |
| edge | minimal | A | 83.3% | 83.3% | 2.93s | 7.76s | 1362 | 53 | $0.003 |
| edge | minimal | B | 83.3% | **50.0%** | 1.56s | 2.28s | 2771 | 40 | $0.005 |

핵심: **의도분류 정확도는 두 arm 이 거의 항상 같다.** 갈리는 축은 결과 정확도이고,
reasoning 예산을 minimal 로 줄이면 B 만 붕괴한다(98.1% → 52.8%). A 는 계산을 파이썬이
하므로 분류만 맞으면 결과가 항상 맞는다.

그래프 실행 자체의 오버헤드는 평균 0.05ms — 지연 차이는 전부 API 구간에서 나온다.
