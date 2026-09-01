# 의도분류표 · 함수 골든라벨

- [intent_taxonomy.csv](intent_taxonomy.csv) — 9개 대분류 / 24개 소분류 정의
- [golden_labels.csv](golden_labels.csv) — 총 99건 다중 라벨
  - `work` 13건 (원본), `invest` 13건 (원본)
  - `ext_ipo` 9건, `ext_basis` 8건, `ext_index` 7건, `ext_tax` 6건, `ext_div` 6건, `ext_fx` 5건 (커버리지 보강 41건)
  - `ext_edge` 32건 (경계 케이스)

## 추가 함수: `get_news`

기존 함수셋에는 종목 한정 뉴스인 `get_stock_news`만 있어, "HBM 수요 증가", "전기차 수요 둔화", "미국 금리 인하" 같은 **종목이 특정되지 않는 시황·산업·매크로 뉴스**를 담을 함수가 없었습니다. 아래 시그니처를 전제로 라벨링했습니다.

```python
def get_news(
    query: Optional[str] = None,      # 주제/키워드 (예: "HBM 수요", "미국 금리 인하")
    base_d: Optional[str] = None,     # 시작일자 ('YYYYMMDD')
    end_d: Optional[str] = None,      # 종료일자 ('YYYYMMDD')
    index_list: Optional[List[str]] = None,  # 마켓/업종/테마명. None이면 전체
    top_k: int = 10
) -> List[Dict[str, str]]:
    """특정 종목에 한정되지 않는 시장 전반·업종/테마·매크로 뉴스를 제공합니다."""
    return [{"base_d": "20241017", "title": "제목", "content": "내용",
             "index_code": 111, "order": "1"}]
```

### `get_news` vs `get_stock_news` 판정 규칙

| 조건 | 함수 |
|---|---|
| 질의의 뉴스 대상이 **특정 종목의 사건**(자사주 매입, 유상증자, 외국인 매도, 신작 부재) | `get_stock_news` |
| 뉴스 대상이 **업종·테마·매크로**(HBM 수요, 방산 업황, 운임지수, 금리, 환율, IPO 시장 분위기) | `get_news` |
| 질의에 **종목명 + 테마가 함께** 등장하고 둘 다 근거가 필요 (예: "삼성전자 HBM 납품 전망") | **둘 다** |
| 종목명이 아예 없는 시황·정책 질의 | `get_news` 단독 |

이 규칙에 따라 기존 invest 13건 중 **9건(idx 0,1,2,3,5,8,9,10,11)에 `get_news`를 추가**했습니다. 종목 한정 이슈인 idx 4·6·7·12는 `get_stock_news`만 유지했습니다.

## 라벨링 규칙

1. **의도는 다중 라벨.** `intent_ids`의 첫 항목이 주 의도(primary), 이후는 부수 의도.
2. **함수도 다중 라벨.** 한 의도가 여러 함수를 요구할 수 있고, 여러 의도가 한 함수로 수렴할 수도 있다(사용 매뉴얼 2개 소분류 → `get_work_manual` 1개).
3. **의도 수 ≠ 함수 수.** 의도는 많은데 함수가 1개인 경우(`ext_div` idx 2, `ext_index` idx 0)와 그 반대(`invest` idx 0) 모두 존재.
4. **"전망/추정치"**: 목표가·컨센서스는 `get_financial_data.forcast.consensus`와 `get_company_evaluation.목표가격`에 중복 존재 → 두 함수 모두 정답 인정.
5. **"업황/테마" 언급 시** `get_sector` + `get_news` 동반.
6. **일반론 질의는 데이터 함수를 호출하지 않는다.** "고배당주 유의할 점"(`ext_div` idx 3)처럼 종목이 특정되지 않은 가이드성 질의는 `get_guide` 계열만 정답. 데이터 함수 호출 시 오답 처리.
7. **지수 단위는 `get_index_*`, 종목 단위는 `get_stock_*`.** 코스피/코스닥/업종/테마가 대상이면 `get_index_price`·`get_index_investor_trading`·`get_index_shorting_period`·`get_index_multiple_period`를 쓰고 `get_stock_*`을 쓰면 오답.

## 의도 → 허용 함수 매핑

| 대분류 | 허용 함수 |
|---|---|
| 기본적인 금융 지식 | `get_basic_financial_knowledge` |
| 규정 및 지침 | `get_guide_and_policy`, `get_guide`, `get_news`(정책 시행 이슈) |
| 사용 매뉴얼 | `get_work_manual`, `get_guide` |
| 기업 상장 및 IPO | `get_initial_listing`, `get_ipo_subscription_allocation`, `search_top_stocks_by_event(metric='IPO')`, `get_work_manual`, `get_news` |
| 시장 동향 및 분석 | `get_news`, `get_sector`, `get_index_price`, `get_index_investor_trading`, `get_index_multiple_period`, `get_index_shorting_period`, `search_top_sector_theme`, `search_top_stock` |
| 기업정보 | `get_basis_data`, `get_stock_news`, `get_news`, `get_announcement`, `get_financial_data`, `get_company_evaluation` |
| 시세 정보 | `get_stock_price`, `get_stock_investor_trading`, `get_stock_shorting_period`, `search_top_stock`, `search_top_stocks_by_event` |
| 평가 및 밸류에이션 | `get_stock_multiple_period`, `get_index_multiple_period`, `get_financial_data`, `get_company_evaluation` |
| 기타 | `get_exchange`, `get_news` |

## 보강분 커버리지

| 소분류 | 보강 전 | 보강 후 | 대표 샘플 |
|---|---|---|---|
| I4-1 IPO 기본 정보 | 0 | 4 | `ext_ipo` 0~3 |
| I4-2 IPO 일정 및 절차 | 0 | 5 | `ext_ipo` 4~8 |
| I2-1 세금 관련 정보 | 0 | 6 | `ext_tax` 0~5 |
| I6-5 배당 정보 | 0 | 6 | `ext_div` 0~5 |
| I9-1 기타 질의(환율) | 0 | 6 | `ext_fx` 0~4, `ext_index` 6 |
| I5-1 시장 전망(지수 단위) | 0 | 12 | `ext_index` 0~6 |
| I1-1 금융 개념 및 용어 | 0 | 5 | `ext_ipo` 7, `ext_tax` 1 등 |
| I6-1 기업 기본 정보 | 0 | 6 | `ext_basis` 0~2, 4, 7 |
| I6-2 기업 개요 | 0 | 4 | `ext_basis` 3, 5~7 |

`get_news`가 정답에 포함된 샘플은 총 18건(invest 9 + 보강분 9)으로, 함수별 등장 빈도 1위입니다.

## `ext_basis` — 과잉 호출 검증용

`get_basis_data`는 정적 프로필 조회라 **단일 함수로 끝나야 하는데 모델이 뉴스·시세를 덧붙이기 쉬운** 구간입니다. 8건 중 4건(idx 0~3)을 의도적으로 단일 함수 정답으로 두어, 과잉 호출을 페널티로 잡을 수 있게 했습니다.

주의할 중복 하나 — **상장일**은 `get_basis_data.상장일`과 `get_initial_listing.상장일자` 양쪽에 존재합니다(`ext_basis` idx 4). 두 함수 모두 정답 인정하되, 발행주식수처럼 현재 시점 값이 함께 요구되면 `get_basis_data`가 정본입니다.

## 전체 소분류 커버리지

24개 소분류, 25개 함수 **전부 1건 이상** 커버됩니다.

---

# `ext_edge` — 경계 케이스 세트 (32건)

라벨러/모델이 실제로 갈리는 지점만 모았습니다. 대부분 **표면 어휘는 거의 같은데 정답 함수가 다른 미니멀 페어**로 구성했고, `rationale` 앞에 `[축:X-이름]` 태그를 달아 오답을 축별로 슬라이싱할 수 있게 했습니다.

| 축 | 쟁점 | 건수 | 미니멀 페어 |
|---|---|---|---|
| A | 재무 정보 / 기업 분석 / 평가·전망 | 3 | 0·1·2 — 같은 종목·같은 지표, 시제와 '왜'만 다름 |
| B | `get_news` / `get_stock_news` | 4 | 3·4·5·6 — 종목만 / 테마만 / 둘 다 / 아무것도 없음 |
| C | `get_index_*` / `get_stock_*` | 4 | 9·30 — 코스피200 공매도 vs 삼성전자 공매도 |
| D | 현재 시점 / 상장 시점 | 2 | 10·11 — "액면가" vs "상장할 때 액면가" |
| E | 목표가 함수 중복 | 2 | 12·13 — 목표가(둘 다 인정) vs 투자점수(전용) |
| F | 매뉴얼 / 규정 / 가이드 / 용어 | 4 | 14~17 — "신용거래" 4-way |
| G | `search_top_stocks_by_event` / `search_top_stock` | 3 | 18·19 — "상한가" vs "상승률 상위" |
| H | 종목 랭킹 / 업종 랭킹 | 3 | 21·22 — "PER 낮은 종목" vs "PER 낮은 업종" |
| I | 일반론 / 스크리닝 | 1 | 23 — "PER 낮으면 좋은 거야?" (21과 어휘 유사, 정답 정반대) |
| J | 공시 / 뉴스 | 2 | 24·25 |
| K | 동일 함수 다중 호출 | 2 | 26·27 |
| L | 범위 밖 / 투자자문 | 2 | 28·29 |

## 경계 케이스에서 나온 추가 표기 규약

1. **`×N` = 호출 횟수.** `get_stock_multiple_period×2`는 종목별로 2회 호출해야 정답(idx 26). 함수 종류만 맞히고 1회만 호출하면 부분점수.
2. **`functions=none`, `n_functions=0` = 폴백이 정답.** 비트코인 시세(idx 28)처럼 커버리지 밖 질의는 **임의 함수 호출이 오답**입니다. 의도는 `I9-1`로 남겨 분류기와 라우터를 분리 채점합니다.
3. **과잉 호출도 오답.** `n_functions=1`인 샘플이 40건(전체의 40%)입니다. 초기 26건이 전부 다중 함수였던 탓에 "많이 부를수록 유리"한 편향이 있었는데, 이 세트로 상쇄됩니다.

## 채점 시 유의

- **A축은 함수가 겹칩니다.** idx 0·1·2 모두 `get_financial_data`를 포함하므로 함수 정확도만 보면 셋 다 맞은 것처럼 보입니다. **의도 라벨을 반드시 함께 채점**해야 이 축이 의미를 가집니다.
- **D·E축은 파라미터까지 봐야 합니다.** 함수는 맞고 필드가 틀리는 경우(액면가 vs 최초액면가)를 잡으려면 반환 필드 단위 채점이 필요합니다.
- **G축은 enum 배분이 원인**입니다. '상한가·급등·52주 신고가'는 `search_top_stocks_by_event`, '전일대비 상승율·수익율'은 `search_top_stock`에 갈려 있어 모델이 자주 틀립니다. 오답률이 높으면 라벨 문제가 아니라 **함수 설계를 합칠지 검토**할 신호입니다.
