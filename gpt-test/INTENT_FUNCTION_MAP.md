# 의도분류표 · 함수 골든라벨

- [intent_taxonomy.csv](intent_taxonomy.csv) — 9개 대분류 / 24개 소분류 정의
- [golden_labels.csv](golden_labels.csv) — 총 67건 다중 라벨
  - `work` 13건 (원본), `invest` 13건 (원본)
  - `ext_ipo` 9건, `ext_basis` 8건, `ext_index` 7건, `ext_tax` 6건, `ext_div` 6건, `ext_fx` 5건 (보강분 41건)

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

24개 소분류 중 **23개가 1건 이상** 커버됩니다. 유일한 1건짜리는 `I6-7 기업 분석`(invest idx 0)인데, 이는 `I6-6 재무 정보`(9건)·`I6-8 기업 평가 및 전망`(9건)과 실무상 경계가 모호해 별도 보강보다 라벨 병합 검토를 권합니다.
