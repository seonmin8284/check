"""의도 경계에 걸친 케이스만 모은 셋 (v2 taxonomy).

각 항목은 '정답 의도의 어휘가 아니라 다른 의도의 어휘가 더 강하게 보이는' 발화다.
분류기가 표면 단어에 끌려가는지, 발화 구조를 보는지 가른다.
"""

EDGE_CASES_ANNOTATED: list[tuple[str, str, dict, str]] = [
    (
        "피보나치 말고 그냥 3, 5, 8 이 세 개 평균만",
        "mean", {"numbers": [3, 5, 8]},
        "부정 어휘: '피보나치'가 있으나 부정됐고 수열은 사용자가 줌 → fib_mean 아님",
    ),
    (
        "피보나치 10항까지 순서대로 보여줘",
        "fibonacci", {"n": 10},
        "'순서대로'가 정렬 요구처럼 보임 → fib_sorted 로 끌리는지",
    ),
    (
        "5! 이랑 피보나치 5번째 중에 뭐가 더 커?",
        "unsupported", {},
        "factorial/fibonacci 어휘가 둘 다 있으나 '비교'는 taxonomy 밖 → 과잉발동 여부",
    ),
    (
        "50 이하 소수 중 마지막 게 뭐야",
        "primes_last_k", {"n": 50, "k": 1},
        "'마지막 하나' → max 로 끌리기 쉬움. k=1 을 스스로 채워야 함",
    ),
    (
        "피보나치 12항까지에서 항이 몇 개나 되지",
        "unsupported", {},
        "fib+count 조합 계획이 없음 → 없는 의도를 지어내는지",
    ),
    (
        "36의 약수 개수 말고 약수들의 합",
        "divisors_sum", {"n": 36},
        "divisors_count 어휘가 먼저 나오고 부정됨 → 마지막 요구를 잡는지",
    ),
]

EDGE_CASES = [(u, i, s) for u, i, s, _ in EDGE_CASES_ANNOTATED]
NOTES = {u: note for u, _, _, note in EDGE_CASES_ANNOTATED}
