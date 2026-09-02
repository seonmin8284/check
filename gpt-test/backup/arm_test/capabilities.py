"""Capability 정의 — 순수 결정적 파이썬 함수 + 타입 포트.

각 capability 는 타입 태그가 붙은 입력 포트와 하나의 출력 타입을 선언한다.
그래프 실행기는 이 타입 정보만으로 노드 간 배선을 결정적으로 해결한다.

v2: 의도 11개 -> 42개로 확장. 집계 연산을 mean/sum/median/max/min/count/spread 로
    잘게 쪼개 분류 난이도를 올렸다.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# 소스 — 수열/스칼라를 만들어내는 노드
# ---------------------------------------------------------------------------


def fibonacci(n: int) -> list[int]:
    """F[0]..F[n]."""
    if n < 0:
        raise ValueError("n은 0 이상이어야 합니다.")
    seq = [0, 1]
    for _ in range(n - 1):
        seq.append(seq[-1] + seq[-2])
    return seq[: n + 1]


def primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = [False] * len(sieve[i * i :: i])
    return [i for i, ok in enumerate(sieve) if ok]


def collatz(n: int) -> list[int]:
    """n 에서 1 까지의 콜라츠 궤적 (양끝 포함)."""
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다.")
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq


def divisors(n: int) -> list[int]:
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다.")
    return [d for d in range(1, n + 1) if n % d == 0]


def prime_factors(n: int) -> list[int]:
    """중복 포함 소인수 목록."""
    if n < 2:
        return []
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def squares(n: int) -> list[int]:
    """1^2 .. n^2."""
    return [i * i for i in range(1, n + 1)]


def triangular(n: int) -> list[int]:
    """앞에서부터 n 개의 삼각수."""
    return [i * (i + 1) // 2 for i in range(1, n + 1)]


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n은 0 이상이어야 합니다.")
    return math.factorial(n)


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    return math.lcm(a, b)


# ---------------------------------------------------------------------------
# 집계 — 수열을 스칼라/객체로 접는 노드 (여기가 분류 난이도의 핵심)
# ---------------------------------------------------------------------------


def _need(numbers: list) -> list:
    if not numbers:
        raise ValueError("빈 수열입니다.")
    return numbers


def mean_of(numbers: list[float]) -> float:
    return round(statistics.mean(_need(numbers)), 4)


def sum_of(numbers: list[float]) -> float:
    return sum(numbers)


def median_of(numbers: list[float]) -> float:
    return statistics.median(_need(numbers))


def max_of(numbers: list[float]) -> float:
    return max(_need(numbers))


def min_of(numbers: list[float]) -> float:
    return min(_need(numbers))


def count_of(numbers: list[float]) -> int:
    return len(numbers)


def spread_of(numbers: list[float]) -> dict:
    """산포도 — 범위/표본분산/표본표준편차."""
    _need(numbers)
    if len(numbers) < 2:
        return {"range": 0, "variance": 0.0, "stdev": 0.0}
    return {
        "range": max(numbers) - min(numbers),
        "variance": round(statistics.variance(numbers), 4),
        "stdev": round(statistics.stdev(numbers), 4),
    }


# ---------------------------------------------------------------------------
# 변환 — 수열을 수열로 바꾸는 노드
# ---------------------------------------------------------------------------


def filter_primes(numbers: list[int]) -> list[int]:
    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        return all(x % d for d in range(2, int(x**0.5) + 1))

    return [x for x in numbers if is_prime(int(x))]


def sort_numbers(numbers: list[float], desc: bool = False) -> list[float]:
    return sorted(numbers, reverse=bool(desc))


def unique_of(numbers: list[float]) -> list[float]:
    """등장 순서를 보존한 중복 제거."""
    seen, out = set(), []
    for x in numbers:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def evens_of(numbers: list[int]) -> list[int]:
    return [x for x in numbers if int(x) % 2 == 0]


def odds_of(numbers: list[int]) -> list[int]:
    return [x for x in numbers if int(x) % 2 != 0]


def cumsum_of(numbers: list[float]) -> list[float]:
    out, acc = [], 0
    for x in numbers:
        acc += x
        out.append(acc)
    return out


def last_k(numbers: list[float], k: int) -> list[float]:
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    return numbers[-k:]


# ---------------------------------------------------------------------------
# 포트 스키마
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    name: str
    fn: Callable
    params: dict[str, str]      # 파라미터명 -> 타입 태그
    optional: frozenset[str]
    produces: str


def _cap(name, fn, params, produces, optional=()):
    return Capability(name, fn, params, frozenset(optional), produces)


_NUMS = {"numbers": "int[]"}

CAPABILITIES: dict[str, Capability] = {
    c.name: c
    for c in [
        # 소스
        _cap("fibonacci", fibonacci, {"n": "int"}, "int[]"),
        _cap("primes_upto", primes_upto, {"n": "int"}, "int[]"),
        _cap("collatz", collatz, {"n": "int"}, "int[]"),
        _cap("divisors", divisors, {"n": "int"}, "int[]"),
        _cap("prime_factors", prime_factors, {"n": "int"}, "int[]"),
        _cap("squares", squares, {"n": "int"}, "int[]"),
        _cap("triangular", triangular, {"n": "int"}, "int[]"),
        _cap("factorial", factorial, {"n": "int"}, "int"),
        _cap("gcd", gcd, {"a": "int", "b": "int"}, "int"),
        _cap("lcm", lcm, {"a": "int", "b": "int"}, "int"),
        # 집계
        _cap("mean_of", mean_of, dict(_NUMS), "num"),
        _cap("sum_of", sum_of, dict(_NUMS), "num"),
        _cap("median_of", median_of, dict(_NUMS), "num"),
        _cap("max_of", max_of, dict(_NUMS), "num"),
        _cap("min_of", min_of, dict(_NUMS), "num"),
        _cap("count_of", count_of, dict(_NUMS), "num"),
        _cap("spread_of", spread_of, dict(_NUMS), "dict"),
        # 변환
        _cap("filter_primes", filter_primes, dict(_NUMS), "int[]"),
        _cap("sort_numbers", sort_numbers, {**_NUMS, "desc": "bool"}, "int[]", ["desc"]),
        _cap("unique_of", unique_of, dict(_NUMS), "int[]"),
        _cap("evens_of", evens_of, dict(_NUMS), "int[]"),
        _cap("odds_of", odds_of, dict(_NUMS), "int[]"),
        _cap("cumsum_of", cumsum_of, dict(_NUMS), "int[]"),
        _cap("last_k", last_k, {**_NUMS, "k": "int"}, "int[]"),
    ]
}


# ---------------------------------------------------------------------------
# 의도 -> 실행 계획(DAG)
# ---------------------------------------------------------------------------

PLANS: dict[str, list[str]] = {
    # 소스 단독
    "fibonacci": ["fibonacci"],
    "primes": ["primes_upto"],
    "collatz": ["collatz"],
    "divisors": ["divisors"],
    "prime_factors": ["prime_factors"],
    "squares": ["squares"],
    "triangular": ["triangular"],
    "factorial": ["factorial"],
    "gcd": ["gcd"],
    "lcm": ["lcm"],
    # 사용자가 준 수열 -> 집계 (세분화된 구간)
    "mean": ["mean_of"],
    "sum": ["sum_of"],
    "median": ["median_of"],
    "max": ["max_of"],
    "min": ["min_of"],
    "count": ["count_of"],
    "spread": ["spread_of"],
    # 사용자가 준 수열 -> 변환
    "sort": ["sort_numbers"],
    "unique": ["unique_of"],
    "evens": ["evens_of"],
    "odds": ["odds_of"],
    "cumsum": ["cumsum_of"],
    "last_k": ["last_k"],
    # 피보나치 복합
    "fib_mean": ["fibonacci", "mean_of"],
    "fib_sum": ["fibonacci", "sum_of"],
    "fib_median": ["fibonacci", "median_of"],
    "fib_max": ["fibonacci", "max_of"],
    "fib_primes": ["fibonacci", "filter_primes"],
    "fib_sorted": ["fibonacci", "sort_numbers"],
    "fib_evens": ["fibonacci", "evens_of"],
    "fib_cumsum": ["fibonacci", "cumsum_of"],
    "fib_last_k": ["fibonacci", "last_k"],
    "fib_evens_sum": ["fibonacci", "evens_of", "sum_of"],
    # 소수 복합
    "primes_mean": ["primes_upto", "mean_of"],
    "primes_sum": ["primes_upto", "sum_of"],
    "primes_count": ["primes_upto", "count_of"],
    "primes_spread": ["primes_upto", "spread_of"],
    "primes_last_k": ["primes_upto", "last_k"],
    # 콜라츠 / 약수 복합
    "collatz_len": ["collatz", "count_of"],
    "collatz_max": ["collatz", "max_of"],
    "divisors_sum": ["divisors", "sum_of"],
    "divisors_count": ["divisors", "count_of"],
    "unsupported": [],
}

INTENTS: list[str] = list(PLANS)

INTENT_DOCS: dict[str, str] = {
    "fibonacci": "피보나치 수열 F[0]..F[n] 자체. 슬롯: n",
    "primes": "n 이하 소수 목록. 슬롯: n",
    "collatz": "n 에서 1 까지의 콜라츠 궤적. 슬롯: n",
    "divisors": "n 의 약수 목록. 슬롯: n",
    "prime_factors": "n 의 소인수분해(중복 포함). 슬롯: n",
    "squares": "1^2..n^2 제곱수 목록. 슬롯: n",
    "triangular": "앞에서부터 n 개의 삼각수. 슬롯: n",
    "factorial": "n! 값 하나. 슬롯: n",
    "gcd": "두 수의 최대공약수. 슬롯: a, b",
    "lcm": "두 수의 최소공배수. 슬롯: a, b",
    "mean": "사용자가 준 수열의 '평균'만. 슬롯: numbers",
    "sum": "사용자가 준 수열의 '합'만. 슬롯: numbers",
    "median": "사용자가 준 수열의 '중앙값'만. 슬롯: numbers",
    "max": "사용자가 준 수열의 '최댓값'만. 슬롯: numbers",
    "min": "사용자가 준 수열의 '최솟값'만. 슬롯: numbers",
    "count": "사용자가 준 수열의 '개수'만. 슬롯: numbers",
    "spread": "사용자가 준 수열의 '산포도'(범위/분산/표준편차). 슬롯: numbers",
    "sort": "사용자가 준 수열 정렬. 슬롯: numbers, desc",
    "unique": "사용자가 준 수열 중복 제거. 슬롯: numbers",
    "evens": "사용자가 준 수열 중 짝수만. 슬롯: numbers",
    "odds": "사용자가 준 수열 중 홀수만. 슬롯: numbers",
    "cumsum": "사용자가 준 수열의 누적합 수열. 슬롯: numbers",
    "last_k": "사용자가 준 수열의 마지막 k 개. 슬롯: numbers, k",
    "fib_mean": "피보나치 수열의 평균. 슬롯: n",
    "fib_sum": "피보나치 수열의 합. 슬롯: n",
    "fib_median": "피보나치 수열의 중앙값. 슬롯: n",
    "fib_max": "피보나치 수열의 최댓값. 슬롯: n",
    "fib_primes": "피보나치 수열 중 소수만. 슬롯: n",
    "fib_sorted": "피보나치 수열을 정렬. 슬롯: n, desc",
    "fib_evens": "피보나치 수열 중 짝수만. 슬롯: n",
    "fib_cumsum": "피보나치 수열의 누적합. 슬롯: n",
    "fib_last_k": "피보나치 수열의 마지막 k 개. 슬롯: n, k",
    "fib_evens_sum": "피보나치 수열 중 짝수만 골라 그 합. 슬롯: n",
    "primes_mean": "n 이하 소수들의 평균. 슬롯: n",
    "primes_sum": "n 이하 소수들의 합. 슬롯: n",
    "primes_count": "n 이하 소수의 개수. 슬롯: n",
    "primes_spread": "n 이하 소수들의 산포도. 슬롯: n",
    "primes_last_k": "n 이하 소수 중 마지막 k 개. 슬롯: n, k",
    "collatz_len": "콜라츠 궤적의 길이(항 개수). 슬롯: n",
    "collatz_max": "콜라츠 궤적의 최고점. 슬롯: n",
    "divisors_sum": "n 의 약수 총합. 슬롯: n",
    "divisors_count": "n 의 약수 개수. 슬롯: n",
    "unsupported": "위 어디에도 해당하지 않음 (비교/날씨/번역/잡담 등)",
}

assert set(INTENT_DOCS) == set(PLANS), "INTENT_DOCS 와 PLANS 가 어긋남"


def taxonomy_text() -> str:
    """두 arm 이 공유하는 의도 목록 (분류 조건을 동일하게 맞추기 위함)."""
    return "\n".join(f"- {name}: {doc}" for name, doc in INTENT_DOCS.items())
