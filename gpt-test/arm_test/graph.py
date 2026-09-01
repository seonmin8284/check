"""결정적 capability graph 실행기.

LLM 은 여기 관여하지 않는다. (intent, slots) 가 주어지면 출력은 항상 동일하다.
"""

from dataclasses import dataclass, field
from typing import Any

from capabilities import CAPABILITIES, PLANS


@dataclass
class GraphRun:
    intent: str
    ok: bool
    result: Any = None
    trace: list[str] = field(default_factory=list)   # 실행된 노드 순서
    bindings: list[dict] = field(default_factory=list)  # 노드별 실제 인자
    error: str | None = None


def _coerce(value: Any, tag: str) -> Any:
    """슬롯 값을 포트 타입에 맞춘다. 실패하면 예외."""
    if tag == "int":
        return int(value)
    if tag == "bool":
        return bool(value)
    if tag == "int[]":
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"수열이 필요한데 {type(value).__name__} 를 받음")
        return [int(v) if float(v).is_integer() else float(v) for v in value]
    return value


def execute(intent: str, slots: dict) -> GraphRun:
    if intent not in PLANS:
        return GraphRun(intent, ok=False, error=f"알 수 없는 의도: {intent}")

    plan = PLANS[intent]
    if not plan:  # unsupported
        return GraphRun(intent, ok=True, result=None, trace=[])

    run = GraphRun(intent, ok=True)
    outputs: list[tuple[str, Any]] = []  # (타입 태그, 값) — 상류 출력 스택

    for node_name in plan:
        cap = CAPABILITIES[node_name]
        kwargs: dict[str, Any] = {}

        for param, tag in cap.params.items():
            # 상류에 타입이 맞는 출력이 있으면 그 간선이 항상 이긴다.
            # (그래프 구조가 결정적이어야 하므로 분류기 슬롯이 배선을 덮어쓰지 못한다)
            upstream = next((v for t, v in reversed(outputs) if t == tag), None)
            if upstream is not None:
                kwargs[param] = upstream
                continue

            raw = slots.get(param)
            if raw is not None:
                try:
                    kwargs[param] = _coerce(raw, tag)
                except (TypeError, ValueError) as e:
                    run.ok = False
                    run.error = f"{node_name}.{param} 슬롯 변환 실패: {e}"
                    return run
            elif param not in cap.optional:
                run.ok = False
                run.error = f"{node_name}.{param} 를 채울 슬롯도 상류 출력도 없음"
                return run

        try:
            value = cap.fn(**kwargs)
        except Exception as e:
            run.ok = False
            run.error = f"{node_name} 실행 실패: {type(e).__name__}: {e}"
            return run

        run.trace.append(node_name)
        run.bindings.append({"node": node_name, "args": kwargs})
        outputs.append((cap.produces, value))
        run.result = value

    return run
