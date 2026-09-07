from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="-",
)


@dataclass
class MetricsRegistry:
    """进程内轻量指标；生产环境可由采集器定期读取 snapshot。"""

    _counters: dict[str, int] = field(default_factory=dict)
    _durations_ms: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, duration_ms: float) -> None:
        with self._lock:
            values = self._durations_ms.setdefault(name, [])
            values.append(round(duration_ms, 3))
            if len(values) > 1000:
                del values[:-1000]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            durations = {
                name: self._summarize(values) for name, values in sorted(self._durations_ms.items())
            }
        return {"counters": counters, "durations_ms": durations}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._durations_ms.clear()

    @staticmethod
    def _summarize(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "avg": 0.0, "p95": 0.0, "max": 0.0}
        ordered = sorted(values)
        p95_index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return {
            "count": len(ordered),
            "avg": round(sum(ordered) / len(ordered), 3),
            "p95": ordered[p95_index],
            "max": ordered[-1],
        }


metrics = MetricsRegistry()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    correlation_id = request_id or str(uuid4())
    token = request_id_var.set(correlation_id)
    try:
        yield correlation_id
    finally:
        request_id_var.reset(token)


@contextmanager
def timed_metric(name: str) -> Iterator[None]:
    started_at = perf_counter()
    try:
        yield
    finally:
        metrics.observe(name, (perf_counter() - started_at) * 1000)


def observe_message_usage(message: Any) -> None:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return
    mappings = {
        "input_tokens": "model.input_tokens",
        "output_tokens": "model.output_tokens",
        "total_tokens": "model.total_tokens",
    }
    for source, metric_name in mappings.items():
        value = usage.get(source)
        if isinstance(value, int):
            metrics.increment(metric_name, value)
