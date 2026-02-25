"""No-op implementations for observability interfaces.

Used as fallbacks when OpenTelemetry / Prometheus are not installed.
"""

from __future__ import annotations

from typing import Any


class NoopSpan:
    """A span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> NoopSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class NoopTracer:
    """A tracer that produces no-op spans."""

    def start_span(self, name: str, **kwargs: Any) -> NoopSpan:
        return NoopSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> NoopSpan:
        return NoopSpan()


class NoopCounter:
    """A counter that does nothing."""

    def add(self, amount: int = 1, attributes: dict[str, Any] | None = None) -> None:
        pass


class NoopHistogram:
    """A histogram that does nothing."""

    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        pass


class NoopGauge:
    """A gauge that does nothing."""

    def set(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        pass


class NoopMeter:
    """A meter that produces no-op instruments."""

    def create_counter(self, name: str, **kwargs: Any) -> NoopCounter:
        return NoopCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> NoopHistogram:
        return NoopHistogram()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> NoopCounter:
        return NoopCounter()
