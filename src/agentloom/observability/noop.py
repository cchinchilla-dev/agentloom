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


class NoopObserver:
    """Workflow observer with pass-through implementations of every hook.

    Use as a safe default instead of ``None`` so the engine can call
    ``self.observer.on_X(...)`` without ``if self.observer:`` guards or
    risking ``AttributeError`` when new hooks are added. Every hook accepts
    ``**kwargs`` so future additions don't break subclasses that inherit
    from this class.
    """

    def on_workflow_start(self, workflow_name: str, **kwargs: Any) -> None:
        pass

    def on_workflow_end(
        self,
        workflow_name: str,
        status: str,
        duration_ms: float,
        total_tokens: int,
        total_cost: float,
        **kwargs: Any,
    ) -> None:
        pass

    def on_step_start(
        self, step_id: str, step_type: str, stream: bool = False, **kwargs: Any
    ) -> None:
        pass

    def attach_step_event(self, step_id: str, event_name: str, attributes: dict[str, Any]) -> None:
        pass

    def on_step_end(
        self,
        step_id: str,
        step_type: str,
        status: str,
        duration_ms: float,
        cost_usd: float = 0.0,
        error: str | None = None,
        attachment_count: int = 0,
        time_to_first_token_ms: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        pass

    def on_provider_call_start(
        self, step_id: str, provider: str, model: str, **kwargs: Any
    ) -> None:
        pass

    def on_provider_call_end(
        self,
        step_id: str,
        provider: str,
        model: str,
        latency_s: float,
        **kwargs: Any,
    ) -> None:
        pass

    def on_provider_error(self, provider: str, error_type: str, **kwargs: Any) -> None:
        pass

    def on_tool_call(self, **kwargs: Any) -> None:
        pass

    def on_stream_response(self, provider: str, model: str, ttft_s: float, **kwargs: Any) -> None:
        pass

    def on_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        **kwargs: Any,
    ) -> None:
        pass

    def on_circuit_state_change(
        self, provider: str, old_state: str, new_state: str, **kwargs: Any
    ) -> None:
        pass

    def on_approval_gate(
        self, step_id: str, workflow_name: str, decision: str, **kwargs: Any
    ) -> None:
        pass

    def on_webhook_delivery(
        self,
        step_id: str,
        workflow_name: str,
        status: str,
        latency_s: float,
        **kwargs: Any,
    ) -> None:
        pass

    def on_mock_replay(
        self, workflow_name: str, step_id: str, matched_by: str, **kwargs: Any
    ) -> None:
        pass

    def on_recording_capture(
        self,
        step_id: str,
        provider: str,
        model: str,
        latency_s: float,
        **kwargs: Any,
    ) -> None:
        pass

    def on_budget_remaining(self, workflow: str, remaining: float, **kwargs: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass
