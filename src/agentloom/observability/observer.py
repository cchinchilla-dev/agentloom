"""WorkflowObserver — wires TracingManager + MetricsManager into engine execution.

This is the bridge between the core engine (which never imports observability
directly) and the optional tracing/metrics backends.  The CLI creates an
observer and injects it into WorkflowEngine via its ``observer`` parameter.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agentloom.observability")


class WorkflowObserver:
    """Observes workflow execution, emitting OTel spans and Prometheus metrics.

    Both *tracing* and *metrics* are optional — pass ``None`` to disable either.

    Note: Hook signatures may gain new keyword arguments with defaults in
    future versions.  Subclasses that override hooks should accept ``**kwargs``
    for forward compatibility.
    """

    def __init__(
        self,
        tracing: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        self._tracing = tracing
        self._metrics = metrics
        self._workflow_span: Any | None = None
        self._step_spans: dict[str, Any] = {}

    def on_workflow_start(self, workflow_name: str, **kwargs: Any) -> None:
        if self._tracing:
            self._workflow_span = self._tracing.start_span(
                f"workflow:{workflow_name}",
                attributes={"workflow.name": workflow_name},
            )

    def on_workflow_end(
        self,
        workflow_name: str,
        status: str,
        duration_ms: float,
        total_tokens: int,
        total_cost: float,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_workflow_run(
                workflow_name, status, duration_ms / 1000.0, total_cost
            )

        span = self._workflow_span
        if span:
            span.set_attribute("workflow.status", status)
            span.set_attribute("workflow.duration_ms", duration_ms)
            span.set_attribute("workflow.total_tokens", total_tokens)
            span.set_attribute("workflow.total_cost_usd", total_cost)
            if self._tracing:
                self._tracing.end_span(span)
            else:
                span.end()
            self._workflow_span = None

    def on_step_start(
        self, step_id: str, step_type: str, stream: bool = False, **kwargs: Any
    ) -> None:
        if self._tracing:
            span = self._tracing.start_span(
                f"step:{step_id}",
                attributes={
                    "step.id": step_id,
                    "step.type": step_type,
                    "step.stream": stream,
                },
            )
            self._step_spans[step_id] = span

    def on_step_end(
        self,
        step_id: str,
        step_type: str,
        status: str,
        duration_ms: float,
        cost_usd: float = 0.0,
        tokens: int = 0,
        error: str | None = None,
        attachment_count: int = 0,
        time_to_first_token_ms: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_step_execution(
                step_type, status, duration_ms / 1000.0, stream=stream
            )
            if attachment_count > 0:
                self._metrics.record_attachments(step_type, attachment_count)

        span = self._step_spans.pop(step_id, None)
        if span:
            span.set_attribute("step.status", status)
            span.set_attribute("step.duration_ms", duration_ms)
            span.set_attribute("step.cost_usd", cost_usd)
            span.set_attribute("step.tokens", tokens)
            if attachment_count > 0:
                span.set_attribute("step.attachments", attachment_count)
            if time_to_first_token_ms is not None:
                span.set_attribute("step.time_to_first_token_ms", time_to_first_token_ms)
            if error:
                span.set_attribute("step.error", error)
            if self._tracing:
                self._tracing.end_span(span)
            else:
                span.end()

    # Provider-level events (called from gateway if observer is attached)

    def on_provider_call(
        self, provider: str, model: str, latency_s: float, stream: bool = False
    ) -> None:
        if self._metrics:
            self._metrics.record_provider_call(provider, model, latency_s, stream=stream)

    def on_provider_error(self, provider: str, error_type: str) -> None:
        if self._metrics:
            self._metrics.record_provider_error(provider, error_type)

    def on_stream_response(self, provider: str, model: str, ttft_s: float) -> None:
        if self._metrics:
            self._metrics.record_stream_response(provider, model)
            self._metrics.record_time_to_first_token(provider, model, ttft_s)

    def on_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if self._metrics:
            self._metrics.record_tokens(provider, model, prompt_tokens, completion_tokens)

    # Circuit breaker events (called from gateway via callback)

    def on_circuit_state_change(self, provider: str, old_state: str, new_state: str) -> None:
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        state_int = state_map.get(str(new_state), 0)
        logger.info(
            "Circuit breaker '%s': %s -> %s",
            provider,
            old_state,
            new_state,
        )
        if self._metrics:
            self._metrics.set_circuit_state(provider, state_int)

    def on_approval_gate(self, step_id: str, workflow_name: str, decision: str) -> None:
        if self._metrics:
            self._metrics.record_approval_gate(workflow_name, decision)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute("approval_gate.decision", decision)

    def on_webhook_delivery(
        self, step_id: str, workflow_name: str, status: str, latency_s: float
    ) -> None:
        if self._metrics:
            self._metrics.record_webhook_delivery(workflow_name, status, latency_s)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute("webhook.status", status)
            span.set_attribute("webhook.latency_s", latency_s)

    def on_mock_replay(self, workflow_name: str, step_id: str, matched_by: str) -> None:
        if self._metrics:
            self._metrics.record_mock_replay(workflow_name, matched_by)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute("mock.matched_by", matched_by)
            if step_id:
                span.set_attribute("mock.step_id", step_id)

    def on_recording_capture(
        self, step_id: str, provider: str, model: str, latency_s: float
    ) -> None:
        if self._metrics:
            self._metrics.record_recording_capture(provider, model, latency_s)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute("recording.provider", provider)
            span.set_attribute("recording.model", model)
            span.set_attribute("recording.latency_s", latency_s)

    def on_budget_remaining(self, workflow: str, remaining: float) -> None:
        if self._metrics:
            self._metrics.set_budget_remaining(workflow, remaining)

    def shutdown(self) -> None:
        if self._metrics:
            shutdown = getattr(self._metrics, "shutdown", None)
            if shutdown:
                shutdown()
        if self._tracing:
            self._tracing.shutdown()
