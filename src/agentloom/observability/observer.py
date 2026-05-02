"""WorkflowObserver — wires TracingManager + MetricsManager into engine execution.

This is the bridge between the core engine (which never imports observability
directly) and the optional tracing/metrics backends.  The CLI creates an
observer and injects it into WorkflowEngine via its ``observer`` parameter.
"""

from __future__ import annotations

import logging
from typing import Any

from agentloom.observability.schema import (
    GenAIOperationName,
    SpanAttr,
    SpanName,
    to_genai_provider_name,
)

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
        # Keyed by (step_id, attempt). One step can produce multiple
        # provider spans when the gateway falls back across providers.
        self._provider_spans: dict[tuple[str, int], Any] = {}
        self._run_id: str = ""
        self._workflow_name: str = ""

    def on_workflow_start(self, workflow_name: str, *, run_id: str = "", **kwargs: Any) -> None:
        self._run_id = run_id
        self._workflow_name = workflow_name
        if self._tracing:
            attrs: dict[str, Any] = {SpanAttr.WORKFLOW_NAME: workflow_name}
            if run_id:
                attrs[SpanAttr.WORKFLOW_RUN_ID] = run_id
            self._workflow_span = self._tracing.start_span(
                SpanName.WORKFLOW.format(workflow_name=workflow_name),
                attributes=attrs,
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
            span.set_attribute(SpanAttr.WORKFLOW_STATUS, status)
            span.set_attribute(SpanAttr.WORKFLOW_DURATION_MS, duration_ms)
            span.set_attribute(SpanAttr.WORKFLOW_TOTAL_TOKENS, total_tokens)
            span.set_attribute(SpanAttr.WORKFLOW_TOTAL_COST_USD, total_cost)
            if self._run_id:
                span.set_attribute(SpanAttr.WORKFLOW_RUN_ID, self._run_id)
            if self._tracing:
                self._tracing.end_span(span)
            else:
                span.end()
            self._workflow_span = None

    def on_step_start(
        self,
        step_id: str,
        step_type: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        if self._tracing:
            attrs: dict[str, Any] = {
                SpanAttr.STEP_ID: step_id,
                SpanAttr.STEP_TYPE: step_type,
                SpanAttr.STEP_STREAM: stream,
            }
            if self._run_id:
                attrs[SpanAttr.WORKFLOW_RUN_ID] = self._run_id
            span = self._tracing.start_span(
                SpanName.STEP.format(step_id=step_id),
                attributes=attrs,
            )
            self._step_spans[step_id] = span

    def attach_step_event(self, step_id: str, event_name: str, attributes: dict[str, Any]) -> None:
        """Attach an event with arbitrary attributes to a live step span.

        Used by the opt-in prompt-capture path in ``llm_call`` so the full
        rendered prompt rides on the span as an OTel event rather than a
        fat attribute (event payloads aren't subject to attribute-size
        limits and stay easy to filter out at the collector).
        """
        span = self._step_spans.get(step_id)
        if span is None:
            return
        add_event = getattr(span, "add_event", None)
        if callable(add_event):
            add_event(event_name, attributes)

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
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        model: str | None = None,
        provider: str | None = None,
        finish_reason: str | None = None,
        prompt_hash: str | None = None,
        prompt_length_chars: int | None = None,
        prompt_template_id: str | None = None,
        prompt_template_vars: str | None = None,
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
            span.set_attribute(SpanAttr.STEP_STATUS, status)
            span.set_attribute(SpanAttr.STEP_DURATION_MS, duration_ms)
            span.set_attribute(SpanAttr.STEP_COST_USD, cost_usd)
            if prompt_tokens or completion_tokens:
                span.set_attribute(SpanAttr.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
                span.set_attribute(SpanAttr.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
            if reasoning_tokens:
                span.set_attribute(SpanAttr.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning_tokens)
            if model:
                span.set_attribute(SpanAttr.GEN_AI_REQUEST_MODEL, model)
            if provider:
                # AgentLoom provider names (``google``) translate to the
                # OTel registry value (``gcp.gemini``).
                span.set_attribute(SpanAttr.GEN_AI_PROVIDER_NAME, to_genai_provider_name(provider))
                # ``gen_ai.operation.name`` is required on inference spans;
                # AgentLoom ``llm_call`` always corresponds to ``chat``.
                if step_type == "llm_call":
                    span.set_attribute(
                        SpanAttr.GEN_AI_OPERATION_NAME, GenAIOperationName.CHAT.value
                    )
            if finish_reason:
                # Spec mandates an array of strings (allows multi-reason
                # responses). Wrap a single reason rather than emitting a
                # bare string.
                span.set_attribute(SpanAttr.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
            if attachment_count > 0:
                span.set_attribute(SpanAttr.STEP_ATTACHMENTS, attachment_count)
            if time_to_first_token_ms is not None:
                span.set_attribute(
                    SpanAttr.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
                    time_to_first_token_ms / 1000.0,
                )
            if prompt_hash:
                span.set_attribute(SpanAttr.PROMPT_HASH, prompt_hash)
            if prompt_length_chars is not None:
                span.set_attribute(SpanAttr.PROMPT_LENGTH_CHARS, prompt_length_chars)
            if prompt_template_id:
                span.set_attribute(SpanAttr.PROMPT_TEMPLATE_ID, prompt_template_id)
            if prompt_template_vars:
                span.set_attribute(SpanAttr.PROMPT_TEMPLATE_VARS, prompt_template_vars)
            if error:
                span.set_attribute(SpanAttr.STEP_ERROR, error)
            if self._tracing:
                self._tracing.end_span(span)
            else:
                span.end()

    # Provider-level events (called from gateway if observer is attached)

    def on_provider_call_start(
        self,
        step_id: str,
        provider: str,
        model: str,
        *,
        attempt: int = 0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        """Open a child span around a provider HTTP call.

        Creates a ``provider:<name>`` span nested under the current step so
        the trace tree reflects the ``workflow > step > provider`` hierarchy.
        ``attempt`` distinguishes fallback retries: a workflow that hits
        provider A and falls back to provider B emits two spans for the
        same ``step_id``. Must be balanced by ``on_provider_call_end``.
        """
        if not self._tracing:
            return
        # Provider spans ARE the GenAI inference spans — name and attributes
        # follow the canonical OTel convention so Jaeger / Grafana GenAI
        # dashboards correlate them out of the box.
        operation = GenAIOperationName.CHAT.value
        attrs: dict[str, Any] = {
            SpanAttr.GEN_AI_OPERATION_NAME: operation,
            SpanAttr.GEN_AI_PROVIDER_NAME: to_genai_provider_name(provider),
            SpanAttr.GEN_AI_REQUEST_MODEL: model,
            # Provider span IS the GenAI inference span — use the canonical
            # ``gen_ai.request.stream`` here. Step-level orchestration spans
            # keep ``step.stream`` since that's an AgentLoom concept.
            SpanAttr.GEN_AI_REQUEST_STREAM: stream,
            SpanAttr.PROVIDER_ATTEMPT: attempt,
        }
        if self._run_id:
            attrs[SpanAttr.WORKFLOW_RUN_ID] = self._run_id
        if temperature is not None:
            attrs[SpanAttr.GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[SpanAttr.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
        span = self._tracing.start_span(
            SpanName.GEN_AI_INFERENCE.format(operation_name=operation, model=model),
            attributes=attrs,
        )
        self._provider_spans[(step_id, attempt)] = span

    def on_provider_call_end(
        self,
        step_id: str,
        provider: str,
        model: str,
        latency_s: float,
        *,
        attempt: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        finish_reason: str | None = None,
        stream: bool = False,
        error: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_provider_call(provider, model, latency_s, stream=stream)
        span = self._provider_spans.pop((step_id, attempt), None)
        if span is None:
            return
        span.set_attribute(SpanAttr.GEN_AI_RESPONSE_MODEL, model)
        if prompt_tokens or completion_tokens:
            span.set_attribute(SpanAttr.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
            span.set_attribute(SpanAttr.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
        if reasoning_tokens:
            span.set_attribute(SpanAttr.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning_tokens)
        if finish_reason:
            span.set_attribute(SpanAttr.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        if error:
            span.set_attribute(SpanAttr.STEP_ERROR, error)
            # ``error.type`` is the OTel general-conventions key, conditionally
            # required on errored spans. Emitted alongside ``step.error`` so
            # OTel-aware consumers (Jaeger error filters, Tempo) light up.
            span.set_attribute(SpanAttr.ERROR_TYPE, error)
            span.set_attribute(SpanAttr.PROVIDER_ATTEMPT_OUTCOME, "error")
        else:
            span.set_attribute(SpanAttr.PROVIDER_ATTEMPT_OUTCOME, "ok")
        if self._tracing:
            self._tracing.end_span(span)
        else:
            span.end()

    def on_provider_error(
        self,
        provider: str,
        error_type: str,
        *,
        step_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_provider_error(provider, error_type)
        # If there's a dangling provider span for this step, close it with
        # the error recorded so traces don't leak open spans. Iterates all
        # attempts because the gateway may have failed mid-fallback.
        if step_id is not None:
            stale = [k for k in self._provider_spans if k[0] == step_id]
            for key in stale:
                span = self._provider_spans.pop(key, None)
                if span is not None:
                    span.set_attribute(SpanAttr.STEP_ERROR, error_type)
                    span.set_attribute(SpanAttr.ERROR_TYPE, error_type)
                    if self._tracing:
                        self._tracing.end_span(span)
                    else:
                        span.end()

    def on_stream_response(self, provider: str, model: str, ttft_s: float, **kwargs: Any) -> None:
        if self._metrics:
            self._metrics.record_stream_response(provider, model)
            self._metrics.record_time_to_first_token(provider, model, ttft_s)

    def on_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        reasoning_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_tokens(
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens=reasoning_tokens,
            )

    # Circuit breaker events (called from gateway via callback)

    def on_circuit_state_change(
        self, provider: str, old_state: str, new_state: str, **kwargs: Any
    ) -> None:
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

    def on_approval_gate(
        self, step_id: str, workflow_name: str, decision: str, **kwargs: Any
    ) -> None:
        if self._metrics:
            self._metrics.record_approval_gate(workflow_name, decision)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute(SpanAttr.APPROVAL_DECISION, decision)

    def on_webhook_delivery(
        self,
        step_id: str,
        workflow_name: str,
        status: str,
        latency_s: float,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_webhook_delivery(workflow_name, status, latency_s)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute(SpanAttr.WEBHOOK_STATUS, status)
            span.set_attribute(SpanAttr.WEBHOOK_LATENCY_S, latency_s)

    def on_mock_replay(
        self, workflow_name: str, step_id: str, matched_by: str, **kwargs: Any
    ) -> None:
        if self._metrics:
            self._metrics.record_mock_replay(workflow_name, matched_by)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute(SpanAttr.MOCK_MATCHED_BY, matched_by)
            if step_id:
                span.set_attribute(SpanAttr.MOCK_STEP_ID, step_id)

    def on_recording_capture(
        self,
        step_id: str,
        provider: str,
        model: str,
        latency_s: float,
        **kwargs: Any,
    ) -> None:
        if self._metrics:
            self._metrics.record_recording_capture(provider, model, latency_s)
        span = self._step_spans.get(step_id)
        if span:
            span.set_attribute(SpanAttr.RECORDING_PROVIDER, provider)
            span.set_attribute(SpanAttr.RECORDING_MODEL, model)
            span.set_attribute(SpanAttr.RECORDING_LATENCY_S, latency_s)

    def on_budget_remaining(self, workflow: str, remaining: float, **kwargs: Any) -> None:
        if self._metrics:
            self._metrics.set_budget_remaining(workflow, remaining)

    def shutdown(self) -> None:
        if self._metrics:
            shutdown = getattr(self._metrics, "shutdown", None)
            if shutdown:
                shutdown()
        if self._tracing:
            self._tracing.shutdown()
