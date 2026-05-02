"""Tests for WorkflowObserver."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentloom.observability.observer import WorkflowObserver


class TestWorkflowLifecycle:
    def test_on_workflow_start_creates_span(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("test-wf")
        tracing.start_span.assert_called_once()
        assert "test-wf" in str(tracing.start_span.call_args)

    def test_on_workflow_end_records_metrics(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_workflow_end("test-wf", "success", 1000.0, 100, 0.01)
        metrics.record_workflow_run.assert_called_once_with("test-wf", "success", 1.0, 0.01)

    def test_on_workflow_end_ends_span(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("test-wf")
        span = tracing.start_span.return_value
        observer.on_workflow_end("test-wf", "success", 500.0, 50, 0.005)
        tracing.end_span.assert_called_once_with(span)

    def test_no_tracing_no_error(self) -> None:
        observer = WorkflowObserver()
        observer.on_workflow_start("test")
        observer.on_workflow_end("test", "success", 100.0, 10, 0.001)

    def test_workflow_end_without_tracing_uses_span_end(self) -> None:
        # No Tracing wrapper: observer must still close the workflow span
        # via ``span.end()`` directly so spans don't leak when tracing is off.
        observer = WorkflowObserver()
        span = MagicMock()
        observer._workflow_span = span
        observer.on_workflow_end("wf", "success", 100.0, 10, 0.001)
        span.end.assert_called_once()
        assert observer._workflow_span is None


class TestStepLifecycle:
    def test_on_step_start_creates_span(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        tracing.start_span.assert_called_once()

    def test_on_step_end_records_metrics(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_step_end("step1", "llm_call", "success", 200.0)
        metrics.record_step_execution.assert_called_once_with(
            "llm_call", "success", 0.2, stream=False
        )

    def test_on_step_end_ends_span(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        observer.on_step_end("step1", "llm_call", "success", 100.0)
        tracing.end_span.assert_called_once()

    def test_step_error_recorded(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end("step1", "llm_call", "failed", 50.0, error="boom")
        span.set_attribute.assert_any_call("step.error", "boom")

    def test_step_reasoning_tokens_set_on_span_when_nonzero(self) -> None:
        # Reasoning tokens land on the span under the OTel GenAI semantic
        # convention name only when reasoning was actually consumed, so
        # non-thinking workflows keep the span surface clean.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end("step1", "llm_call", "success", 100.0, reasoning_tokens=128)
        span.set_attribute.assert_any_call("gen_ai.usage.reasoning.output_tokens", 128)

    def test_step_reasoning_tokens_absent_when_zero(self) -> None:
        # The default path should not emit the reasoning-tokens attribute
        # so dashboards filtering on it see zero events for non-reasoning
        # models.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end("step1", "llm_call", "success", 100.0)
        attribute_keys = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "gen_ai.usage.reasoning.output_tokens" not in attribute_keys


class TestRunIdPropagation:
    def test_workflow_end_sets_run_id_attribute(self) -> None:
        # ``workflow.run_id`` rides on the workflow span end so consumers can
        # correlate a Jaeger trace with a workflow execution by run_id alone.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("wf", run_id="rid-42")
        span = tracing.start_span.return_value
        observer.on_workflow_end("wf", "success", 100.0, 10, 0.001)
        span.set_attribute.assert_any_call("workflow.run_id", "rid-42")

    def test_step_start_sets_run_id_attribute(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("wf", run_id="rid-7")
        observer.on_step_start("s1", "llm_call")
        attrs = tracing.start_span.call_args.kwargs["attributes"]
        assert attrs["workflow.run_id"] == "rid-7"

    def test_provider_call_start_sets_run_id_attribute(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_workflow_start("wf", run_id="rid-99")
        observer.on_provider_call_start("s1", "openai", "gpt-4o-mini")
        attrs = tracing.start_span.call_args.kwargs["attributes"]
        assert attrs["workflow.run_id"] == "rid-99"


class TestStepEndExtras:
    def test_attachment_count_recorded_on_metrics_and_span(self) -> None:
        tracing = MagicMock()
        metrics = MagicMock()
        observer = WorkflowObserver(tracing=tracing, metrics=metrics)
        observer.on_step_start("s1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end("s1", "llm_call", "success", 100.0, attachment_count=3)
        metrics.record_attachments.assert_called_once_with("llm_call", 3)
        span.set_attribute.assert_any_call("step.attachments", 3)

    def test_time_to_first_token_set_in_seconds(self) -> None:
        # The observer accepts ms but emits the canonical OTel
        # ``gen_ai.response.time_to_first_chunk`` in seconds.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("s1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end("s1", "llm_call", "success", 100.0, time_to_first_token_ms=250.0)
        span.set_attribute.assert_any_call("gen_ai.response.time_to_first_chunk", 0.25)

    def test_step_end_without_tracing_uses_span_end(self) -> None:
        # When no Tracing wrapper is attached, the observer falls back to
        # ``span.end()`` directly so spans still close cleanly.
        observer = WorkflowObserver()
        span = MagicMock()
        observer._step_spans["s1"] = span
        observer.on_step_end("s1", "llm_call", "success", 10.0)
        span.end.assert_called_once()


class TestAttachStepEvent:
    def test_attach_event_when_span_exists(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("s1", "llm_call")
        span = tracing.start_span.return_value
        observer.attach_step_event("s1", "agentloom.prompt.captured", {"prompt": "hi"})
        span.add_event.assert_called_once_with("agentloom.prompt.captured", {"prompt": "hi"})

    def test_attach_event_when_span_missing_is_noop(self) -> None:
        # Late or out-of-order events on a step that already ended must not
        # raise — they're silently dropped.
        observer = WorkflowObserver()
        observer.attach_step_event("missing-step", "evt", {"x": 1})


class TestProviderSpans:
    def test_start_without_tracing_is_noop(self) -> None:
        # No tracing wrapper → observer skips span creation but doesn't raise.
        observer = WorkflowObserver()
        observer.on_provider_call_start("s1", "openai", "gpt-4o-mini")
        observer.on_provider_call_end("s1", "openai", "gpt-4o-mini", 0.5)

    def test_end_records_metrics_and_canonical_attrs(self) -> None:
        tracing = MagicMock()
        metrics = MagicMock()
        observer = WorkflowObserver(tracing=tracing, metrics=metrics)
        observer.on_provider_call_start(
            "s1", "openai", "gpt-4o-mini", temperature=0.7, max_tokens=128, stream=True
        )
        span = tracing.start_span.return_value
        observer.on_provider_call_end(
            "s1",
            "openai",
            "gpt-4o-mini",
            1.5,
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=5,
            finish_reason="stop",
            stream=True,
        )
        metrics.record_provider_call.assert_called_once_with(
            "openai", "gpt-4o-mini", 1.5, stream=True
        )
        span.set_attribute.assert_any_call("gen_ai.response.model", "gpt-4o-mini")
        span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 10)
        span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 20)
        span.set_attribute.assert_any_call("gen_ai.usage.reasoning.output_tokens", 5)
        span.set_attribute.assert_any_call("gen_ai.response.finish_reasons", ["stop"])
        span.set_attribute.assert_any_call("agentloom.provider.attempt_outcome", "ok")

    def test_end_with_error_emits_error_type(self) -> None:
        # ``error.type`` (general OTel convention) lands alongside
        # ``step.error`` so OTel-aware consumers light up on the standard key.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_provider_call_start("s1", "openai", "gpt-4o-mini")
        span = tracing.start_span.return_value
        observer.on_provider_call_end("s1", "openai", "gpt-4o-mini", 0.4, error="RateLimitError")
        span.set_attribute.assert_any_call("step.error", "RateLimitError")
        span.set_attribute.assert_any_call("error.type", "RateLimitError")
        span.set_attribute.assert_any_call("agentloom.provider.attempt_outcome", "error")

    def test_end_without_matching_start_is_noop(self) -> None:
        # The gateway may call ``on_provider_call_end`` for an attempt that
        # never opened a span (e.g. circuit-open before ``on_provider_call_start``
        # in some paths) — observer must not raise.
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_provider_call_end("s1", "openai", "gpt-4o-mini", 0.1)
        metrics.record_provider_call.assert_called_once()

    def test_end_without_tracing_uses_span_end(self) -> None:
        observer = WorkflowObserver()
        span = MagicMock()
        observer._provider_spans[("s1", 0)] = span
        observer.on_provider_call_end("s1", "openai", "gpt-4o-mini", 0.2)
        span.end.assert_called_once()


class TestProviderErrorCleanup:
    def test_dangling_provider_span_closed_on_error(self) -> None:
        # If gateway raises before ``on_provider_call_end`` fires, an
        # ``on_provider_error`` with the step_id must close any open provider
        # spans for that step so traces don't leak.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_provider_call_start("s1", "openai", "gpt-4o-mini", attempt=0)
        span = tracing.start_span.return_value
        observer.on_provider_error("openai", "ConnectionError", step_id="s1")
        span.set_attribute.assert_any_call("error.type", "ConnectionError")
        tracing.end_span.assert_called_once_with(span)

    def test_dangling_span_closed_without_tracing(self) -> None:
        observer = WorkflowObserver()
        span = MagicMock()
        observer._provider_spans[("s1", 0)] = span
        observer.on_provider_error("openai", "Boom", step_id="s1")
        span.end.assert_called_once()


class TestStreamResponse:
    def test_records_ttft_and_stream_count(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_stream_response("openai", "gpt-4o-mini", 0.42)
        metrics.record_stream_response.assert_called_once_with("openai", "gpt-4o-mini")
        metrics.record_time_to_first_token.assert_called_once_with("openai", "gpt-4o-mini", 0.42)

    def test_no_metrics_no_error(self) -> None:
        observer = WorkflowObserver()
        observer.on_stream_response("openai", "gpt-4o-mini", 0.1)


class TestProviderEvents:
    def test_on_provider_error(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_provider_error("openai", "timeout")
        metrics.record_provider_error.assert_called_once_with("openai", "timeout")

    def test_on_tokens(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_tokens("openai", "gpt-4o-mini", 100, 200)
        metrics.record_tokens.assert_called_once_with(
            "openai", "gpt-4o-mini", 100, 200, reasoning_tokens=0
        )

    def test_on_tokens_reasoning(self) -> None:
        # Reasoning tokens flow through `on_tokens` as a kwarg-only
        # parameter so non-thinking call sites stay unchanged.
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_tokens("anthropic", "claude-opus-4", 100, 50, reasoning_tokens=200)
        metrics.record_tokens.assert_called_once_with(
            "anthropic", "claude-opus-4", 100, 50, reasoning_tokens=200
        )


class TestCircuitBreakerEvents:
    def test_on_circuit_state_change(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_circuit_state_change("openai", "closed", "open")
        metrics.set_circuit_state.assert_called_once_with("openai", 1)

    def test_state_map_values(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_circuit_state_change("p", "open", "half_open")
        metrics.set_circuit_state.assert_called_with("p", 2)


class TestHITLEvents:
    def test_on_approval_gate_records_metrics(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_approval_gate("gate", "my-wf", "approved")
        metrics.record_approval_gate.assert_called_once_with("my-wf", "approved")

    def test_on_approval_gate_sets_span_attribute(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("gate", "approval_gate")
        span = tracing.start_span.return_value
        observer.on_approval_gate("gate", "my-wf", "rejected")
        span.set_attribute.assert_any_call("agentloom.approval_gate.decision", "rejected")

    def test_on_webhook_delivery_records_metrics(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_webhook_delivery("gate", "my-wf", "success", 1.5)
        metrics.record_webhook_delivery.assert_called_once_with("my-wf", "success", 1.5)

    def test_on_webhook_delivery_sets_span_attributes(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("gate", "approval_gate")
        span = tracing.start_span.return_value
        observer.on_webhook_delivery("gate", "my-wf", "failed", 6.0)
        span.set_attribute.assert_any_call("agentloom.webhook.status", "failed")
        span.set_attribute.assert_any_call("agentloom.webhook.latency_s", 6.0)

    def test_no_metrics_no_error(self) -> None:
        observer = WorkflowObserver()
        observer.on_approval_gate("gate", "wf", "pending")
        observer.on_webhook_delivery("gate", "wf", "success", 0.5)


class TestMockAndRecordingEvents:
    def test_on_mock_replay_records_metrics(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_mock_replay("my-wf", "step_a", "step_id")
        metrics.record_mock_replay.assert_called_once_with("my-wf", "step_id")

    def test_on_mock_replay_sets_span_attribute(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step_a", "llm_call")
        span = tracing.start_span.return_value
        observer.on_mock_replay("my-wf", "step_a", "prompt_hash")
        span.set_attribute.assert_any_call("agentloom.mock.matched_by", "prompt_hash")

    def test_on_recording_capture_records_metrics(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_recording_capture("step_a", "anthropic", "claude", 0.88)
        metrics.record_recording_capture.assert_called_once_with("anthropic", "claude", 0.88)

    def test_on_recording_capture_sets_span_attributes(self) -> None:
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step_a", "llm_call")
        span = tracing.start_span.return_value
        observer.on_recording_capture("step_a", "openai", "gpt-4o-mini", 1.2)
        span.set_attribute.assert_any_call("agentloom.recording.provider", "openai")
        span.set_attribute.assert_any_call("agentloom.recording.latency_s", 1.2)

    def test_no_metrics_no_error(self) -> None:
        observer = WorkflowObserver()
        observer.on_mock_replay("wf", "s", "default")
        observer.on_recording_capture("s", "p", "m", 0.1)


class TestBudgetEvents:
    def test_on_budget_remaining(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_budget_remaining("wf", 0.75)
        metrics.set_budget_remaining.assert_called_once_with("wf", 0.75)

    def test_on_budget_remaining_no_metrics(self) -> None:
        observer = WorkflowObserver()
        observer.on_budget_remaining("wf", 0.5)  # no error


class TestShutdown:
    def test_shutdown_calls_both(self) -> None:
        tracing = MagicMock()
        metrics = MagicMock()
        observer = WorkflowObserver(tracing=tracing, metrics=metrics)
        observer.shutdown()
        tracing.shutdown.assert_called_once()
        metrics.shutdown.assert_called_once()

    def test_shutdown_no_tracing_no_error(self) -> None:
        observer = WorkflowObserver()
        observer.shutdown()
