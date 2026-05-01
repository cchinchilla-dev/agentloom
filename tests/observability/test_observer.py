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
        # ``step.reasoning_tokens`` lands on the span only when reasoning
        # was actually consumed, so non-thinking workflows keep the span
        # surface clean.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end(
            "step1", "llm_call", "success", 100.0, tokens=300, reasoning_tokens=128
        )
        span.set_attribute.assert_any_call("step.reasoning_tokens", 128)

    def test_step_reasoning_tokens_absent_when_zero(self) -> None:
        # The default path should not emit ``step.reasoning_tokens`` so
        # dashboards filtering on the attribute see zero events for
        # non-reasoning models.
        tracing = MagicMock()
        observer = WorkflowObserver(tracing=tracing)
        observer.on_step_start("step1", "llm_call")
        span = tracing.start_span.return_value
        observer.on_step_end("step1", "llm_call", "success", 100.0, tokens=300)
        attribute_keys = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "step.reasoning_tokens" not in attribute_keys


class TestProviderEvents:
    def test_on_provider_call(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_provider_call("openai", "gpt-4o-mini", 0.5)
        metrics.record_provider_call.assert_called_once_with(
            "openai", "gpt-4o-mini", 0.5, stream=False
        )

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
        span.set_attribute.assert_any_call("approval_gate.decision", "rejected")

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
        span.set_attribute.assert_any_call("webhook.status", "failed")
        span.set_attribute.assert_any_call("webhook.latency_s", 6.0)

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
        span.set_attribute.assert_any_call("mock.matched_by", "prompt_hash")

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
        span.set_attribute.assert_any_call("recording.provider", "openai")
        span.set_attribute.assert_any_call("recording.latency_s", 1.2)

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
