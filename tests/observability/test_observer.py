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
        metrics.record_step_execution.assert_called_once_with("llm_call", "success", 0.2)

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


class TestProviderEvents:
    def test_on_provider_call(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_provider_call("openai", "gpt-4o-mini", 0.5)
        metrics.record_provider_call.assert_called_once_with("openai", "gpt-4o-mini", 0.5)

    def test_on_provider_error(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_provider_error("openai", "timeout")
        metrics.record_provider_error.assert_called_once_with("openai", "timeout")

    def test_on_tokens(self) -> None:
        metrics = MagicMock()
        observer = WorkflowObserver(metrics=metrics)
        observer.on_tokens("openai", "gpt-4o-mini", 100, 200)
        metrics.record_tokens.assert_called_once_with("openai", "gpt-4o-mini", 100, 200)


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
