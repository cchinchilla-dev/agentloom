"""Tests for the noop observability implementations."""

from __future__ import annotations

from agentloom.observability.noop import (
    NoopCounter,
    NoopGauge,
    NoopHistogram,
    NoopMeter,
    NoopSpan,
    NoopTracer,
)


class TestNoopSpan:
    """Test that NoopSpan methods don't raise errors."""

    def test_set_attribute(self) -> None:
        span = NoopSpan()
        span.set_attribute("key", "value")

    def test_set_status(self) -> None:
        span = NoopSpan()
        span.set_status("OK")
        span.set_status("ERROR", description="something failed")

    def test_record_exception(self) -> None:
        span = NoopSpan()
        span.record_exception(RuntimeError("test error"))

    def test_end(self) -> None:
        span = NoopSpan()
        span.end()

    def test_context_manager(self) -> None:
        span = NoopSpan()
        with span as s:
            assert s is span
            s.set_attribute("inside", True)


class TestNoopTracer:
    """Test that NoopTracer produces working NoopSpans."""

    def test_start_span(self) -> None:
        tracer = NoopTracer()
        span = tracer.start_span("test-span")
        assert isinstance(span, NoopSpan)

    def test_start_as_current_span(self) -> None:
        tracer = NoopTracer()
        span = tracer.start_as_current_span("test-span")
        assert isinstance(span, NoopSpan)

    def test_span_is_usable(self) -> None:
        tracer = NoopTracer()
        span = tracer.start_span("my-operation")
        span.set_attribute("key", "val")
        span.set_status("OK")
        span.end()

    def test_span_as_context_manager(self) -> None:
        tracer = NoopTracer()
        with tracer.start_span("scoped-span") as span:
            span.set_attribute("step", "test")


class TestNoopMeter:
    """Test that NoopMeter produces working no-op instruments."""

    def test_create_counter(self) -> None:
        meter = NoopMeter()
        counter = meter.create_counter("requests_total")
        assert isinstance(counter, NoopCounter)

    def test_counter_add(self) -> None:
        meter = NoopMeter()
        counter = meter.create_counter("requests_total")
        counter.add(1)
        counter.add(5, attributes={"method": "GET"})

    def test_create_histogram(self) -> None:
        meter = NoopMeter()
        histogram = meter.create_histogram("request_duration")
        assert isinstance(histogram, NoopHistogram)

    def test_histogram_record(self) -> None:
        meter = NoopMeter()
        histogram = meter.create_histogram("request_duration")
        histogram.record(0.5)
        histogram.record(1.2, attributes={"endpoint": "/api"})

    def test_create_up_down_counter(self) -> None:
        meter = NoopMeter()
        counter = meter.create_up_down_counter("active_connections")
        assert isinstance(counter, NoopCounter)
        counter.add(1)
        counter.add(-1)


class TestNoopGauge:
    """Test that NoopGauge doesn't raise errors."""

    def test_set(self) -> None:
        gauge = NoopGauge()
        gauge.set(42.0)
        gauge.set(0.0, attributes={"unit": "celsius"})


class TestNoopIntegration:
    """Test that noop implementations work together without errors."""

    def test_full_workflow_tracing(self) -> None:
        tracer = NoopTracer()
        meter = NoopMeter()

        counter = meter.create_counter("workflow_runs")
        histogram = meter.create_histogram("workflow_duration_ms")

        with tracer.start_span("workflow-execution") as span:
            span.set_attribute("workflow.name", "test-workflow")
            counter.add(1, attributes={"workflow": "test"})

            with tracer.start_span("step-execution") as step_span:
                step_span.set_attribute("step.id", "step-1")
                step_span.set_status("OK")

            histogram.record(150.0, attributes={"status": "success"})
            span.set_status("OK")


class TestNoopObserver:
    """NoopObserver must implement every hook of WorkflowObserver so a
    caller can pass it in as a safe default with no AttributeError risk."""

    def test_has_all_hooks_of_workflow_observer(self) -> None:
        from agentloom.observability.noop import NoopObserver
        from agentloom.observability.observer import WorkflowObserver

        required = {
            name
            for name in dir(WorkflowObserver)
            if not name.startswith("_") and callable(getattr(WorkflowObserver, name))
        }
        provided = {
            name
            for name in dir(NoopObserver)
            if not name.startswith("_") and callable(getattr(NoopObserver, name))
        }
        missing = required - provided
        assert missing == set(), f"NoopObserver missing hooks: {missing}"

    def test_every_hook_tolerates_extra_kwargs(self) -> None:
        from agentloom.observability.noop import NoopObserver

        obs = NoopObserver()
        obs.on_workflow_start("wf", new_future_arg=True)
        obs.on_step_start("s1", "llm_call", future_extra=1)
        obs.on_step_end("s1", "llm_call", "success", 1.0, future_extra=True, newer_arg="x")
        obs.on_webhook_delivery("s1", "wf", "success", 0.1, future_extra=True)

    def test_every_hook_executes_no_op_body(self) -> None:
        """Calling every NoopObserver hook executes the ``pass`` body so the
        full no-op surface stays exercised — guards against regressions where
        a new hook accidentally raises."""
        from agentloom.observability.noop import NoopObserver

        obs = NoopObserver()
        obs.on_workflow_start("wf")
        obs.on_workflow_end("wf", "success", 100.0, 50, 0.001)
        obs.on_step_start("s1", "llm_call")
        obs.on_step_end("s1", "llm_call", "success", 5.0, 10, 0.0001)
        obs.on_provider_call("openai", "gpt-4o-mini", 1.5)
        obs.on_provider_error("openai", "RateLimitError")
        obs.on_stream_response("openai", "gpt-4o-mini", 0.3)
        obs.on_tokens("openai", "gpt-4o-mini", 100, 50)
        obs.on_mock_replay("wf", "s1", "step_id")
        obs.on_recording_capture("s1", "openai", "gpt-4o-mini", 1.2)
        obs.on_budget_remaining("wf", 0.5)
        obs.on_circuit_state_change("openai", "closed", "open")
        obs.on_approval_gate("s1", "wf", "approved")
        obs.shutdown()
