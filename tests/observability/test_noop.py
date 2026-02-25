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
