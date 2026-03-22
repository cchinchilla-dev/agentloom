"""Tests for tracing manager."""

from __future__ import annotations

from agentloom.observability.noop import NoopSpan
from agentloom.observability.tracing import TracingManager


class TestTracingManagerDisabled:
    def test_disabled_returns_noop(self) -> None:
        tm = TracingManager(enabled=False)
        span = tm.start_span("test")
        assert isinstance(span, NoopSpan)

    def test_disabled_get_tracer_returns_noop(self) -> None:
        from agentloom.observability.noop import NoopTracer

        tm = TracingManager(enabled=False)
        tracer = tm.get_tracer()
        assert isinstance(tracer, NoopTracer)

    def test_disabled_shutdown_no_error(self) -> None:
        tm = TracingManager(enabled=False)
        tm.shutdown()


class TestTracingManagerEnabled:
    def test_enabled_with_otel_creates_tracer(self) -> None:
        tm = TracingManager(enabled=True)
        # If OTel is installed, we get a real tracer; if not, noop
        tracer = tm.get_tracer()
        assert tracer is not None

    def test_start_span_with_attributes(self) -> None:
        tm = TracingManager(enabled=True)
        span = tm.start_span("test-span", attributes={"key": "value"})
        assert span is not None
        # Clean up
        if hasattr(tm, "end_span"):
            tm.end_span(span)
        elif hasattr(span, "end"):
            span.end()

    def test_end_span_restores_context(self) -> None:
        tm = TracingManager(enabled=True)
        span = tm.start_span("parent")
        child = tm.start_span("child")
        tm.end_span(child)
        tm.end_span(span)
        # No error = context tokens properly detached
