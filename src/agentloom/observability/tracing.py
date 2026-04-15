"""OpenTelemetry tracing wrapper — gracefully degrades if not installed."""

from __future__ import annotations

from typing import Any

from agentloom.compat import is_available, try_import
from agentloom.observability.noop import NoopSpan, NoopTracer

# Conditional imports
otel_api = try_import("opentelemetry.trace", extra="observability")
otel_context = try_import("opentelemetry.context", extra="observability")
otel_sdk_trace = try_import("opentelemetry.sdk.trace", extra="observability")
otel_sdk_resources = try_import("opentelemetry.sdk.resources", extra="observability")
otel_exporter = try_import(
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", extra="observability"
)

_HAS_OTEL = is_available(otel_api) and is_available(otel_sdk_trace)


class TracingManager:
    """Manages OpenTelemetry tracing. Returns no-ops if OTel is not installed."""

    def __init__(
        self,
        service_name: str = "agentloom",
        endpoint: str = "http://localhost:4317",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and _HAS_OTEL
        self._tracer: Any = None
        self._context_tokens: dict[int, Any] = {}

        if self._enabled:
            self._setup(service_name, endpoint)

    def _setup(self, service_name: str, endpoint: str) -> None:
        """Initialize OTel tracer provider with OTLP exporter."""
        resource = otel_sdk_resources.Resource.create({"service.name": service_name})
        provider = otel_sdk_trace.TracerProvider(resource=resource)

        if is_available(otel_exporter):
            exporter = otel_exporter.OTLPSpanExporter(endpoint=endpoint)
            processor = otel_sdk_trace.export.BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        otel_api.set_tracer_provider(provider)
        self._tracer = otel_api.get_tracer("agentloom")

    def get_tracer(self) -> Any:
        """Get the OTel tracer or a NoopTracer."""
        if self._enabled and self._tracer:
            return self._tracer
        return NoopTracer()

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
        """Start a new span as child of the current active span.

        Sets the span as current so subsequent start_span calls create
        nested children (workflow > step hierarchy).
        """
        tracer = self.get_tracer()
        if self._enabled:
            span = tracer.start_span(name)
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            # Set as active context so child spans nest correctly
            ctx = otel_api.set_span_in_context(span)
            token = otel_context.attach(ctx)
            # Store detach token so end_span can restore parent context
            self._context_tokens[id(span)] = token
            return span
        return NoopSpan()

    def end_span(self, span: Any) -> None:
        """End a span and restore the parent context."""
        if self._enabled and span:
            token = self._context_tokens.pop(id(span), None)
            span.end()
            if token is not None:
                otel_context.detach(token)

    def shutdown(self) -> None:
        """Flush and shut down the tracer provider."""
        if self._enabled:
            provider = otel_api.get_tracer_provider()
            shutdown_fn = getattr(provider, "shutdown", None)
            if shutdown_fn is not None:
                shutdown_fn()
