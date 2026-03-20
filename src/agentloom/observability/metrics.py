"""Metrics wrapper — pushes via OTLP, falls back to prometheus_client.

Short-lived CLI processes can't reliably serve a /metrics endpoint, so we push
metrics through the same OTLP pipeline used for traces.  The OTel collector
then exports them to Prometheus via its prometheus exporter (port 8889).
"""

from __future__ import annotations

import logging
from typing import Any

from agentloom.compat import is_available, try_import

logger = logging.getLogger("agentloom.observability.metrics")

# Prefer OTel metrics (push-based, works for short-lived CLIs)
otel_metrics = try_import("opentelemetry.sdk.metrics", extra="observability")
otel_metric_exporter = try_import(
    "opentelemetry.exporter.otlp.proto.grpc.metric_exporter", extra="observability"
)
otel_api_metrics = try_import("opentelemetry.metrics", extra="observability")

_HAS_OTEL_METRICS = (
    is_available(otel_metrics)
    and is_available(otel_metric_exporter)
    and is_available(otel_api_metrics)
)

# Fallback: prometheus_client (pull-based)
prom = try_import("prometheus_client", extra="observability")
_HAS_PROM = is_available(prom)


class MetricsManager:
    """Records workflow/step/provider metrics, pushing them via OTLP.

    Falls back to prometheus_client if OTel metrics SDK is unavailable.
    All methods are silent no-ops if no metrics backend is installed.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and (_HAS_OTEL_METRICS or _HAS_PROM)
        self._backend: str = "none"
        self._meter_provider: Any = None

        # OTel instruments
        self._workflow_counter: Any = None
        self._step_counter: Any = None
        self._step_histogram: Any = None
        self._provider_counter: Any = None
        self._provider_error_counter: Any = None
        self._provider_histogram: Any = None
        self._token_counter: Any = None

        # prometheus_client instruments (fallback)
        self._prom_counters: dict[str, Any] = {}
        self._prom_histograms: dict[str, Any] = {}
        self._prom_gauges: dict[str, Any] = {}

        if not self._enabled:
            return

        if _HAS_OTEL_METRICS:
            self._setup_otel(endpoint)
        elif _HAS_PROM:
            self._setup_prom()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_otel(self, endpoint: str) -> None:
        exporter = otel_metric_exporter.OTLPMetricExporter(endpoint=endpoint, insecure=True)
        reader = otel_metrics.export.PeriodicExportingMetricReader(
            exporter, export_interval_millis=5000
        )
        self._meter_provider = otel_metrics.MeterProvider(metric_readers=[reader])
        otel_api_metrics.set_meter_provider(self._meter_provider)
        meter = otel_api_metrics.get_meter("agentloom")

        self._workflow_counter = meter.create_counter(
            "agentloom_workflow_runs_total",
            description="Total workflow executions",
        )
        self._step_counter = meter.create_counter(
            "agentloom_step_executions_total",
            description="Total step executions",
        )
        self._step_histogram = meter.create_histogram(
            "agentloom_step_duration_seconds",
            description="Step execution duration",
            unit="s",
        )
        self._provider_counter = meter.create_counter(
            "agentloom_provider_calls_total",
            description="Total provider API calls",
        )
        self._provider_error_counter = meter.create_counter(
            "agentloom_provider_errors_total",
            description="Total provider errors",
        )
        self._provider_histogram = meter.create_histogram(
            "agentloom_provider_latency_seconds",
            description="Provider API call latency",
            unit="s",
        )
        self._token_counter = meter.create_counter(
            "agentloom_tokens_total",
            description="Total tokens consumed",
        )
        self._backend = "otel"
        logger.debug("Metrics: OTLP push → %s", endpoint)

    def _setup_prom(self) -> None:
        self._prom_counters["workflow_runs"] = prom.Counter(
            "agentloom_workflow_runs_total",
            "Total workflow executions",
            ["workflow", "status"],
        )
        self._prom_counters["step_executions"] = prom.Counter(
            "agentloom_step_executions_total",
            "Total step executions",
            ["step_type", "status"],
        )
        self._prom_counters["provider_calls"] = prom.Counter(
            "agentloom_provider_calls_total",
            "Total provider API calls",
            ["provider", "model"],
        )
        self._prom_counters["provider_errors"] = prom.Counter(
            "agentloom_provider_errors_total",
            "Total provider errors",
            ["provider", "error_type"],
        )
        self._prom_counters["tokens_total"] = prom.Counter(
            "agentloom_tokens_total",
            "Total tokens consumed",
            ["provider", "model", "direction"],
        )
        self._prom_histograms["step_duration"] = prom.Histogram(
            "agentloom_step_duration_seconds",
            "Step execution duration",
            ["step_type"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )
        self._prom_histograms["provider_latency"] = prom.Histogram(
            "agentloom_provider_latency_seconds",
            "Provider API call latency",
            ["provider"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
        )
        self._prom_gauges["budget_remaining"] = prom.Gauge(
            "agentloom_budget_remaining_usd",
            "Remaining budget in USD",
            ["workflow"],
        )
        self._prom_gauges["circuit_state"] = prom.Gauge(
            "agentloom_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["provider"],
        )
        self._backend = "prom"
        logger.debug("Metrics: prometheus_client (pull)")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_workflow_run(self, workflow: str, status: str) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._workflow_counter.add(1, {"workflow": workflow, "status": status})
        else:
            self._prom_counters["workflow_runs"].labels(workflow=workflow, status=status).inc()

    def record_step_execution(self, step_type: str, status: str, duration_s: float) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._step_counter.add(1, {"step_type": step_type, "status": status})
            self._step_histogram.record(duration_s, {"step_type": step_type})
        else:
            self._prom_counters["step_executions"].labels(step_type=step_type, status=status).inc()
            self._prom_histograms["step_duration"].labels(step_type=step_type).observe(duration_s)

    def record_provider_call(self, provider: str, model: str, latency_s: float) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._provider_counter.add(1, {"provider": provider, "model": model})
            self._provider_histogram.record(latency_s, {"provider": provider})
        else:
            self._prom_counters["provider_calls"].labels(provider=provider, model=model).inc()
            self._prom_histograms["provider_latency"].labels(provider=provider).observe(latency_s)

    def record_provider_error(self, provider: str, error_type: str) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._provider_error_counter.add(1, {"provider": provider, "error_type": error_type})
        else:
            self._prom_counters["provider_errors"].labels(
                provider=provider, error_type=error_type
            ).inc()

    def record_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._token_counter.add(
                prompt_tokens,
                {"provider": provider, "model": model, "direction": "input"},
            )
            self._token_counter.add(
                completion_tokens,
                {"provider": provider, "model": model, "direction": "output"},
            )
        else:
            self._prom_counters["tokens_total"].labels(
                provider=provider, model=model, direction="input"
            ).inc(prompt_tokens)
            self._prom_counters["tokens_total"].labels(
                provider=provider, model=model, direction="output"
            ).inc(completion_tokens)

    def set_budget_remaining(self, workflow: str, remaining: float) -> None:
        if self._enabled and self._backend == "prom":
            self._prom_gauges["budget_remaining"].labels(workflow=workflow).set(remaining)

    def set_circuit_state(self, provider: str, state: int) -> None:
        if self._enabled and self._backend == "prom":
            self._prom_gauges["circuit_state"].labels(provider=provider).set(state)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush pending metrics before process exit."""
        if self._backend == "otel" and self._meter_provider:
            self._meter_provider.shutdown()
