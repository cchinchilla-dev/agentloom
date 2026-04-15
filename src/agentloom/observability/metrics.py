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
        self._workflow_histogram: Any = None
        self._cost_counter: Any = None
        self._attachment_counter: Any = None
        self._stream_counter: Any = None
        self._ttft_histogram: Any = None
        self._mock_replay_counter: Any = None
        self._recording_capture_counter: Any = None
        self._recording_latency_histogram: Any = None
        self._circuit_states: dict[str, int] = {}  # provider -> state int
        self._budget_remaining: dict[str, float] = {}  # workflow -> remaining USD

        # prometheus_client instruments (fallback)
        self._prom_counters: dict[str, Any] = {}
        self._prom_histograms: dict[str, Any] = {}
        self._prom_gauges: dict[str, Any] = {}

        if not self._enabled:
            return

        if _HAS_OTEL_METRICS:
            self._setup_otel(endpoint)
        elif _HAS_PROM:  # pragma: no cover — prom fallback, only active when OTel unavailable
            self._setup_prom()

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
        self._workflow_histogram = meter.create_histogram(
            "agentloom_workflow_duration_seconds",
            description="Workflow execution duration",
            unit="s",
        )
        self._cost_counter = meter.create_counter(
            "agentloom_cost_usd_total",
            description="Cumulative cost in USD",
        )
        self._attachment_counter = meter.create_counter(
            "agentloom_attachments_total",
            description="Total multi-modal attachments processed",
        )
        self._stream_counter = meter.create_counter(
            "agentloom_stream_responses_total",
            description="Total streamed LLM responses",
        )
        self._ttft_histogram = meter.create_histogram(
            "agentloom_time_to_first_token_seconds",
            description="Time to first token for streamed responses",
            unit="s",
        )
        self._approval_gate_counter = meter.create_counter(
            "agentloom_approval_gates_total",
            description="Total approval gate decisions",
        )
        self._webhook_counter = meter.create_counter(
            "agentloom_webhook_deliveries_total",
            description="Total webhook delivery attempts",
        )
        self._webhook_histogram = meter.create_histogram(
            "agentloom_webhook_latency_seconds",
            description="Webhook delivery latency",
            unit="s",
        )
        self._mock_replay_counter = meter.create_counter(
            "agentloom_mock_replays_total",
            description="Total MockProvider replay lookups",
        )
        self._recording_capture_counter = meter.create_counter(
            "agentloom_recording_captures_total",
            description="Total RecordingProvider captures",
        )
        self._recording_latency_histogram = meter.create_histogram(
            "agentloom_recording_latency_seconds",
            description="Latency of real provider calls captured by RecordingProvider",
            unit="s",
        )

        # Circuit breaker gauge (callback-based, reads from _circuit_states)
        states = self._circuit_states

        def _cb_circuit(options: Any) -> Any:
            Observation = otel_api_metrics.Observation
            for prov, val in list(states.items()):  # pragma: no cover — fires on OTel export
                yield Observation(val, {"provider": prov})

        meter.create_observable_gauge(
            "agentloom_circuit_breaker_state",
            callbacks=[_cb_circuit],
            description="Circuit breaker state (0=closed, 1=open, 2=half_open)",
        )

        # Budget remaining gauge (callback-based, reads from _budget_remaining)
        budget = self._budget_remaining

        def _cb_budget(options: Any) -> Any:  # pragma: no cover
            Observation = otel_api_metrics.Observation
            for wf, val in list(budget.items()):
                yield Observation(val, {"workflow": wf})

        meter.create_observable_gauge(
            "agentloom_budget_remaining_usd",
            callbacks=[_cb_budget],
            description="Remaining budget in USD",
        )

        self._backend = "otel"
        logger.debug("Metrics: OTLP push → %s", endpoint)

    def _setup_prom(
        self,
    ) -> None:  # pragma: no cover — prom fallback, only active when OTel unavailable
        self._prom_counters["workflow_runs"] = prom.Counter(
            "agentloom_workflow_runs_total",
            "Total workflow executions",
            ["workflow", "status"],
        )
        self._prom_counters["step_executions"] = prom.Counter(
            "agentloom_step_executions_total",
            "Total step executions",
            ["step_type", "status", "stream"],
        )
        self._prom_counters["provider_calls"] = prom.Counter(
            "agentloom_provider_calls_total",
            "Total provider API calls",
            ["provider", "model", "stream"],
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
            ["step_type", "stream"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )
        self._prom_histograms["provider_latency"] = prom.Histogram(
            "agentloom_provider_latency_seconds",
            "Provider API call latency",
            ["provider", "stream"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
        )
        self._prom_counters["attachments"] = prom.Counter(
            "agentloom_attachments_total",
            "Total multi-modal attachments processed",
            ["step_type"],
        )
        self._prom_counters["stream_responses"] = prom.Counter(
            "agentloom_stream_responses_total",
            "Total streamed LLM responses",
            ["provider", "model"],
        )
        self._prom_histograms["ttft"] = prom.Histogram(
            "agentloom_time_to_first_token_seconds",
            "Time to first token for streamed responses",
            ["provider", "model"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
        )
        self._prom_counters["approval_gates"] = prom.Counter(  # pragma: no cover
            "agentloom_approval_gates_total",
            "Total approval gate decisions",
            ["decision", "workflow"],
        )
        self._prom_counters["webhook_deliveries"] = prom.Counter(  # pragma: no cover
            "agentloom_webhook_deliveries_total",
            "Total webhook delivery attempts",
            ["status", "workflow"],
        )
        self._prom_counters["mock_replays"] = prom.Counter(  # pragma: no cover — prom fallback
            "agentloom_mock_replays_total",
            "Total MockProvider replay lookups",
            ["workflow", "matched_by"],
        )
        self._prom_counters["recording_captures"] = (
            prom.Counter(  # pragma: no cover — prom fallback
                "agentloom_recording_captures_total",
                "Total RecordingProvider captures",
                ["provider", "model"],
            )
        )
        self._prom_histograms["recording_latency"] = (
            prom.Histogram(  # pragma: no cover — prom fallback
                "agentloom_recording_latency_seconds",
                "Latency of real provider calls captured by RecordingProvider",
                ["provider", "model"],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
            )
        )
        self._prom_histograms["webhook_latency"] = prom.Histogram(  # pragma: no cover
            "agentloom_webhook_latency_seconds",
            "Webhook delivery latency",
            ["status"],
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

    def record_workflow_run(
        self, workflow: str, status: str, duration_s: float = 0.0, cost_usd: float = 0.0
    ) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._workflow_counter.add(1, {"workflow": workflow, "status": status})
            if duration_s > 0:
                attrs = {"workflow": workflow, "status": status}
                self._workflow_histogram.record(duration_s, attrs)
            if cost_usd > 0:
                self._cost_counter.add(cost_usd, {"workflow": workflow, "provider": "total"})
        else:  # pragma: no cover — prom fallback
            self._prom_counters["workflow_runs"].labels(workflow=workflow, status=status).inc()

    def record_step_execution(
        self, step_type: str, status: str, duration_s: float, stream: bool = False
    ) -> None:
        if not self._enabled:
            return
        stream_str = str(stream).lower()
        if self._backend == "otel":
            self._step_counter.add(
                1, {"step_type": step_type, "status": status, "stream": stream_str}
            )
            self._step_histogram.record(duration_s, {"step_type": step_type, "stream": stream_str})
        else:  # pragma: no cover — prom fallback
            self._prom_counters["step_executions"].labels(
                step_type=step_type, status=status, stream=stream_str
            ).inc()
            self._prom_histograms["step_duration"].labels(
                step_type=step_type, stream=stream_str
            ).observe(duration_s)

    def record_provider_call(
        self, provider: str, model: str, latency_s: float, stream: bool = False
    ) -> None:
        if not self._enabled:
            return
        stream_str = str(stream).lower()
        if self._backend == "otel":
            self._provider_counter.add(
                1, {"provider": provider, "model": model, "stream": stream_str}
            )
            self._provider_histogram.record(latency_s, {"provider": provider, "stream": stream_str})
        else:  # pragma: no cover — prom fallback
            self._prom_counters["provider_calls"].labels(
                provider=provider, model=model, stream=stream_str
            ).inc()
            self._prom_histograms["provider_latency"].labels(
                provider=provider, stream=stream_str
            ).observe(latency_s)

    def record_provider_error(self, provider: str, error_type: str) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._provider_error_counter.add(1, {"provider": provider, "error_type": error_type})
        else:  # pragma: no cover — prom fallback
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
        else:  # pragma: no cover — prom fallback
            self._prom_counters["tokens_total"].labels(
                provider=provider, model=model, direction="input"
            ).inc(prompt_tokens)
            self._prom_counters["tokens_total"].labels(
                provider=provider, model=model, direction="output"
            ).inc(completion_tokens)

    def record_attachments(self, step_type: str, count: int) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._attachment_counter.add(count, {"step_type": step_type})
        else:  # pragma: no cover — prom fallback
            self._prom_counters["attachments"].labels(step_type=step_type).inc(count)

    def record_stream_response(self, provider: str, model: str) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._stream_counter.add(1, {"provider": provider, "model": model})
        else:  # pragma: no cover — prom fallback
            self._prom_counters["stream_responses"].labels(provider=provider, model=model).inc()

    def record_time_to_first_token(self, provider: str, model: str, ttft_s: float) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._ttft_histogram.record(ttft_s, {"provider": provider, "model": model})
        else:  # pragma: no cover — prom fallback
            self._prom_histograms["ttft"].labels(provider=provider, model=model).observe(ttft_s)

    def record_approval_gate(self, workflow: str, decision: str) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._approval_gate_counter.add(1, {"decision": decision, "workflow": workflow})
        else:  # pragma: no cover — prom fallback, only active when OTel unavailable
            self._prom_counters["approval_gates"].labels(decision=decision, workflow=workflow).inc()

    def record_webhook_delivery(self, workflow: str, status: str, latency_s: float) -> None:
        if not self._enabled:
            return
        if self._backend == "otel":
            self._webhook_counter.add(1, {"status": status, "workflow": workflow})
            self._webhook_histogram.record(latency_s, {"status": status})
        else:  # pragma: no cover — prom fallback, only active when OTel unavailable
            self._prom_counters["webhook_deliveries"].labels(status=status, workflow=workflow).inc()
            self._prom_histograms["webhook_latency"].labels(status=status).observe(latency_s)

    def record_mock_replay(self, workflow: str, matched_by: str) -> None:
        """Record a MockProvider lookup. ``matched_by`` ∈ {step_id, prompt_hash, default}."""
        if not self._enabled:
            return
        if self._backend == "otel":
            self._mock_replay_counter.add(1, {"workflow": workflow, "matched_by": matched_by})
        else:  # pragma: no cover — prom fallback
            self._prom_counters["mock_replays"].labels(
                workflow=workflow, matched_by=matched_by
            ).inc()

    def record_recording_capture(self, provider: str, model: str, latency_s: float) -> None:
        """Record a RecordingProvider capture of a real provider call."""
        if not self._enabled:
            return
        if self._backend == "otel":
            self._recording_capture_counter.add(1, {"provider": provider, "model": model})
            self._recording_latency_histogram.record(
                latency_s, {"provider": provider, "model": model}
            )
        else:  # pragma: no cover — prom fallback
            self._prom_counters["recording_captures"].labels(provider=provider, model=model).inc()
            self._prom_histograms["recording_latency"].labels(
                provider=provider, model=model
            ).observe(latency_s)

    def set_budget_remaining(self, workflow: str, remaining: float) -> None:
        if not self._enabled:
            return
        self._budget_remaining[workflow] = remaining
        if self._backend == "prom":  # pragma: no cover — prom fallback
            self._prom_gauges["budget_remaining"].labels(workflow=workflow).set(remaining)

    def set_circuit_state(self, provider: str, state: int) -> None:
        if not self._enabled:
            return
        # OTel: update dict read by the observable gauge callback
        self._circuit_states[provider] = state
        if self._backend == "prom":  # pragma: no cover — prom fallback
            self._prom_gauges["circuit_state"].labels(provider=provider).set(state)

    def shutdown(self) -> None:
        """Flush pending metrics before process exit."""
        if self._backend == "otel" and self._meter_provider:
            self._meter_provider.shutdown()
