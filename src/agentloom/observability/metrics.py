"""Metrics wrapper — pushes via OTLP, falls back to prometheus_client.

Short-lived CLI processes can't reliably serve a /metrics endpoint, so we push
metrics through the same OTLP pipeline used for traces.  The OTel collector
then exports them to Prometheus via its prometheus exporter (port 8889).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

from agentloom.compat import is_available, try_import
from agentloom.observability.schema import to_genai_provider_name

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
        # LRU-bounded gauges. Workflows that template provider/workflow
        # names (multi-tenant, scoped runs, dynamic env) produce unbounded
        # keys; without an explicit cap the export callback iterates the
        # full dict on every OTel scrape (Prometheus default 15 s) and
        # eventually stalls. Cap defaults to 1024 — override via env.
        max_keys = int(os.environ.get("AGENTLOOM_METRICS_MAX_KEYS", "1024"))
        self._max_metric_keys = max_keys
        self._circuit_states: OrderedDict[str, int] = OrderedDict()
        self._budget_remaining: OrderedDict[str, float] = OrderedDict()

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
        # Canonical OTel GenAI client metric — replaces the AgentLoom-prefixed
        # ``agentloom_provider_latency_seconds`` with the spec's name.
        self._operation_duration_histogram = meter.create_histogram(
            "gen_ai.client.operation.duration",
            description="GenAI operation duration (per OTel GenAI conventions)",
            unit="s",
        )
        # Canonical OTel GenAI token usage histogram — replaces the
        # AgentLoom counter ``agentloom_tokens_total``. The OTel spec
        # mandates a histogram so distributions are queryable.
        self._token_histogram = meter.create_histogram(
            "gen_ai.client.token.usage",
            description="GenAI token usage per operation (per OTel GenAI conventions)",
            unit="{token}",
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
        # Canonical OTel GenAI metric — replaces the AgentLoom-prefixed
        # ``agentloom_time_to_first_token_seconds`` with the spec name.
        self._time_to_first_chunk_histogram = meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            description="GenAI streaming time-to-first-chunk (per OTel GenAI conventions)",
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
        # GenAI client token-usage histogram — Prometheus fallback name
        # mirrors the canonical OTel form (``gen_ai.client.token.usage``)
        # with dots collapsed to underscores per the OTLP→Prom convention.
        self._prom_histograms["token_usage"] = prom.Histogram(
            "gen_ai_client_token_usage",
            "GenAI token usage per operation (per OTel GenAI conventions)",
            ["gen_ai_provider_name", "gen_ai_request_model", "gen_ai_token_type"],
        )
        self._prom_histograms["step_duration"] = prom.Histogram(
            "agentloom_step_duration_seconds",
            "Step execution duration",
            ["step_type", "stream"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )
        self._prom_histograms["operation_duration"] = prom.Histogram(
            "gen_ai_client_operation_duration_seconds",
            "GenAI operation duration (per OTel GenAI conventions)",
            ["gen_ai_provider_name", "stream"],
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
        self._prom_histograms["time_to_first_chunk"] = prom.Histogram(
            "gen_ai_client_operation_time_to_first_chunk_seconds",
            "GenAI streaming time-to-first-chunk (per OTel GenAI conventions)",
            ["gen_ai_provider_name", "gen_ai_request_model"],
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
            # Counter (AgentLoom-specific) — keeps the per-(provider,model)
            # call count; OTel's spec doesn't have a direct equivalent.
            self._provider_counter.add(
                1, {"provider": provider, "model": model, "stream": stream_str}
            )
            # Histogram — canonical OTel GenAI client metric. Attributes
            # follow the spec: ``gen_ai.operation.name`` + ``gen_ai.provider.name``.
            self._operation_duration_histogram.record(
                latency_s,
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": to_genai_provider_name(provider),
                    "gen_ai.request.model": model,
                    "stream": stream_str,
                },
            )
        else:  # pragma: no cover — prom fallback
            self._prom_counters["provider_calls"].labels(
                provider=provider, model=model, stream=stream_str
            ).inc()
            self._prom_histograms["operation_duration"].labels(
                gen_ai_provider_name=to_genai_provider_name(provider), stream=stream_str
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
        *,
        reasoning_tokens: int = 0,
    ) -> None:
        """Record token usage on the ``gen_ai.client.token.usage`` histogram.

        Each call generates one observation per non-zero token type with
        the canonical OTel GenAI attributes (``gen_ai.operation.name``,
        ``gen_ai.provider.name``, ``gen_ai.request.model``,
        ``gen_ai.token.type``). ``reasoning_tokens`` rides under
        ``gen_ai.token.type="reasoning"`` — an AgentLoom extension to the
        OTel registry's ``input``/``output`` enum so chain-of-thought spend
        stays attributable on dashboards. Sourced from OpenAI o-series
        ``completion_tokens_details.reasoning_tokens`` and Gemini 2.5+
        ``thoughtsTokenCount``; Anthropic / Ollama don't expose a split
        and stay at ``0``.
        """
        if not self._enabled:
            return
        canonical_provider = to_genai_provider_name(provider)
        if self._backend == "otel":
            common: dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": canonical_provider,
                "gen_ai.request.model": model,
            }
            self._token_histogram.record(prompt_tokens, {**common, "gen_ai.token.type": "input"})
            self._token_histogram.record(
                completion_tokens, {**common, "gen_ai.token.type": "output"}
            )
            if reasoning_tokens:
                self._token_histogram.record(
                    reasoning_tokens, {**common, "gen_ai.token.type": "reasoning"}
                )
        else:  # pragma: no cover — prom fallback
            self._prom_histograms["token_usage"].labels(
                gen_ai_provider_name=canonical_provider,
                gen_ai_request_model=model,
                gen_ai_token_type="input",
            ).observe(prompt_tokens)
            self._prom_histograms["token_usage"].labels(
                gen_ai_provider_name=canonical_provider,
                gen_ai_request_model=model,
                gen_ai_token_type="output",
            ).observe(completion_tokens)
            if reasoning_tokens:
                self._prom_histograms["token_usage"].labels(
                    gen_ai_provider_name=canonical_provider,
                    gen_ai_request_model=model,
                    gen_ai_token_type="reasoning",
                ).observe(reasoning_tokens)

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
        canonical_provider = to_genai_provider_name(provider)
        if self._backend == "otel":
            self._time_to_first_chunk_histogram.record(
                ttft_s,
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": canonical_provider,
                    "gen_ai.request.model": model,
                },
            )
        else:  # pragma: no cover — prom fallback
            self._prom_histograms["time_to_first_chunk"].labels(
                gen_ai_provider_name=canonical_provider, gen_ai_request_model=model
            ).observe(ttft_s)

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

    def _bound_set(self, dct: OrderedDict[str, Any], key: str, value: Any) -> None:
        """Insert ``key=value`` into *dct* with LRU eviction past the cap.

        Touches existing keys to MRU; on overflow drops the oldest key so
        the export callback's iteration cost stays bounded regardless of
        upstream key cardinality.
        """
        if key in dct:
            dct.move_to_end(key)
            dct[key] = value
            return
        dct[key] = value
        if len(dct) > self._max_metric_keys:
            dct.popitem(last=False)

    def set_budget_remaining(self, workflow: str, remaining: float) -> None:
        if not self._enabled:
            return
        self._bound_set(self._budget_remaining, workflow, remaining)
        if self._backend == "prom":  # pragma: no cover — prom fallback
            self._prom_gauges["budget_remaining"].labels(workflow=workflow).set(remaining)

    def set_circuit_state(self, provider: str, state: int) -> None:
        if not self._enabled:
            return
        # OTel: update dict read by the observable gauge callback
        self._bound_set(self._circuit_states, provider, state)
        if self._backend == "prom":  # pragma: no cover — prom fallback
            self._prom_gauges["circuit_state"].labels(provider=provider).set(state)

    def shutdown(self) -> None:
        """Flush pending metrics before process exit."""
        if self._backend == "otel" and self._meter_provider:
            self._meter_provider.shutdown()
