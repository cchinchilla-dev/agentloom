"""Prometheus metrics wrapper — gracefully degrades if not installed."""

from __future__ import annotations

from typing import Any

from agentloom.compat import is_available, try_import

prom = try_import("prometheus_client", extra="observability")
_HAS_PROM = is_available(prom)


class MetricsManager:
    """Manages Prometheus metrics. Returns no-ops if prometheus-client is not installed."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and _HAS_PROM
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}

        if self._enabled:
            self._setup()

    def _setup(self) -> None:
        """Create Prometheus metrics."""
        self._counters["workflow_runs"] = prom.Counter(
            "agentloom_workflow_runs_total",
            "Total workflow executions",
            ["workflow", "status"],
        )
        self._counters["step_executions"] = prom.Counter(
            "agentloom_step_executions_total",
            "Total step executions",
            ["step_type", "status"],
        )
        self._counters["provider_calls"] = prom.Counter(
            "agentloom_provider_calls_total",
            "Total provider API calls",
            ["provider", "model"],
        )
        self._counters["provider_errors"] = prom.Counter(
            "agentloom_provider_errors_total",
            "Total provider errors",
            ["provider", "error_type"],
        )
        self._counters["tokens_total"] = prom.Counter(
            "agentloom_tokens_total",
            "Total tokens consumed",
            ["provider", "model", "direction"],
        )

        self._histograms["step_duration"] = prom.Histogram(
            "agentloom_step_duration_seconds",
            "Step execution duration",
            ["step_type"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )
        self._histograms["provider_latency"] = prom.Histogram(
            "agentloom_provider_latency_seconds",
            "Provider API call latency",
            ["provider"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
        )

        self._gauges["budget_remaining"] = prom.Gauge(
            "agentloom_budget_remaining_usd",
            "Remaining budget in USD",
            ["workflow"],
        )
        self._gauges["circuit_state"] = prom.Gauge(
            "agentloom_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["provider"],
        )

    def record_workflow_run(self, workflow: str, status: str) -> None:
        if self._enabled:
            self._counters["workflow_runs"].labels(workflow=workflow, status=status).inc()

    def record_step_execution(self, step_type: str, status: str, duration_s: float) -> None:
        if self._enabled:
            self._counters["step_executions"].labels(step_type=step_type, status=status).inc()
            self._histograms["step_duration"].labels(step_type=step_type).observe(duration_s)

    def record_provider_call(self, provider: str, model: str, latency_s: float) -> None:
        if self._enabled:
            self._counters["provider_calls"].labels(provider=provider, model=model).inc()
            self._histograms["provider_latency"].labels(provider=provider).observe(latency_s)

    def record_provider_error(self, provider: str, error_type: str) -> None:
        if self._enabled:
            self._counters["provider_errors"].labels(provider=provider, error_type=error_type).inc()

    def record_tokens(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        if self._enabled:
            self._counters["tokens_total"].labels(
                provider=provider, model=model, direction="input"
            ).inc(prompt_tokens)
            self._counters["tokens_total"].labels(
                provider=provider, model=model, direction="output"
            ).inc(completion_tokens)

    def set_budget_remaining(self, workflow: str, remaining: float) -> None:
        if self._enabled:
            self._gauges["budget_remaining"].labels(workflow=workflow).set(remaining)

    def set_circuit_state(self, provider: str, state: int) -> None:
        if self._enabled:
            self._gauges["circuit_state"].labels(provider=provider).set(state)
