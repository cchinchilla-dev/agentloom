"""Tests for MetricsManager — exercises available backend."""

from __future__ import annotations

import pytest

from agentloom.observability.metrics import (
    _HAS_OTEL_METRICS,
    _HAS_PROM,
    MetricsManager,
)

_HAS_BACKEND = _HAS_OTEL_METRICS or _HAS_PROM


class TestMetricsDisabled:
    def test_disabled_no_error(self) -> None:
        mm = MetricsManager(enabled=False)
        assert not mm._enabled
        mm.record_workflow_run("wf", "success", 1.0, 0.01)
        mm.record_step_execution("llm_call", "success", 0.5)
        mm.record_provider_call("openai", "gpt-4o-mini", 0.3)
        mm.record_provider_error("openai", "timeout")
        mm.record_tokens("openai", "gpt-4o-mini", 100, 50)
        mm.set_budget_remaining("wf", 0.5)
        mm.set_circuit_state("openai", 1)
        mm.shutdown()


@pytest.mark.skipif(not _HAS_BACKEND, reason="No metrics backend installed")
class TestMetricsEnabled:
    """Test with whatever backend is available (OTel in CI, may vary locally)."""

    def test_init(self) -> None:
        mm = MetricsManager(enabled=True)
        assert mm._enabled
        assert mm._backend in ("otel", "prom", "none")
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_workflow_run(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_workflow_run("wf", "success", 1.5, 0.01)
        mm.record_workflow_run("wf", "failed", 0.5)  # no cost
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_step_execution(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_step_execution("llm_call", "success", 0.5)
        mm.record_step_execution("tool", "failed", 0.1)
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_provider_call(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_provider_call("openai", "gpt-4o-mini", 0.3)
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_provider_error(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_provider_error("openai", "timeout")
        mm.record_provider_error("anthropic", "rate_limit")
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_tokens(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_tokens("openai", "gpt-4o-mini", 100, 50)
        if mm._backend == "otel":
            mm.shutdown()

    def test_set_circuit_state(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.set_circuit_state("openai", 0)  # closed
        mm.set_circuit_state("openai", 1)  # open
        mm.set_circuit_state("openai", 2)  # half_open
        assert mm._circuit_states["openai"] == 2
        if mm._backend == "otel":
            mm.shutdown()

    def test_set_budget_remaining(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.set_budget_remaining("wf", 0.42)
        assert mm._budget_remaining["wf"] == 0.42
        if mm._backend == "otel":
            mm.shutdown()

    def test_shutdown(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.shutdown()
        # Double shutdown should be safe
        mm.shutdown()


class TestMetricsOTelSetup:
    """Verify OTel instruments are created when available."""

    def test_otel_instruments_created(self) -> None:
        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            return
        assert mm._workflow_counter is not None
        assert mm._step_counter is not None
        assert mm._step_histogram is not None
        assert mm._provider_counter is not None
        assert mm._provider_error_counter is not None
        assert mm._provider_histogram is not None
        assert mm._token_counter is not None
        assert mm._workflow_histogram is not None
        assert mm._cost_counter is not None
        mm.shutdown()
