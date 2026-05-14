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
        mm.record_approval_gate("wf", "approved")
        mm.record_webhook_delivery("wf", "success", 0.5)
        mm.record_attachments("llm_call", 2)
        mm.record_stream_response("openai", "gpt-4o-mini")
        mm.record_time_to_first_token("openai", "gpt-4o-mini", 0.1)
        mm.record_mock_replay("wf", "step_id")
        mm.record_recording_capture("anthropic", "m", 0.5)
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

    def test_record_tool_call(self) -> None:
        # Records both the per-tool counter (with status label) and the
        # latency histogram. Status maps to ``"success"`` / ``"failure"``
        # so dashboards can derive a per-tool failure rate.
        from unittest.mock import MagicMock

        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            return
        counter_spy = MagicMock()
        histogram_spy = MagicMock()
        mm._tool_call_counter = counter_spy
        mm._tool_call_histogram = histogram_spy
        mm.record_tool_call("add", success=True, duration_s=0.42)
        mm.record_tool_call("add", success=False, duration_s=1.1)

        counter_spy.add.assert_any_call(1, {"tool_name": "add", "status": "success"})
        counter_spy.add.assert_any_call(1, {"tool_name": "add", "status": "failure"})
        histogram_spy.record.assert_any_call(0.42, {"tool_name": "add"})
        histogram_spy.record.assert_any_call(1.1, {"tool_name": "add"})
        mm.shutdown()

    def test_reasoning_tokens_metric_emitted(self) -> None:
        # When ``reasoning_tokens`` is non-zero, ``record_tokens`` must
        # observe the histogram a third time with
        # ``gen_ai.token.type="reasoning"`` so dashboards can split
        # chain-of-thought spend from regular completion spend.
        from unittest.mock import MagicMock

        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            # prom fallback exercised via smoke; OTel path is what we audit.
            return
        spy = MagicMock()
        mm._token_histogram = spy
        mm.record_tokens("anthropic", "claude-opus-4", 100, 50, reasoning_tokens=200)
        # Expect 3 record() calls: input, output, reasoning.
        assert spy.record.call_count == 3
        token_types = {call.args[1]["gen_ai.token.type"] for call in spy.record.call_args_list}
        assert token_types == {"input", "output", "reasoning"}
        # The reasoning observation must carry the right value and attrs.
        reasoning_calls = [
            call
            for call in spy.record.call_args_list
            if call.args[1]["gen_ai.token.type"] == "reasoning"
        ]
        assert len(reasoning_calls) == 1
        assert reasoning_calls[0].args[0] == 200
        assert reasoning_calls[0].args[1]["gen_ai.provider.name"] == "anthropic"
        assert reasoning_calls[0].args[1]["gen_ai.request.model"] == "claude-opus-4"
        mm.shutdown()

    def test_token_histogram_translates_provider_name(self) -> None:
        # The histogram must carry the canonical OTel ``gen_ai.provider.name``
        # value, not the AgentLoom internal short name. ``google`` →
        # ``gcp.gemini``. Without this translation, token series for Gemini
        # land on a non-canonical label and don't correlate with the spans
        # (which already translate via ``to_genai_provider_name``).
        from unittest.mock import MagicMock

        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            return
        spy = MagicMock()
        mm._token_histogram = spy
        mm.record_tokens("google", "gemini-2.5-flash", 50, 30)
        for call in spy.record.call_args_list:
            assert call.args[1]["gen_ai.provider.name"] == "gcp.gemini"
        mm.shutdown()

    def test_operation_duration_histogram_translates_provider_name(self) -> None:
        from unittest.mock import MagicMock

        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            return
        spy = MagicMock()
        mm._operation_duration_histogram = spy
        mm.record_provider_call("google", "gemini-2.5-flash", 0.42, stream=False)
        spy.record.assert_called_once()
        attrs = spy.record.call_args.args[1]
        assert attrs["gen_ai.provider.name"] == "gcp.gemini"

    def test_time_to_first_chunk_histogram_translates_provider_name(self) -> None:
        from unittest.mock import MagicMock

        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            return
        spy = MagicMock()
        mm._time_to_first_chunk_histogram = spy
        mm.record_time_to_first_token("google", "gemini-2.5-flash", 0.18)
        spy.record.assert_called_once()
        attrs = spy.record.call_args.args[1]
        assert attrs["gen_ai.provider.name"] == "gcp.gemini"

    def test_reasoning_tokens_zero_does_not_emit_third_observation(self) -> None:
        # Default path — no reasoning tokens — must not emit a reasoning
        # observation. Dashboards filtering on
        # ``gen_ai.token.type="reasoning"`` should see zero events for
        # non-thinking models.
        from unittest.mock import MagicMock

        mm = MetricsManager(enabled=True)
        if mm._backend != "otel":
            return
        spy = MagicMock()
        mm._token_histogram = spy
        mm.record_tokens("openai", "gpt-4o-mini", 100, 50)
        token_types = {call.args[1]["gen_ai.token.type"] for call in spy.record.call_args_list}
        assert "reasoning" not in token_types
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

    def test_record_approval_gate(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_approval_gate("wf", "approved")
        mm.record_approval_gate("wf", "rejected")
        mm.record_approval_gate("wf", "pending")
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_webhook_delivery(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_webhook_delivery("wf", "success", 0.5)
        mm.record_webhook_delivery("wf", "failed", 6.0)
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_attachments(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_attachments("llm_call", 3)
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_stream_response(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_stream_response("openai", "gpt-4o-mini")
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_time_to_first_token(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_time_to_first_token("openai", "gpt-4o-mini", 0.25)
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_mock_replay(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_mock_replay("wf", "step_id")
        mm.record_mock_replay("wf", "prompt_hash")
        mm.record_mock_replay("wf", "default")
        if mm._backend == "otel":
            mm.shutdown()

    def test_record_recording_capture(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.record_recording_capture("anthropic", "claude-haiku-4-5-20251001", 0.88)
        if mm._backend == "otel":
            mm.shutdown()

    def test_shutdown(self) -> None:
        mm = MetricsManager(enabled=True)
        mm.shutdown()
        # Double shutdown should be safe
        mm.shutdown()


class TestBoundedCardinality:
    """LRU cap on the lifelong process-global metric dicts. Backend-agnostic
    — the cap lives in pure Python, not the OTel/Prom layer."""

    def test_bounded_cardinality_circuit_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from collections import OrderedDict

        monkeypatch.setenv("AGENTLOOM_METRICS_MAX_KEYS", "3")
        mm = MetricsManager(enabled=False)  # backend-agnostic
        # Inject the bound directly — covers the cap regardless of backend.
        mm._max_metric_keys = 3
        states: OrderedDict[str, int] = OrderedDict()

        for i in range(3):
            mm._bound_set(states, f"provider-{i}", 0)
        assert list(states.keys()) == ["provider-0", "provider-1", "provider-2"]

        # Touch provider-0 → moves to MRU end.
        mm._bound_set(states, "provider-0", 1)
        assert list(states.keys()) == ["provider-1", "provider-2", "provider-0"]

        # Adding provider-3 evicts provider-1 (LRU).
        mm._bound_set(states, "provider-3", 0)
        assert list(states.keys()) == ["provider-2", "provider-0", "provider-3"]
        assert states["provider-0"] == 1  # value preserved on touch

    def test_bounded_cardinality_default_cap(self) -> None:
        # The default cap should be reasonable; we read it back from the
        # manager rather than hard-coding so a future tuning change only
        # needs one edit.
        mm = MetricsManager(enabled=False)
        assert mm._max_metric_keys >= 64


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
        assert mm._operation_duration_histogram is not None
        assert mm._token_histogram is not None
        assert mm._time_to_first_chunk_histogram is not None
        assert mm._workflow_histogram is not None
        assert mm._cost_counter is not None
        mm.shutdown()
