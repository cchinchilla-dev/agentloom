"""Tests for the ProviderGateway module."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.gateway import ProviderGateway
from tests.conftest import MockProvider


# StreamResponse unit tests
class TestStreamResponse:
    async def test_max_accumulated_bytes_exceeded(self) -> None:
        sr = StreamResponse(model="m", provider="p")

        async def _huge() -> AsyncIterator[str]:
            while True:
                yield "x" * 4096

        sr._set_iterator(_huge())
        with pytest.raises(ProviderError, match="byte limit"):
            async for _ in sr:
                pass

    async def test_content_accumulates(self) -> None:
        sr = StreamResponse(model="m", provider="p")

        async def _gen() -> AsyncIterator[str]:
            yield "hello"
            yield " world"

        sr._set_iterator(_gen())
        chunks = [c async for c in sr]
        assert chunks == ["hello", " world"]
        assert sr.content == "hello world"

    async def test_to_provider_response(self) -> None:
        from agentloom.core.results import TokenUsage

        sr = StreamResponse(model="gpt-4o", provider="openai")
        sr.usage = TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        sr.cost_usd = 0.001
        sr.finish_reason = "stop"

        async def _gen() -> AsyncIterator[str]:
            yield "ok"

        sr._set_iterator(_gen())
        async for _ in sr:
            pass
        resp = sr.to_provider_response()
        assert resp.content == "ok"
        assert resp.usage.total_tokens == 8
        assert resp.cost_usd == 0.001
        assert resp.finish_reason == "stop"


# ProviderGateway tests
class MidStreamFailProvider(BaseProvider):
    """A provider that yields some chunks then fails mid-stream."""

    name = "midstream_fail"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise ProviderError("midstream_fail", "Not used")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        sr = StreamResponse(model=model, provider="midstream_fail")

        async def _generate() -> AsyncIterator[str]:
            yield "Hello"
            yield " world"
            raise ConnectionError("Connection dropped mid-stream")

        sr._set_iterator(_generate())
        return sr

    def supports_model(self, model: str) -> bool:
        return True


class FailingProvider(BaseProvider):
    """A provider that always raises an error on complete()."""

    name = "failing"

    def __init__(self, error_msg: str = "Provider exploded") -> None:
        super().__init__()
        self._error_msg = error_msg
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.calls.append({"messages": messages, "model": model})
        raise ProviderError("failing", self._error_msg)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        self.calls.append({"messages": messages, "model": model})
        sr = StreamResponse(model=model, provider="failing")

        async def _generate() -> AsyncIterator[str]:
            raise ProviderError("failing", self._error_msg)
            yield ""

        sr._set_iterator(_generate())
        return sr

    def supports_model(self, model: str) -> bool:
        return True


class TestProviderRouting:
    """Test provider selection and routing."""

    async def test_single_provider_routes_correctly(self) -> None:
        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)

        response = await gateway.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        assert response.content == "Mock response"
        assert response.provider == "mock"

    async def test_higher_priority_provider_used_first(self) -> None:
        gateway = ProviderGateway()
        low_priority = MockProvider(responses={"hello": "low priority answer"})
        low_priority.name = "low"
        high_priority = MockProvider(responses={"hello": "high priority answer"})
        high_priority.name = "high"

        gateway.register(low_priority, priority=10)
        gateway.register(high_priority, priority=0)

        response = await gateway.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        assert response.content == "high priority answer"

    async def test_no_provider_raises_error(self) -> None:
        gateway = ProviderGateway()
        # Register a provider that does not support the model
        provider = MockProvider()
        provider.supports_model = lambda m: False  # type: ignore[assignment]
        gateway.register(provider, priority=0)

        with pytest.raises(ProviderError, match="No provider registered"):
            await gateway.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="unsupported-model",
            )


class TestProviderFallback:
    """Test fallback behavior when primary provider fails."""

    async def test_fallback_on_primary_failure(self) -> None:
        gateway = ProviderGateway()
        failing = FailingProvider()
        succeeding = MockProvider(responses={"hello": "fallback answer"})

        gateway.register(failing, priority=0)
        gateway.register(succeeding, priority=1)

        response = await gateway.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        assert response.content == "fallback answer"
        # The failing provider should have been called first
        assert len(failing.calls) == 1

    async def test_all_providers_fail_raises_error(self) -> None:
        gateway = ProviderGateway()
        failing1 = FailingProvider(error_msg="first failed")
        failing1.name = "failing1"
        failing2 = FailingProvider(error_msg="second failed")
        failing2.name = "failing2"

        gateway.register(failing1, priority=0)
        gateway.register(failing2, priority=1)

        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="test-model",
            )

    async def test_first_succeeds_no_fallback_called(self) -> None:
        gateway = ProviderGateway()
        primary = MockProvider(responses={"hello": "primary answer"})
        primary.name = "primary"
        fallback = MockProvider(responses={"hello": "fallback answer"})
        fallback.name = "fallback_provider"

        gateway.register(primary, priority=0)
        gateway.register(fallback, priority=1, is_fallback=True)

        response = await gateway.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        assert response.content == "primary answer"
        assert len(fallback.calls) == 0

    async def test_custom_response_mapping(self) -> None:
        gateway = ProviderGateway()
        provider = MockProvider(
            responses={
                "What is Python?": "Python is a programming language",
                "What is Rust?": "Rust is a systems language",
            }
        )
        gateway.register(provider, priority=0)

        resp1 = await gateway.complete(
            messages=[{"role": "user", "content": "What is Python?"}],
            model="test-model",
        )
        assert resp1.content == "Python is a programming language"

        resp2 = await gateway.complete(
            messages=[{"role": "user", "content": "What is Rust?"}],
            model="test-model",
        )
        assert resp2.content == "Rust is a systems language"


class TestProviderSpansPerFallbackAttempt:
    """The gateway emits one provider span per attempt during fallback so
    Jaeger / Grafana can show the failed primary attempt alongside the
    successful fallback. Without this the trace tree collapses both into
    a single span and the latency split between attempts is invisible.
    """

    async def test_provider_span_emitted_per_fallback_attempt(self) -> None:
        from unittest.mock import MagicMock

        gateway = ProviderGateway()
        failing = FailingProvider(error_msg="primary down")
        failing.name = "primary"
        succeeding = MockProvider(responses={"hello": "fallback answer"})
        succeeding.name = "fallback_provider"

        gateway.register(failing, priority=0)
        gateway.register(succeeding, priority=1)

        observer = MagicMock()
        gateway.set_observer(observer)

        response = await gateway.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            step_id="my_step",
        )
        assert response.content == "fallback answer"

        # Two start hooks (one per attempt) and two end hooks.
        assert observer.on_provider_call_start.call_count == 2
        assert observer.on_provider_call_end.call_count == 2

        # First attempt: primary, attempt=0.
        first_start = observer.on_provider_call_start.call_args_list[0]
        assert first_start.kwargs["provider"] == "primary"
        assert first_start.kwargs["attempt"] == 0
        assert first_start.kwargs["step_id"] == "my_step"

        # First attempt closed with error.
        first_end = observer.on_provider_call_end.call_args_list[0]
        assert first_end.kwargs["provider"] == "primary"
        assert first_end.kwargs["attempt"] == 0
        assert first_end.kwargs["error"] is not None

        # Second attempt: fallback, attempt=1, success.
        second_start = observer.on_provider_call_start.call_args_list[1]
        assert second_start.kwargs["provider"] == "fallback_provider"
        assert second_start.kwargs["attempt"] == 1

        second_end = observer.on_provider_call_end.call_args_list[1]
        assert second_end.kwargs["provider"] == "fallback_provider"
        assert second_end.kwargs["attempt"] == 1
        assert second_end.kwargs.get("error") is None

    async def test_no_step_id_skips_provider_spans(self) -> None:
        # Backwards compat: callers that don't pass step_id (legacy code,
        # tests) must not cause an exception. The hooks simply aren't fired.
        from unittest.mock import MagicMock

        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)
        observer = MagicMock()
        gateway.set_observer(observer)

        await gateway.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
        )
        assert observer.on_provider_call_start.call_count == 0
        assert observer.on_provider_call_end.call_count == 0

    async def test_step_id_not_forwarded_to_provider(self) -> None:
        # The step_id kwarg is consumed by the gateway and must not leak
        # into the provider's HTTP payload (it would be a 400 from any
        # provider that validates extras).
        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)
        await gateway.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            step_id="should_be_consumed",
        )
        # MockProvider records each call's kwargs; assert step_id absent.
        assert "step_id" not in provider.calls[0]


class TestGatewayStreaming:
    """Test gateway stream() method."""

    async def test_stream_routes_to_provider(self) -> None:
        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)

        sr = await gateway.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        chunks = [chunk async for chunk in sr]
        assert "".join(chunks) == "Mock response"
        assert sr.usage.total_tokens == 30

    async def test_stream_fallback_on_circuit_open(self) -> None:
        gateway = ProviderGateway()
        failing = FailingProvider()
        succeeding = MockProvider(responses={"hello": "fallback answer"})

        gateway.register(failing, priority=0, circuit_fail_threshold=1)
        gateway.register(succeeding, priority=1)

        # First call: failing provider is tried, fails at iteration, CB records failure
        sr1 = await gateway.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        with pytest.raises(ProviderError):
            async for _ in sr1:
                pass

        # Second call: failing provider circuit is open, falls through to succeeding
        sr2 = await gateway.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        chunks = [chunk async for chunk in sr2]
        assert "".join(chunks) == "fallback answer"

    async def test_stream_no_provider_raises(self) -> None:
        gateway = ProviderGateway()
        provider = MockProvider()
        provider.supports_model = lambda m: False  # type: ignore[assignment]
        gateway.register(provider, priority=0)

        with pytest.raises(ProviderError, match="No provider registered"):
            await gateway.stream(
                messages=[{"role": "user", "content": "hello"}],
                model="unsupported",
            )

    async def test_stream_midstream_failure_records_cb_failure(self) -> None:
        gateway = ProviderGateway()
        provider = MidStreamFailProvider()
        gateway.register(provider, priority=0, circuit_fail_threshold=1)

        sr = await gateway.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
        )
        with pytest.raises(ConnectionError, match="mid-stream"):
            async for _ in sr:
                pass

        # Circuit breaker should have recorded the failure
        from agentloom.resilience.circuit_breaker import CircuitState

        cb = gateway._providers[0].circuit_breaker
        assert cb.state == CircuitState.OPEN

    async def test_stream_accumulates_metadata(self) -> None:
        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)

        sr = await gateway.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        async for _ in sr:
            pass
        resp = sr.to_provider_response()
        assert resp.content == "Mock response"
        assert resp.usage.total_tokens == 30
        assert resp.cost_usd > 0


class TestStreamCancellationDoesNotFailCircuit:
    """A caller cancelling a stream must not be counted as a provider fault."""

    async def test_stream_aclose_does_not_record_failure(self) -> None:
        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)

        sr = await gateway.stream(
            messages=[{"role": "user", "content": "one two three four five"}],
            model="test-model",
        )
        # Consume one chunk then abort the underlying generator explicitly.
        iterator = sr.__aiter__()
        await iterator.__anext__()
        assert sr._iterator is not None
        await sr._iterator.aclose()

        from agentloom.resilience.circuit_breaker import CircuitState

        cb = gateway._providers[0].circuit_breaker
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerOrderingInCompleteFlow:
    """Gateway must not consume rate-limit tokens when the circuit is open."""

    async def test_circuit_open_does_not_consume_rate_limit_tokens(self) -> None:
        from agentloom.resilience.circuit_breaker import CircuitState

        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0, max_rpm=5, max_tpm=1000)
        entry = gateway._providers[0]

        # Force the circuit OPEN.
        for _ in range(entry.circuit_breaker.fail_threshold):
            entry.circuit_breaker.record_failure()
        assert entry.circuit_breaker.state == CircuitState.OPEN

        rpm_before = entry.rate_limiter._request_tokens
        tpm_before = entry.rate_limiter._token_tokens

        with pytest.raises(ProviderError):
            await gateway.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="test-model",
            )

        # Rate limiter must be untouched.
        assert entry.rate_limiter._request_tokens == rpm_before
        assert entry.rate_limiter._token_tokens == tpm_before


class TestCandidateCacheLRU:
    """The model→candidates cache must evict LRU entries past its bound.

    Without this, dynamic model strings (templated names, multi-tenant
    naming) accumulate entries forever — each lookup eventually walks
    the full table.
    """

    def test_candidate_cache_evicts_lru(self) -> None:
        gateway = ProviderGateway(candidate_cache_max=3)
        provider = MockProvider()
        gateway.register(provider, priority=0)

        # Fill with 3 entries.
        for i in range(3):
            gateway._get_candidates(f"model-{i}")
        assert list(gateway._candidate_cache.keys()) == [
            "model-0",
            "model-1",
            "model-2",
        ]

        # Touch model-0 → it becomes MRU.
        gateway._get_candidates("model-0")
        assert list(gateway._candidate_cache.keys()) == [
            "model-1",
            "model-2",
            "model-0",
        ]

        # Adding model-3 evicts model-1 (oldest).
        gateway._get_candidates("model-3")
        assert list(gateway._candidate_cache.keys()) == [
            "model-2",
            "model-0",
            "model-3",
        ]

    def test_candidate_cache_max_from_env(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("AGENTLOOM_CANDIDATE_CACHE_MAX", "5")
        gateway = ProviderGateway()
        assert gateway._candidate_cache_max == 5

    def test_candidate_cache_default_bound(self) -> None:
        gateway = ProviderGateway()
        # Default tracks the class constant — change both together.
        assert gateway._candidate_cache_max == ProviderGateway._DEFAULT_CANDIDATE_CACHE_MAX

    def test_explicit_model_mapping_bypasses_cache(self) -> None:
        gateway = ProviderGateway(candidate_cache_max=2)
        provider = MockProvider()
        gateway.register(provider, priority=0, models=["explicit-model"])

        # Explicit mapping must not consume cache slots.
        for _ in range(5):
            gateway._get_candidates("explicit-model")
        assert "explicit-model" not in gateway._candidate_cache


class TestObserverWiring:
    """Observers attached to the gateway must receive circuit-breaker state changes."""

    async def test_register_after_set_observer_wires_callback(self) -> None:
        gateway = ProviderGateway()
        events: list[tuple[str, str, str]] = []

        class Observer:
            def on_circuit_state_change(self, name: str, old: str, new: str) -> None:
                events.append((name, old, new))

        gateway.set_observer(Observer())

        # Register provider AFTER set_observer — must wire callback at register time.
        failing = FailingProvider(error_msg="x")
        gateway.register(failing, priority=0, circuit_fail_threshold=1)

        with pytest.raises(ProviderError):
            await gateway.complete(messages=[{"role": "user", "content": "hi"}], model="m")

        # The single failure with threshold=1 trips the breaker → state change emitted.
        assert any(new == "open" for _, _, new in events)

    def test_observer_without_callback_attribute_does_not_crash(self) -> None:
        """A bare object lacking ``on_circuit_state_change`` must not break wiring."""
        gateway = ProviderGateway()
        gateway.set_observer(object())  # no hook attribute
        provider = MockProvider()
        gateway.register(provider, priority=0)

        # Trigger _set_state by transitioning manually — must not raise.
        provider_entry = gateway._providers[0]
        provider_entry.circuit_breaker._set_state(
            provider_entry.circuit_breaker._state.__class__.OPEN
        )


class TestGatewayStreamErrorPaths:
    """Setup/error branches in ``ProviderGateway.stream()``."""

    async def test_stream_rate_limit_releases_half_open_slot(self) -> None:
        """When ``provider.stream()`` raises RateLimitError while the breaker
        is HALF_OPEN, the gateway must release the slot it just claimed via
        ``allow_request()`` so future calls can proceed."""
        from agentloom.exceptions import RateLimitError
        from agentloom.resilience.circuit_breaker import CircuitState

        class ThrottledProvider(BaseProvider):
            name = "throttled"

            async def complete(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            async def stream(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise RateLimitError("throttled", retry_after_s=1.0)

            def supports_model(self, model: str) -> bool:
                return True

        gateway = ProviderGateway()
        gateway.register(
            ThrottledProvider(),
            priority=0,
            circuit_fail_threshold=1,
            circuit_reset_timeout=0.01,
        )
        cb = gateway._providers[0].circuit_breaker

        # Force the breaker into HALF_OPEN so allow_request claims a slot.
        import time as _time

        cb._set_state(CircuitState.OPEN)
        cb._last_failure_time = _time.monotonic() - 1.0  # past reset_timeout

        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.stream(messages=[{"role": "user", "content": "hi"}], model="any-model")

        # Slot released — counter back to 0.
        assert cb._half_open_calls == 0

    async def test_stream_setup_failure_records_breaker_failure(self) -> None:
        """Setup failures raised by ``provider.stream()`` itself (not the
        iterator) must feed back to the circuit breaker."""

        class SetupFailProvider(BaseProvider):
            name = "setup_fail"

            async def complete(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            async def stream(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise ProviderError("setup_fail", "connection refused")

            def supports_model(self, model: str) -> bool:
                return True

        gateway = ProviderGateway()
        gateway.register(SetupFailProvider(), priority=0, circuit_fail_threshold=3)

        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.stream(messages=[{"role": "user", "content": "hi"}], model="any-model")

        cb = gateway._providers[0].circuit_breaker
        assert cb._failure_count >= 1

    async def test_stream_iterator_failure_records_breaker_failure(self) -> None:
        """An exception raised by the iterator (after stream() returned)
        must invoke ``record_failure()`` and surface to the caller."""
        gateway = ProviderGateway()
        gateway.register(MidStreamFailProvider(), priority=0, circuit_fail_threshold=2)

        cb = gateway._providers[0].circuit_breaker
        sr = await gateway.stream(messages=[{"role": "user", "content": "hi"}], model="any-model")
        with pytest.raises(ConnectionError):
            async for _ in sr:
                pass

        assert cb._failure_count == 1
