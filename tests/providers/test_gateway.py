"""Tests for the ProviderGateway module."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import httpx
import pytest
import respx

from agentloom.core.attempts import attempt_spend, enter_attempt
from agentloom.core.results import TokenUsage
from agentloom.exceptions import (
    ProviderError,
    RateLimitError,
    RecordingMismatchError,
    UnsupportedRequestError,
)
from agentloom.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    ProviderResponse,
    StreamResponse,
    ToolCall,
)
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.ollama import OllamaProvider
from agentloom.providers.openai import OpenAIProvider
from agentloom.resilience.circuit_breaker import CircuitBreaker, CircuitState
from tests.conftest import MockProvider


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
    """A provider that always fails: on ``complete()`` and on the first chunk of ``stream()``."""

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
        provider = MockProvider()
        provider.supports_model = lambda m: False  # type: ignore[assignment]
        gateway.register(provider, priority=0)

        with pytest.raises(ProviderError, match="No provider registered"):
            await gateway.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="unsupported-model",
            )


class TestProviderFallback:
    """Test fallback behaviour when the primary provider fails."""

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
    """One provider span per fallback attempt, so a trace splits the latency between attempts."""

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

        assert observer.on_provider_call_start.call_count == 2
        assert observer.on_provider_call_end.call_count == 2

        first_start = observer.on_provider_call_start.call_args_list[0]
        assert first_start.kwargs["provider"] == "primary"
        assert first_start.kwargs["attempt"] == 0
        assert first_start.kwargs["step_id"] == "my_step"

        first_end = observer.on_provider_call_end.call_args_list[0]
        assert first_end.kwargs["provider"] == "primary"
        assert first_end.kwargs["attempt"] == 0
        assert first_end.kwargs["error"] is not None

        second_start = observer.on_provider_call_start.call_args_list[1]
        assert second_start.kwargs["provider"] == "fallback_provider"
        assert second_start.kwargs["attempt"] == 1

        second_end = observer.on_provider_call_end.call_args_list[1]
        assert second_end.kwargs["provider"] == "fallback_provider"
        assert second_end.kwargs["attempt"] == 1
        assert second_end.kwargs.get("error") is None

    async def test_no_step_id_skips_provider_spans(self) -> None:
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
        # A provider that validates extras would answer 400 to a leaked ``step_id``.
        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)
        await gateway.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            step_id="should_be_consumed",
        )
        assert "step_id" not in provider.calls[0]

    async def test_circuit_open_after_start_closes_provider_span(self) -> None:
        # The breaker admits the call, then trips inside ``call()``: the span must still close.
        from unittest.mock import AsyncMock, MagicMock

        from agentloom.exceptions import CircuitOpenError

        gateway = ProviderGateway()
        provider = MockProvider()
        gateway.register(provider, priority=0)
        gateway._providers[0].circuit_breaker.allow_request = MagicMock(return_value=None)
        gateway._providers[0].circuit_breaker.call = AsyncMock(
            side_effect=CircuitOpenError("tripped")
        )

        observer = MagicMock()
        gateway.set_observer(observer)
        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                step_id="s1",
            )
        assert observer.on_provider_call_end.call_count == 1
        end_call = observer.on_provider_call_end.call_args
        assert end_call.kwargs["error"] == "CircuitOpenError"


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

    async def test_stream_forwards_step_id_when_provider_opts_in(self) -> None:
        # Recordings are keyed by step id, so a streaming workflow could never replay without it.
        class StreamStepIdCapture(MockProvider):
            accepts_step_id = True

            async def stream(
                self,
                messages: list[dict[str, Any]],
                model: str,
                temperature: float | None = None,
                max_tokens: int | None = None,
                **kwargs: Any,
            ) -> Any:
                self.calls.append({"stream_kwargs": dict(kwargs)})
                return await super().stream(messages, model, temperature, max_tokens, **kwargs)

        gateway = ProviderGateway()
        provider = StreamStepIdCapture()
        gateway.register(provider, priority=0)

        sr = await gateway.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            step_id="my_stream_step",
        )
        async for _ in sr:
            pass
        assert provider.calls[0]["stream_kwargs"].get("step_id") == "my_stream_step"

    async def test_stream_fallback_on_circuit_open(self) -> None:
        gateway = ProviderGateway()
        failing = FailingProvider()
        succeeding = MockProvider(responses={"hello": "fallback answer"})

        gateway.register(failing, priority=0, circuit_fail_threshold=1)
        gateway.register(succeeding, priority=1)

        # The first stream fails before any chunk and falls back; threshold 1 opens the circuit.
        sr1 = await gateway.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        assert "".join([chunk async for chunk in sr1]) == "fallback answer"

        # The second stream skips the open circuit and never reaches the failing provider.
        sr2 = await gateway.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        chunks = [chunk async for chunk in sr2]
        assert "".join(chunks) == "fallback answer"
        assert len(failing.calls) == 1

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

    async def test_stream_keeps_reasoning_and_tool_calls(self) -> None:
        class _ToolStream(BaseProvider):
            name = "tool_stream"

            async def complete(self, **_kw: Any) -> ProviderResponse:  # pragma: no cover
                raise NotImplementedError

            async def stream(self, **_kw: Any) -> StreamResponse:
                sr = StreamResponse(model="m", provider=self.name)

                async def _gen() -> AsyncIterator[str]:
                    yield "Let me check."
                    # Adapters fill these once the stream ends, as they do usage.
                    sr.reasoning_content = "The user wants the time."
                    sr.tool_calls = [ToolCall(id="c1", name="clock", arguments={})]

                sr._set_iterator(_gen())
                return sr

            def supports_model(self, _m: str) -> bool:
                return True

        gateway = ProviderGateway()
        gateway.register(_ToolStream(), models=["m"])
        sr = await gateway.stream(messages=[{"role": "user", "content": "time?"}], model="m")
        async for _ in sr:
            pass
        resp = sr.to_provider_response()
        assert resp.content == "Let me check."
        assert resp.reasoning_content == "The user wants the time."
        assert resp.tool_calls == [ToolCall(id="c1", name="clock", arguments={})]


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

        assert entry.rate_limiter._request_tokens == rpm_before
        assert entry.rate_limiter._token_tokens == tpm_before


class TestCandidateCacheLRU:
    """The model-to-candidates cache evicts LRU entries past its bound.

    Templated or multi-tenant model names would otherwise grow it without limit.
    """

    def test_candidate_cache_evicts_lru(self) -> None:
        gateway = ProviderGateway(candidate_cache_max=3)
        provider = MockProvider()
        gateway.register(provider, priority=0)

        for i in range(3):
            gateway._get_candidates(f"model-{i}")
        assert list(gateway._candidate_cache.keys()) == [
            "model-0",
            "model-1",
            "model-2",
        ]

        gateway._get_candidates("model-0")
        assert list(gateway._candidate_cache.keys()) == [
            "model-1",
            "model-2",
            "model-0",
        ]

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
        assert gateway._candidate_cache_max == ProviderGateway._DEFAULT_CANDIDATE_CACHE_MAX

    def test_explicit_model_mapping_bypasses_cache(self) -> None:
        gateway = ProviderGateway(candidate_cache_max=2)
        provider = MockProvider()
        gateway.register(provider, priority=0, models=["explicit-model"])

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

        failing = FailingProvider(error_msg="x")
        gateway.register(failing, priority=0, circuit_fail_threshold=1)

        with pytest.raises(ProviderError):
            await gateway.complete(messages=[{"role": "user", "content": "hi"}], model="m")

        assert any(new == "open" for _, _, new in events)

    def test_observer_without_callback_attribute_does_not_crash(self) -> None:
        """A bare object lacking ``on_circuit_state_change`` must not break wiring."""
        gateway = ProviderGateway()
        gateway.set_observer(object())
        provider = MockProvider()
        gateway.register(provider, priority=0)

        # The test passes if the transition does not raise.
        provider_entry = gateway._providers[0]
        provider_entry.circuit_breaker._set_state(
            provider_entry.circuit_breaker._state.__class__.OPEN
        )


class TestGatewayStreamErrorPaths:
    """Setup/error branches in ``ProviderGateway.stream()``."""

    async def test_stream_rate_limit_releases_half_open_slot(self) -> None:
        """A RateLimitError from ``stream()`` in HALF_OPEN gives back the half-open slot.

        Otherwise the slot ``allow_request()`` claimed would block every later call.
        """
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

        # OPEN with an expired timeout: ``allow_request`` moves to HALF_OPEN and claims a slot.
        import time as _time

        cb._set_state(CircuitState.OPEN)
        cb._last_failure_time = _time.monotonic() - 1.0  # past reset_timeout

        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.stream(messages=[{"role": "user", "content": "hi"}], model="any-model")

        assert cb._half_open_calls == 0

    async def test_stream_setup_failure_records_breaker_failure(self) -> None:
        """A failure raised by ``stream()`` itself, not by its iterator, reaches the breaker."""

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
        """An iterator failure after ``stream()`` returned is recorded and reaches the caller."""
        gateway = ProviderGateway()
        gateway.register(MidStreamFailProvider(), priority=0, circuit_fail_threshold=2)

        cb = gateway._providers[0].circuit_breaker
        sr = await gateway.stream(messages=[{"role": "user", "content": "hi"}], model="any-model")
        with pytest.raises(ConnectionError):
            async for _ in sr:
                pass

        assert cb._failure_count == 1

    async def test_stream_setup_failure_closes_provider_span(self) -> None:
        from unittest.mock import MagicMock

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

        observer = MagicMock()
        gateway.set_observer(observer)
        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.stream(
                messages=[{"role": "user", "content": "hi"}],
                model="any-model",
                step_id="s1",
            )
        observer.on_provider_call_end.assert_called_once()
        assert observer.on_provider_call_end.call_args.kwargs["error"] == "ProviderError"
        observer.on_provider_error.assert_called_once_with("setup_fail", "ProviderError")

    async def test_stream_rate_limit_closes_provider_span(self) -> None:
        from unittest.mock import MagicMock

        from agentloom.exceptions import RateLimitError

        class ThrottledProvider(BaseProvider):
            name = "throttled"

            async def complete(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            async def stream(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise RateLimitError("throttled", retry_after_s=1.0)

            def supports_model(self, model: str) -> bool:
                return True

        gateway = ProviderGateway()
        gateway.register(ThrottledProvider(), priority=0)

        observer = MagicMock()
        gateway.set_observer(observer)
        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.stream(
                messages=[{"role": "user", "content": "hi"}],
                model="any-model",
                step_id="s1",
            )
        observer.on_provider_call_end.assert_called_once()
        assert observer.on_provider_call_end.call_args.kwargs["error"] == "RateLimitError"

    async def test_stream_circuit_open_after_start_closes_span(self) -> None:
        # As with ``complete()``: the breaker admits the call, then trips inside ``stream()``.
        from unittest.mock import MagicMock

        from agentloom.exceptions import CircuitOpenError

        class TrippingProvider(BaseProvider):
            name = "tripping"

            async def complete(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            async def stream(self, *a, **kw):  # type: ignore[no-untyped-def]
                raise CircuitOpenError("tripped mid-stream")

            def supports_model(self, model: str) -> bool:
                return True

        gateway = ProviderGateway()
        gateway.register(TrippingProvider(), priority=0)

        observer = MagicMock()
        gateway.set_observer(observer)
        with pytest.raises(ProviderError, match="All providers failed"):
            await gateway.stream(
                messages=[{"role": "user", "content": "hi"}],
                model="any-model",
                step_id="s1",
            )
        observer.on_provider_call_end.assert_called_once()
        assert observer.on_provider_call_end.call_args.kwargs["error"] == "CircuitOpenError"
        assert observer.on_provider_call_end.call_args.kwargs["stream"] is True

    async def test_stream_iterator_failure_fires_provider_error_hook(self) -> None:
        from unittest.mock import MagicMock

        gateway = ProviderGateway()
        gateway.register(MidStreamFailProvider(), priority=0, circuit_fail_threshold=2)
        observer = MagicMock()
        gateway.set_observer(observer)

        sr = await gateway.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="any-model",
            step_id="s1",
        )
        with pytest.raises(ConnectionError):
            async for _ in sr:
                pass

        observer.on_provider_error.assert_called_once()
        # The end hook fires from the iterator's ``finally``.
        observer.on_provider_call_end.assert_called_once()


class _Scripted(BaseProvider):
    """A provider that streams ``script``: strings are chunks, an exception is raised in turn."""

    def __init__(
        self,
        name: str,
        script: list[Any] | None = None,
        supports: bool = True,
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.script = script if script is not None else ["ok"]
        self.supports = supports
        self.usage = usage
        self.models: list[str] = []

    async def complete(self, messages: Any, model: str, **_kw: Any) -> ProviderResponse:
        self.models.append(model)
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
        return ProviderResponse(content="".join(self.script), model=model, provider=self.name)

    async def stream(self, messages: Any, model: str, **_kw: Any) -> StreamResponse:
        self.models.append(model)
        sr = StreamResponse(model=model, provider=self.name)

        async def _generate() -> AsyncIterator[str]:
            if self.usage is not None:
                sr.usage = self.usage
                sr.cost_usd = 0.5
            for item in self.script:
                if isinstance(item, BaseException):
                    raise item
                yield item

        sr._set_iterator(_generate())
        return sr

    async def embed(self, inputs: list[str], model: str, **_kw: Any) -> EmbeddingResponse:
        self.models.append(model)
        return EmbeddingResponse(embeddings=[[0.0]] * len(inputs), model=model, provider=self.name)

    def supports_model(self, model: str) -> bool:
        return self.supports


_HI = [{"role": "user", "content": "hi"}]


def _half_open(gateway: ProviderGateway, index: int = 0) -> CircuitBreaker:
    """Put a candidate's breaker in the state where the next call claims the half-open slot."""
    cb = gateway._providers[index].circuit_breaker
    cb._set_state(CircuitState.OPEN)
    cb._last_failure_time = -1.0e9
    return cb


class TestStreamFallsBackBeforeFirstChunk:
    async def test_failure_before_the_first_chunk_moves_to_the_next_candidate(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [ProviderError("first", "503", status_code=503)]))
        gateway.register(_Scripted("second", ["a", "b"]), priority=1)

        sr = await gateway.stream(messages=_HI, model="m")

        assert [c async for c in sr] == ["a", "b"]
        assert sr.provider == "second"

    @respx.mock
    async def test_a_status_error_opening_the_request_falls_back(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(503, text="overloaded")
        )
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                content=json.dumps({"message": {"content": "local"}, "done": False}).encode(),
            )
        )
        gateway = ProviderGateway()
        gateway.register(OpenAIProvider(api_key="dummy-key"))
        gateway.register(OllamaProvider(), priority=1)

        sr = await gateway.stream(messages=_HI, model="gpt-4o")

        assert [c async for c in sr] == ["local"]
        assert sr.provider == "ollama"
        await gateway.close()

    async def test_every_candidate_failing_first_raises_the_gateway_error(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [ProviderError("first", "503", status_code=503)]))
        gateway.register(_Scripted("second", [ProviderError("second", "503", status_code=503)]))

        with pytest.raises(ProviderError, match="All providers failed") as exc:
            await gateway.stream(messages=_HI, model="m")

        assert exc.value.status_code == 503

    async def test_a_failure_after_the_first_chunk_does_not_fall_back(self) -> None:
        second = _Scripted("second")
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", ["a", ConnectionError("dropped")]))
        gateway.register(second, priority=1)

        sr = await gateway.stream(messages=_HI, model="m")

        received = []
        with pytest.raises(ConnectionError, match="dropped"):
            async for chunk in sr:
                received.append(chunk)
        assert received == ["a"]
        assert second.models == []

    async def test_the_first_chunk_is_served_once_and_in_order(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", ["a", "b", "c"]))

        sr = await gateway.stream(messages=_HI, model="m")

        assert [c async for c in sr] == ["a", "b", "c"]
        assert sr.content == "abc"

    async def test_an_empty_stream_is_an_answer(self) -> None:
        observer = MagicMock()
        second = _Scripted("second")
        gateway = ProviderGateway()
        gateway.set_observer(observer)
        gateway.register(_Scripted("first", []))
        gateway.register(second, priority=1)
        cb = _half_open(gateway, 0)

        sr = await gateway.stream(messages=_HI, model="m", step_id="s1")

        assert [c async for c in sr] == []
        assert second.models == []
        assert sr.time_to_first_token_ms is None
        assert cb.state == CircuitState.CLOSED
        observer.on_provider_call_end.assert_called_once()
        assert observer.on_provider_call_end.call_args.kwargs["error"] is None

    async def test_failing_first_feeds_the_breaker_and_closes_the_span(self) -> None:
        observer = MagicMock()
        gateway = ProviderGateway()
        gateway.set_observer(observer)
        gateway.register(_Scripted("first", [ProviderError("first", "boom")]))
        gateway.register(_Scripted("second"), priority=1)

        sr = await gateway.stream(messages=_HI, model="m", step_id="s1")
        [c async for c in sr]

        assert gateway._providers[0].circuit_breaker.failure_count == 1
        assert observer.on_provider_call_start.call_count == 2
        ends = observer.on_provider_call_end.call_args_list
        assert [c.kwargs["provider"] for c in ends] == ["first", "second"]
        assert [c.kwargs["error"] for c in ends] == ["ProviderError", None]
        observer.on_provider_error.assert_called_once_with("first", "ProviderError")

    async def test_a_429_before_the_first_chunk_frees_the_slot_without_a_failure(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [RateLimitError("first", retry_after_s=2.0)]))
        gateway.register(_Scripted("second"), priority=1)
        cb = _half_open(gateway, 0)

        sr = await gateway.stream(messages=_HI, model="m")
        [c async for c in sr]

        assert sr.provider == "second"
        assert cb._half_open_calls == 0
        assert cb.failure_count == 0

    async def test_one_refusal_on_the_first_pull_moves_on_without_a_failure(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [UnsupportedRequestError("first", "no audio")]))
        gateway.register(_Scripted("second"), priority=1)
        cb = _half_open(gateway, 0)

        sr = await gateway.stream(messages=_HI, model="m")
        [c async for c in sr]

        assert sr.provider == "second"
        assert cb.failure_count == 0
        assert cb._half_open_calls == 0

    async def test_every_candidate_refusing_on_the_first_pull_is_permanent(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [UnsupportedRequestError("first", "no audio")]))
        gateway.register(_Scripted("second", [UnsupportedRequestError("second", "no audio")]))

        with pytest.raises(UnsupportedRequestError, match="no audio"):
            await gateway.stream(messages=_HI, model="m")

    async def test_a_stale_recording_is_not_answered_by_another_provider(self) -> None:
        second = _Scripted("second")
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [RecordingMismatchError("drifted")]))
        gateway.register(second, priority=1)

        with pytest.raises(RecordingMismatchError):
            await gateway.stream(messages=_HI, model="m")
        assert second.models == []

    async def test_a_cancel_on_the_first_pull_frees_the_slot_and_closes_the_span(self) -> None:
        started = anyio.Event()

        class Hanging(_Scripted):
            async def stream(self, messages: Any, model: str, **_kw: Any) -> StreamResponse:
                sr = StreamResponse(model=model, provider=self.name)

                async def _generate() -> AsyncIterator[str]:
                    started.set()
                    await anyio.sleep_forever()
                    yield ""

                sr._set_iterator(_generate())
                return sr

        observer = MagicMock()
        second = _Scripted("second")
        gateway = ProviderGateway()
        gateway.set_observer(observer)
        gateway.register(Hanging("first"))
        gateway.register(second, priority=1)
        cb = _half_open(gateway, 0)

        with anyio.fail_after(5):  # a stream that never pulls would otherwise hang the test
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: gateway.stream(messages=_HI, model="m", step_id="s1"))
                await started.wait()
                tg.cancel_scope.cancel()

        assert cb._half_open_calls == 0
        assert cb.failure_count == 0
        assert second.models == []
        observer.on_provider_call_end.assert_called_once()

    async def test_a_failed_candidate_consumes_no_response_tokens(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [ProviderError("first", "boom")]))
        gateway.register(_Scripted("second", ["a"]), priority=1)
        limiter = gateway._providers[0].rate_limiter
        assert limiter is not None
        limiter.consume_response_tokens = AsyncMock()  # type: ignore[method-assign]

        sr = await gateway.stream(messages=_HI, model="m")
        [c async for c in sr]

        limiter.consume_response_tokens.assert_not_awaited()

    async def test_usage_billed_before_a_failed_first_pull_still_counts(self) -> None:
        enter_attempt(())
        usage = TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [ProviderError("first", "boom")], usage=usage))
        gateway.register(_Scripted("second", ["a"]), priority=1)

        sr = await gateway.stream(messages=_HI, model="m")
        [c async for c in sr]

        spend = attempt_spend()
        assert spend.cost_usd == 0.5
        assert spend.by_model[("first", "m")] == usage

    async def test_a_first_pull_without_usage_adds_no_empty_spend(self) -> None:
        enter_attempt(())
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [ProviderError("first", "boom")]))
        gateway.register(_Scripted("second", ["a"]), priority=1)

        sr = await gateway.stream(messages=_HI, model="m")
        [c async for c in sr]

        assert ("first", "m") not in attempt_spend().by_model

    async def test_the_time_to_first_token_is_measured_at_the_first_pull(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([10.0, 10.25])
        monkeypatch.setattr("agentloom.providers.gateway.anyio.current_time", lambda: next(clock))
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", ["a"]))

        sr = await gateway.stream(messages=_HI, model="m")

        assert sr.time_to_first_token_ms == 250.0


class TestStreamRateLimitAfterFirstChunk:
    async def test_a_429_mid_stream_does_not_count_against_the_breaker(self) -> None:
        observer = MagicMock()
        gateway = ProviderGateway()
        gateway.set_observer(observer)
        gateway.register(_Scripted("first", ["a", RateLimitError("first")]))
        cb = _half_open(gateway, 0)

        sr = await gateway.stream(messages=_HI, model="m", step_id="s1")
        with pytest.raises(RateLimitError):
            async for _ in sr:
                pass

        assert cb.failure_count == 0
        assert cb._half_open_calls == 0
        assert cb.state == CircuitState.HALF_OPEN
        assert observer.on_provider_call_end.call_args.kwargs["error"] == "RateLimitError"
        observer.on_provider_error.assert_not_called()


class TestCandidateOrder:
    @staticmethod
    def _names(gateway: ProviderGateway, model: str) -> list[str]:
        return [e.provider.name for e in gateway._get_candidates(model)]

    def test_a_listed_model_still_gets_the_fallback_providers(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("cloud", supports=False), models=["m"])
        gateway.register(_Scripted("local", supports=False), priority=100, is_fallback=True)

        assert self._names(gateway, "m") == ["cloud", "local"]

    async def test_a_listed_model_falls_back_when_its_provider_fails(self) -> None:
        local = _Scripted("local", ["from local"], supports=False)
        gateway = ProviderGateway()
        gateway.register(_Scripted("cloud", [ProviderError("cloud", "down")]), models=["m"])
        gateway.register(local, priority=100, is_fallback=True)

        response = await gateway.complete(messages=_HI, model="m")

        assert response.provider == "local"

    def test_the_primary_goes_first_whatever_its_priority(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("cheap"), priority=0)
        gateway.register(_Scripted("named"), priority=50, is_primary=True)

        assert self._names(gateway, "m") == ["named", "cheap"]

    def test_the_primary_goes_ahead_of_a_listed_model(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("listed", supports=False), priority=0, models=["m"])
        gateway.register(_Scripted("named"), priority=50, is_primary=True)

        assert self._names(gateway, "m") == ["named", "listed"]

    def test_a_primary_that_cannot_serve_the_name_is_left_out(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("named", supports=False), is_primary=True)
        gateway.register(_Scripted("serves"), priority=10)

        assert self._names(gateway, "m") == ["serves"]

    def test_a_primary_that_is_also_a_fallback_appears_once(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("cloud"), priority=10)
        gateway.register(_Scripted("local"), priority=100, is_fallback=True, is_primary=True)

        assert self._names(gateway, "m") == ["local", "cloud"]

    def test_the_others_keep_their_priority_after_the_primary(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("b"), priority=20)
        gateway.register(_Scripted("a"), priority=10)
        gateway.register(_Scripted("named"), priority=30, is_primary=True)

        assert self._names(gateway, "m") == ["named", "a", "b"]


class TestModelAliases:
    async def test_each_candidate_gets_its_own_id(self) -> None:
        first = _Scripted("first", [ProviderError("first", "down")])
        second = _Scripted("second", ["answer"])
        gateway = ProviderGateway()
        gateway.register(first, priority=0)
        gateway.register(second, priority=1, model_aliases={"gpt-4o": "claude-x"})

        response = await gateway.complete(messages=_HI, model="gpt-4o")

        assert first.models == ["gpt-4o"]
        assert second.models == ["claude-x"]
        assert response.model == "claude-x"

    async def test_a_streamed_call_translates_too(self) -> None:
        second = _Scripted("second", ["answer"])
        gateway = ProviderGateway()
        gateway.register(_Scripted("first", [ProviderError("first", "down")]), priority=0)
        gateway.register(second, priority=1, model_aliases={"gpt-4o": "claude-x"})

        sr = await gateway.stream(messages=_HI, model="gpt-4o")
        [c async for c in sr]

        assert second.models == ["claude-x"]

    async def test_embeddings_translate_too(self) -> None:
        provider = _Scripted("first")
        gateway = ProviderGateway()
        gateway.register(provider, model_aliases={"text-embedding-3-small": "nomic"})

        response = await gateway.embed(inputs=["a"], model="text-embedding-3-small")

        assert provider.models == ["nomic"]
        assert response.model == "nomic"

    def test_an_alias_makes_a_provider_a_candidate_for_a_name_it_does_not_serve(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("openai"), priority=0)
        gateway.register(
            _Scripted("anthropic", supports=False),
            priority=1,
            model_aliases={"gpt-4o": "claude-x"},
        )

        assert [e.provider.name for e in gateway._get_candidates("gpt-4o")] == [
            "openai",
            "anthropic",
        ]
        assert [e.provider.name for e in gateway._get_candidates("gpt-3.5")] == ["openai"]

    def test_an_alias_does_not_pin_the_name_away_from_the_others(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("listed", supports=False), priority=5, models=["m"])
        gateway.register(_Scripted("aliased", supports=False), priority=1, model_aliases={"m": "x"})

        assert [e.provider.name for e in gateway._get_candidates("m")] == ["aliased", "listed"]

    async def test_a_name_without_an_alias_goes_unchanged(self) -> None:
        provider = _Scripted("first")
        gateway = ProviderGateway()
        gateway.register(provider, model_aliases={"other": "x"})

        await gateway.complete(messages=_HI, model="m")

        assert provider.models == ["m"]


class _Hanging(MockProvider):
    """Never answers, so only a cancel ends the call."""

    async def complete(self, *args: Any, **kwargs: Any) -> ProviderResponse:
        await anyio.Event().wait()
        raise AssertionError("unreachable")

    async def embed(self, *args: Any, **kwargs: Any) -> EmbeddingResponse:
        await anyio.Event().wait()
        raise AssertionError("unreachable")


class _StallsAfterFirstChunk(MockProvider):
    async def stream(self, *args: Any, model: str, **kwargs: Any) -> StreamResponse:
        sr = StreamResponse(model=model, provider="mock")

        async def _generate() -> AsyncIterator[str]:
            yield "first"
            await anyio.Event().wait()

        sr._set_iterator(_generate())
        return sr


class TestACancelEndsTheAttempt:
    """A call cut by a timeout or a cancel still reports its end, so its span closes."""

    @pytest.mark.parametrize("call", ["complete", "embed"])
    async def test_a_call_in_flight(self, call: str) -> None:
        gateway = ProviderGateway()
        gateway.register(_Hanging(), priority=0)
        observer = MagicMock()
        gateway.set_observer(observer)

        with anyio.move_on_after(0.01):
            if call == "complete":
                await gateway.complete(
                    messages=[{"role": "user", "content": "hi"}], model="m", step_id="s1"
                )
            else:
                await gateway.embed(inputs=["hi"], model="m", step_id="s1")

        observer.on_provider_call_end.assert_called_once()
        end = observer.on_provider_call_end.call_args.kwargs
        assert end["step_id"] == "s1"
        assert end["error"] == anyio.get_cancelled_exc_class().__name__

    @pytest.mark.parametrize("call", ["complete", "embed"])
    async def test_a_half_open_circuit_gets_its_slot_back(self, call: str) -> None:
        gateway = ProviderGateway()
        gateway.register(_Hanging(), priority=0)
        breaker = gateway._providers[0].circuit_breaker
        breaker._set_state(CircuitState.HALF_OPEN)

        with anyio.move_on_after(0.01):
            if call == "complete":
                await gateway.complete(messages=[{"role": "user", "content": "hi"}], model="m")
            else:
                await gateway.embed(inputs=["hi"], model="m")

        assert breaker._half_open_calls == 0

    async def test_an_embed_refused_by_a_circuit_probing_elsewhere(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Hanging(), priority=0)
        breaker = gateway._providers[0].circuit_breaker
        breaker._set_state(CircuitState.HALF_OPEN)
        breaker._half_open_calls = breaker.half_open_max_calls
        observer = MagicMock()
        gateway.set_observer(observer)

        with pytest.raises(ProviderError, match="circuit is open"):
            await gateway.embed(inputs=["hi"], model="m", step_id="s1")

        observer.on_provider_call_end.assert_called_once()
        end = observer.on_provider_call_end.call_args.kwargs
        assert end["error"] == "CircuitOpenError"
        assert end["operation"] == "embeddings"

    async def test_a_stream_after_its_first_chunk(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_StallsAfterFirstChunk(), priority=0)
        observer = MagicMock()
        gateway.set_observer(observer)

        with anyio.move_on_after(0.01):
            sr = await gateway.stream(
                messages=[{"role": "user", "content": "hi"}], model="m", step_id="s1"
            )
            async for _ in sr:
                pass

        observer.on_provider_call_end.assert_called_once()
        end = observer.on_provider_call_end.call_args.kwargs
        assert end["error"] == anyio.get_cancelled_exc_class().__name__


class TestHooksThatRaise:
    async def test_the_error_hook_leaves_the_fallback_alone(self) -> None:
        gateway = ProviderGateway()
        gateway.register(FailingProvider("boom"), priority=0)
        answers = MockProvider()
        gateway.register(answers, priority=1)
        observer = MagicMock()
        observer.on_provider_error.side_effect = RuntimeError("hook bug")
        gateway.set_observer(observer)

        response = await gateway.complete(
            messages=[{"role": "user", "content": "hi"}], model="m", step_id="s1"
        )

        assert response.provider == "mock"
        assert len(answers.calls) == 1

    async def test_the_transition_hook_leaves_the_providers_error(self) -> None:
        gateway = ProviderGateway()
        observer = MagicMock()
        observer.on_circuit_state_change.side_effect = RuntimeError("hook bug")
        gateway.set_observer(observer)
        gateway.register(FailingProvider("boom"), priority=0, circuit_fail_threshold=1)

        with pytest.raises(ProviderError, match="boom"):
            await gateway.complete(messages=[{"role": "user", "content": "hi"}], model="m")

        assert gateway._providers[0].circuit_breaker.state == CircuitState.OPEN


class TestCircuitGaugeRefresh:
    async def test_each_provider_call_reports_its_circuit(self) -> None:
        gateway = ProviderGateway()
        gateway.register(MockProvider(), priority=0)
        observer = MagicMock()
        gateway.set_observer(observer)
        messages = [{"role": "user", "content": "hi"}]

        await gateway.complete(messages=messages, model="m")
        observer.on_circuit_state.assert_called_once_with("mock", "closed")

        await gateway.stream(messages=messages, model="m")
        assert observer.on_circuit_state.call_count == 2


class _Stale(_Scripted):
    """A replay that misses its recording, on every kind of call."""

    def __init__(self) -> None:
        super().__init__("mock", [RecordingMismatchError("drifted")])

    async def embed(self, inputs: list[str], model: str, **_kw: Any) -> EmbeddingResponse:
        raise RecordingMismatchError("drifted")


async def _stale_call(gateway: ProviderGateway, method: str) -> None:
    with pytest.raises(RecordingMismatchError):
        if method == "embed":
            await gateway.embed(inputs=["hi"], model="m")
        else:
            await getattr(gateway, method)(messages=_HI, model="m")


class TestAStaleRecordingIsNoProviderFault:
    @pytest.mark.parametrize("method", ["complete", "stream", "embed"])
    async def test_it_does_not_count_against_the_breaker(self, method: str) -> None:
        gateway = ProviderGateway()
        gateway.register(_Stale(), circuit_fail_threshold=2)
        cb = gateway._providers[0].circuit_breaker

        for _ in range(2):
            await _stale_call(gateway, method)

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.parametrize("method", ["complete", "stream", "embed"])
    async def test_it_gives_back_the_half_open_slot(self, method: str) -> None:
        gateway = ProviderGateway()
        gateway.register(_Stale())
        cb = _half_open(gateway)

        await _stale_call(gateway, method)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb._half_open_calls == 0


class TestARequestOverTheTokenLimit:
    _HUGE: Any = [{"role": "user", "content": "x" * 400}]

    async def test_a_stream_leaves_the_breaker_alone(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("p"), max_tpm=1, circuit_fail_threshold=2)
        cb = gateway._providers[0].circuit_breaker

        for _ in range(2):
            with pytest.raises(ProviderError, match="exceeds max_tokens_per_minute"):
                await gateway.stream(messages=self._HUGE, model="m")

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    async def test_a_stream_gives_back_the_half_open_slot(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted("p"), max_tpm=1)
        cb = _half_open(gateway)

        with pytest.raises(ProviderError, match="exceeds max_tokens_per_minute"):
            await gateway.stream(messages=self._HUGE, model="m")

        assert cb.state == CircuitState.HALF_OPEN
        assert cb._half_open_calls == 0
