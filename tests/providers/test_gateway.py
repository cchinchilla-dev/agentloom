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
