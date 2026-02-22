"""Tests for the ProviderGateway module."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from tests.conftest import MockProvider


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
