"""E2E tests for the Ollama provider against a live Ollama instance.

These tests require a running Ollama server with the test model pulled.
They are excluded from normal test runs via the ``e2e`` marker.

To run locally::

    ollama pull qwen2.5:0.5b
    uv run pytest -m e2e -v
"""

from __future__ import annotations

import pytest

from agentloom.providers.base import ProviderResponse, StreamResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.ollama import OllamaProvider

pytestmark = pytest.mark.e2e


class TestOllamaComplete:
    """Non-streaming completion against a live Ollama instance."""

    async def test_basic_completion(
        self, ollama_provider: OllamaProvider, ollama_model: str
    ) -> None:
        result = await ollama_provider.complete(
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            model=ollama_model,
        )

        assert isinstance(result, ProviderResponse)
        assert len(result.content) > 0
        assert result.provider == "ollama"
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens > 0
        assert result.cost_usd == 0.0
        assert result.finish_reason == "stop"

    async def test_system_and_user_messages(
        self, ollama_provider: OllamaProvider, ollama_model: str
    ) -> None:
        result = await ollama_provider.complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer with just the number."},
            ],
            model=ollama_model,
        )

        assert isinstance(result, ProviderResponse)
        assert len(result.content) > 0

    async def test_options_max_tokens(
        self, ollama_provider: OllamaProvider, ollama_model: str
    ) -> None:
        result = await ollama_provider.complete(
            messages=[{"role": "user", "content": "Write a long story about a cat."}],
            model=ollama_model,
            temperature=0.1,
            max_tokens=10,
        )

        assert isinstance(result, ProviderResponse)
        # Ollama's num_predict is approximate; allow some margin
        assert result.usage.completion_tokens <= 20


class TestOllamaStream:
    """Streaming completion against a live Ollama instance."""

    async def test_streaming_completion(
        self, ollama_provider: OllamaProvider, ollama_model: str
    ) -> None:
        sr = await ollama_provider.stream(
            messages=[{"role": "user", "content": "Say hello."}],
            model=ollama_model,
        )

        assert isinstance(sr, StreamResponse)

        chunks: list[str] = []
        async for chunk in sr:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert len(sr.content) > 0
        assert sr.usage.prompt_tokens > 0
        assert sr.usage.completion_tokens > 0
        assert sr.finish_reason == "stop"
        assert sr.cost_usd == 0.0


class TestGatewayIntegration:
    """Ollama through the full ProviderGateway resilience layer."""

    async def test_gateway_routes_to_ollama(
        self, ollama_provider: OllamaProvider, ollama_model: str
    ) -> None:
        gateway = ProviderGateway()
        gateway.register(ollama_provider, priority=0)

        result = await gateway.complete(
            messages=[{"role": "user", "content": "Say yes."}],
            model=ollama_model,
        )

        assert isinstance(result, ProviderResponse)
        assert len(result.content) > 0
        assert result.provider == "ollama"
        await gateway.close()
