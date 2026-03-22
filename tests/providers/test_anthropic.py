"""Tests for Anthropic provider adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from agentloom.exceptions import ProviderError
from agentloom.providers.anthropic import AnthropicProvider

MOCK_RESPONSE = {
    "content": [{"type": "text", "text": "Hi there!"}],
    "model": "claude-haiku-4-5-20251001",
    "usage": {"input_tokens": 12, "output_tokens": 8},
    "stop_reason": "end_turn",
}


class TestAnthropicProvider:
    @respx.mock
    async def test_successful_completion(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = AnthropicProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="claude-haiku-4-5-20251001",
        )
        assert result.content == "Hi there!"
        assert result.provider == "anthropic"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 20
        await provider.close()

    @respx.mock
    async def test_system_message_extracted(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = AnthropicProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hi"},
            ],
            model="claude-haiku-4-5-20251001",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["system"] == "Be helpful"
        assert all(m["role"] != "system" for m in body["messages"])
        await provider.close()

    @respx.mock
    async def test_api_error_raises(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500, text="internal error")
        )
        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="500"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-haiku-4-5-20251001",
            )
        await provider.close()

    def test_supports_claude_models(self) -> None:
        p = AnthropicProvider(api_key="k")
        assert p.supports_model("claude-haiku-4-5-20251001")
        assert p.supports_model("claude-opus-4-6")
        assert not p.supports_model("gpt-4o-mini")
