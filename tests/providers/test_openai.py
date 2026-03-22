"""Tests for OpenAI provider adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from agentloom.exceptions import ProviderError
from agentloom.providers.openai import OpenAIProvider

MOCK_RESPONSE = {
    "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
    "model": "gpt-4o-mini",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class TestOpenAIProvider:
    @respx.mock
    async def test_successful_completion(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OpenAIProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        assert result.content == "Hello!"
        assert result.provider == "openai"
        assert result.usage.total_tokens == 15
        await provider.close()

    @respx.mock
    async def test_api_error_raises(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, text="rate limited")
        )
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="429"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
            )
        await provider.close()

    @respx.mock
    async def test_temperature_and_max_tokens_passed(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OpenAIProvider(api_key="test-key")
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=100,
        )
        payload = route.calls[0].request.content
        import json

        body = json.loads(payload)
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 100
        await provider.close()

    def test_supports_gpt_models(self) -> None:
        p = OpenAIProvider(api_key="k")
        assert p.supports_model("gpt-4o-mini")
        assert p.supports_model("gpt-4.1")
        assert p.supports_model("o3")
        assert p.supports_model("o4-mini")
        assert not p.supports_model("claude-opus-4-6")
        assert not p.supports_model("gemini-2.5-flash")
