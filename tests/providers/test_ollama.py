"""Tests for Ollama provider adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from agentloom.exceptions import ProviderError
from agentloom.providers.ollama import OllamaProvider

MOCK_RESPONSE = {
    "model": "phi4",
    "message": {"role": "assistant", "content": "Local response"},
    "prompt_eval_count": 15,
    "eval_count": 10,
    "done_reason": "stop",
}


class TestOllamaProvider:
    @respx.mock
    async def test_successful_completion(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}], model="phi4")
        assert result.content == "Local response"
        assert result.provider == "ollama"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 10
        assert result.cost_usd == 0.0
        await provider.close()

    @respx.mock
    async def test_api_error_raises(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(404, text='{"error":"model not found"}')
        )
        provider = OllamaProvider()
        with pytest.raises(ProviderError, match="404"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}], model="nonexistent"
            )
        await provider.close()

    @respx.mock
    async def test_options_passed(self) -> None:
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider()
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="phi4",
            temperature=0.7,
            max_tokens=50,
        )
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["options"]["temperature"] == 0.7
        assert body["options"]["num_predict"] == 50
        assert body["stream"] is False
        await provider.close()

    def test_supports_any_model(self) -> None:
        p = OllamaProvider()
        assert p.supports_model("phi4")
        assert p.supports_model("anything")
        assert p.supports_model("gpt-4o-mini")

    @respx.mock
    async def test_custom_base_url(self) -> None:
        respx.post("http://192.168.1.100:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider(base_url="http://192.168.1.100:11434")
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}], model="phi4")
        assert result.content == "Local response"
        await provider.close()
