"""Tests for Google Gemini provider adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from agentloom.exceptions import ProviderError
from agentloom.providers.google import GoogleProvider

MOCK_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Gemini says hi"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 8,
        "candidatesTokenCount": 4,
        "totalTokenCount": 12,
    },
}


class TestGoogleProvider:
    @respx.mock
    async def test_successful_completion(self) -> None:
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = GoogleProvider(api_key="test-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
        )
        assert result.content == "Gemini says hi"
        assert result.provider == "google"
        assert result.usage.total_tokens == 12
        await provider.close()

    @respx.mock
    async def test_system_instruction(self) -> None:
        route = respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = GoogleProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "hi"},
            ],
            model="gemini-2.5-flash",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        assert "systemInstruction" in body
        assert body["systemInstruction"]["parts"][0]["text"] == "Be concise"
        await provider.close()

    @respx.mock
    async def test_api_error_raises(self) -> None:
        respx.post(url__regex=r".*/models/gemini.*").mock(
            return_value=httpx.Response(404, text="model not found")
        )
        provider = GoogleProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="404"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="gemini-2.5-flash",
            )
        await provider.close()

    def test_supports_gemini_models(self) -> None:
        p = GoogleProvider(api_key="k")
        assert p.supports_model("gemini-2.5-flash")
        assert p.supports_model("gemini-3.1-pro")
        assert not p.supports_model("gpt-4o-mini")
