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

    @respx.mock
    async def test_multimodal_image_block_formatting(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import ImageBlock, TextBlock

        provider = OpenAIProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Describe this image"),
                        ImageBlock(data="abc123", media_type="image/jpeg"),
                    ],
                }
            ],
            model="gpt-4o",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Describe this image"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
        await provider.close()

    @respx.mock
    async def test_multimodal_url_passthrough(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import ImageURLBlock, TextBlock

        provider = OpenAIProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Describe"),
                        ImageURLBlock(url="https://example.com/img.jpg", media_type="image/jpeg"),
                    ],
                }
            ],
            model="gpt-4o",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert content[1]["image_url"]["url"] == "https://example.com/img.jpg"
        await provider.close()

    async def test_pdf_attachment_raises(self) -> None:
        from agentloom.providers.multimodal import DocumentBlock, TextBlock

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="does not support PDF"):
            provider._format_messages([
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Summarize"),
                        DocumentBlock(data="abc", media_type="application/pdf"),
                    ],
                }
            ])
        await provider.close()

    def test_base_url_normalization(self) -> None:
        p = OpenAIProvider(api_key="k", base_url="https://api.openai.com")
        assert p.base_url == "https://api.openai.com/v1"

    def test_supports_gpt_models(self) -> None:
        p = OpenAIProvider(api_key="k")
        assert p.supports_model("gpt-4o-mini")
        assert p.supports_model("gpt-4.1")
        assert p.supports_model("o3")
        assert p.supports_model("o4-mini")
        assert not p.supports_model("claude-opus-4-6")
        assert not p.supports_model("gemini-2.5-flash")
