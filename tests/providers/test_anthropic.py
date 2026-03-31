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

    @respx.mock
    async def test_multimodal_image_block_formatting(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import ImageBlock, TextBlock

        provider = AnthropicProvider(api_key="test-key")
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
            model="claude-haiku-4-5-20251001",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Describe this image"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/jpeg"
        assert content[1]["source"]["data"] == "abc123"
        await provider.close()

    @respx.mock
    async def test_multimodal_url_passthrough(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import ImageURLBlock, TextBlock

        provider = AnthropicProvider(api_key="test-key")
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
            model="claude-haiku-4-5-20251001",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "url"
        assert content[1]["source"]["url"] == "https://example.com/img.jpg"
        await provider.close()

    @respx.mock
    async def test_multimodal_document_block_formatting(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import DocumentBlock, TextBlock

        provider = AnthropicProvider(api_key="test-key")
        await provider.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextBlock(text="Summarize this PDF"),
                        DocumentBlock(data="pdf_b64", media_type="application/pdf"),
                    ],
                }
            ],
            model="claude-haiku-4-5-20251001",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        content = body["messages"][0]["content"]
        assert content[1]["type"] == "document"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "application/pdf"
        assert content[1]["source"]["data"] == "pdf_b64"
        await provider.close()

    async def test_audio_attachment_raises(self) -> None:
        from agentloom.providers.multimodal import AudioBlock, TextBlock

        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="does not support audio"):
            provider._format_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            TextBlock(text="Transcribe"),
                            AudioBlock(data="abc", media_type="audio/wav"),
                        ],
                    }
                ]
            )
        await provider.close()

    def test_base_url_normalization(self) -> None:
        p = AnthropicProvider(api_key="k", base_url="https://api.anthropic.com")
        assert p.base_url == "https://api.anthropic.com/v1"

    def test_supports_claude_models(self) -> None:
        p = AnthropicProvider(api_key="k")
        assert p.supports_model("claude-haiku-4-5-20251001")
        assert p.supports_model("claude-opus-4-6")
        assert not p.supports_model("gpt-4o-mini")
