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
    async def test_multimodal_image_block_formatting(self) -> None:
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        from agentloom.providers.multimodal import ImageBlock, TextBlock

        provider = OllamaProvider()
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
            model="llava",
        )
        import json

        body = json.loads(route.calls[0].request.content)
        msg = body["messages"][0]
        assert msg["content"] == "Describe this image"
        assert msg["images"] == ["abc123"]
        await provider.close()

    async def test_multimodal_url_passthrough_raises(self) -> None:
        from agentloom.providers.multimodal import ImageURLBlock, TextBlock

        provider = OllamaProvider()
        with pytest.raises(ProviderError, match="does not support URL passthrough"):
            await provider.complete(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            TextBlock(text="Describe"),
                            ImageURLBlock(
                                url="https://example.com/img.jpg", media_type="image/jpeg"
                            ),
                        ],
                    }
                ],
                model="llava",
            )
        await provider.close()

    async def test_document_attachment_raises(self) -> None:
        from agentloom.providers.multimodal import DocumentBlock, TextBlock

        provider = OllamaProvider()
        with pytest.raises(ProviderError, match="does not support document"):
            provider._format_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            TextBlock(text="Read"),
                            DocumentBlock(data="abc", media_type="application/pdf"),
                        ],
                    }
                ]
            )
        await provider.close()

    async def test_audio_attachment_raises(self) -> None:
        from agentloom.providers.multimodal import AudioBlock, TextBlock

        provider = OllamaProvider()
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

    @respx.mock
    async def test_custom_base_url(self) -> None:
        respx.post("http://192.168.1.100:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider(base_url="http://192.168.1.100:11434")
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}], model="phi4")
        assert result.content == "Local response"
        await provider.close()
