"""Tests for OpenAI provider adapter."""

from __future__ import annotations

import json

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
        # 500 — generic server error stays a plain ProviderError with the
        # status code encoded in the message. 429 gets its own dedicated
        # test below that asserts RateLimitError.
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="server error")
        )
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(ProviderError, match="500"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
            )
        await provider.close()

    @respx.mock
    async def test_429_raises_rate_limit_error(self) -> None:
        from agentloom.exceptions import RateLimitError

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
        )
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(RateLimitError) as excinfo:
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
            )
        assert excinfo.value.retry_after_s == 2.0
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
            provider._format_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            TextBlock(text="Summarize"),
                            DocumentBlock(data="abc", media_type="application/pdf"),
                        ],
                    }
                ]
            )
        await provider.close()

    def test_base_url_normalization(self) -> None:
        p = OpenAIProvider(api_key="k", base_url="https://api.openai.com")
        assert p.base_url == "https://api.openai.com/v1"

    @respx.mock
    async def test_streaming_yields_chunks(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"},'
            '"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,'
            '"completion_tokens":2,"total_tokens":12}}\n\n',
            "data: [DONE]\n\n",
        ]
        sse = "".join(lines)
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse.encode())
        )
        provider = OpenAIProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
        )
        chunks = [chunk async for chunk in sr]
        assert chunks == ["Hello", " world"]
        assert sr.content == "Hello world"
        assert sr.usage.total_tokens == 12
        assert sr.usage.prompt_tokens == 10
        assert sr.finish_reason == "stop"
        assert sr.cost_usd > 0
        await provider.close()

    @respx.mock
    async def test_streaming_sends_stream_options(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, content="".join(lines).encode())
        )
        provider = OpenAIProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        async for _ in sr:
            pass
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        await provider.close()

    @respx.mock
    async def test_streaming_api_error(self) -> None:
        from agentloom.exceptions import RateLimitError

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, text="rate limited")
        )
        provider = OpenAIProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini"
        )
        with pytest.raises(RateLimitError):
            async for _ in sr:
                pass
        await provider.close()

    def test_supports_gpt_models(self) -> None:
        p = OpenAIProvider(api_key="k")
        assert p.supports_model("gpt-4o-mini")
        assert p.supports_model("gpt-4.1")
        assert p.supports_model("o3")
        assert p.supports_model("o4-mini")
        assert not p.supports_model("claude-opus-4-6")
        assert not p.supports_model("gemini-2.5-flash")


class TestOpenAIKwargsAllowlist:
    """Extra kwargs must flow into the HTTP payload or raise TypeError."""

    @respx.mock
    async def test_complete_forwards_top_p(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OpenAIProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            top_p=0.7,
            stop=["END"],
        )
        body = json.loads(route.calls[0].request.content)
        assert body["top_p"] == 0.7
        assert body["stop"] == ["END"]
        await provider.close()

    async def test_complete_rejects_unknown_kwarg(self) -> None:
        provider = OpenAIProvider(api_key="k")
        with pytest.raises(TypeError, match="Unsupported parameters"):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
                not_a_real_param=True,
            )
        await provider.close()
