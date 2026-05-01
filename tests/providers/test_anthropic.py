"""Tests for Anthropic provider adapter."""

from __future__ import annotations

import json

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

    @respx.mock
    async def test_streaming_yields_chunks(self) -> None:
        lines = [
            'data: {"type":"message_start","message":'
            '{"id":"msg_1","model":"claude-haiku-4-5-20251001",'
            '"usage":{"input_tokens":12,"output_tokens":0}}}\n\n',
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hi"}}\n\n',
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":" there"}}\n\n',
            'data: {"type":"message_delta",'
            '"delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":8}}\n\n',
            'data: {"type":"message_stop"}\n\n',
        ]
        sse = "".join(lines)
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse.encode())
        )
        provider = AnthropicProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="claude-haiku-4-5-20251001",
        )
        chunks = [chunk async for chunk in sr]
        assert chunks == ["Hi", " there"]
        assert sr.content == "Hi there"
        assert sr.usage.prompt_tokens == 12
        assert sr.usage.completion_tokens == 8
        assert sr.finish_reason == "end_turn"
        assert sr.cost_usd > 0
        await provider.close()

    @respx.mock
    async def test_streaming_api_error(self) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500, text="internal error")
        )
        provider = AnthropicProvider(api_key="test-key")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-haiku-4-5-20251001",
        )
        with pytest.raises(ProviderError, match="500"):
            async for _ in sr:
                pass
        await provider.close()

    def test_base_url_normalization(self) -> None:
        p = AnthropicProvider(api_key="k", base_url="https://api.anthropic.com")
        assert p.base_url == "https://api.anthropic.com/v1"

    def test_supports_claude_models(self) -> None:
        p = AnthropicProvider(api_key="k")
        assert p.supports_model("claude-haiku-4-5-20251001")
        assert p.supports_model("claude-opus-4-6")
        assert not p.supports_model("gpt-4o-mini")


class TestAnthropicReasoning:
    @respx.mock
    async def test_thinking_response_does_not_split_reasoning_tokens(self) -> None:
        # Anthropic rolls extended-thinking tokens into ``output_tokens``
        # rather than exposing a separate field, so ``reasoning_tokens``
        # must stay 0 (documented limitation, same as Ollama). Cost is
        # still correct because the rate is applied to ``output_tokens``
        # which already includes the thinking volume.
        body = {
            "content": [
                {"type": "thinking", "thinking": "Let me think step by step..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "model": "claude-opus-4",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 208},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=body)
        )
        provider = AnthropicProvider(api_key="k")
        r = await provider.complete(
            messages=[{"role": "user", "content": "solve"}], model="claude-opus-4"
        )
        assert r.usage.reasoning_tokens == 0
        assert r.usage.completion_tokens == 208  # includes thinking volume
        assert r.usage.billable_completion_tokens == 208
        assert r.content == "The answer is 42."
        assert r.reasoning_content == "Let me think step by step..."
        await provider.close()

    @respx.mock
    async def test_capture_reasoning_writes_thinking_blocks(self) -> None:
        # Multiple thinking blocks must be concatenated into reasoning_content
        # in document order — each block's `thinking` field carries the trace.
        body = {
            "content": [
                {"type": "thinking", "thinking": "First, recall the formula. "},
                {"type": "thinking", "thinking": "Now substitute the numbers. "},
                {"type": "text", "text": "The answer is 42."},
            ],
            "model": "claude-opus-4",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 54},
        }
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=body)
        )
        provider = AnthropicProvider(api_key="k")
        r = await provider.complete(
            messages=[{"role": "user", "content": "solve"}], model="claude-opus-4"
        )
        assert r.reasoning_content == ("First, recall the formula. Now substitute the numbers. ")
        assert r.content == "The answer is 42."
        await provider.close()

    @respx.mock
    async def test_thinking_config_translated_to_payload(self) -> None:
        # ``ThinkingConfig`` must be translated by the adapter into
        # Anthropic's wire-format ``thinking`` payload.
        from agentloom.core.models import ThinkingConfig

        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = AnthropicProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "solve"}],
            model="claude-opus-4",
            thinking_config=ThinkingConfig(enabled=True, budget_tokens=2048),
        )
        assert route.called
        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }
        await provider.close()

    @respx.mock
    async def test_raw_thinking_kwarg_still_passes_through(self) -> None:
        # Power users that already build the wire-format ``thinking`` dict
        # themselves must keep working — ``thinking`` stays on the
        # allowlist as a passthrough and wins over ``thinking_config``.
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = AnthropicProvider(api_key="k")
        await provider.complete(
            messages=[{"role": "user", "content": "solve"}],
            model="claude-opus-4",
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["thinking"] == {
            "type": "enabled",
            "budget_tokens": 4096,
        }
        await provider.close()

    @respx.mock
    async def test_stream_translates_thinking_config_to_payload(self) -> None:
        # The stream path must exercise the same ``_translate_thinking_config``
        # helper as ``complete``: ``ThinkingConfig`` arriving via
        # ``thinking_config`` is rewritten to the wire-format ``thinking``
        # payload before the request leaves the adapter.
        from agentloom.core.models import ThinkingConfig

        # Minimal SSE payload — message_start + message_delta with usage.
        sse = (
            'data: {"type":"message_start","message":{"model":"claude-opus-4",'
            '"usage":{"input_tokens":3}}}\n\n'
            'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":1}}\n\n'
        )
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse.encode())
        )
        provider = AnthropicProvider(api_key="k")
        sr = await provider.stream(
            messages=[{"role": "user", "content": "solve"}],
            model="claude-opus-4",
            thinking_config=ThinkingConfig(enabled=True, budget_tokens=1024),
        )
        async for _ in sr:
            pass
        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["thinking"] == {
            "type": "enabled",
            "budget_tokens": 1024,
        }
        # ``thinking_config`` must not survive into the wire payload.
        assert "thinking_config" not in sent_payload
        await provider.close()

    def test_translate_thinking_config_respects_existing_thinking(self) -> None:
        # When the caller already provided a raw ``thinking`` dict alongside
        # ``thinking_config``, the raw form wins — the helper must not
        # clobber it.
        from agentloom.core.models import ThinkingConfig
        from agentloom.providers.anthropic import _translate_thinking_config

        extras: dict = {
            "thinking": {"type": "enabled", "budget_tokens": 9999},
            "thinking_config": ThinkingConfig(enabled=True, budget_tokens=128),
        }
        _translate_thinking_config(extras)
        assert extras["thinking"] == {"type": "enabled", "budget_tokens": 9999}
        assert "thinking_config" not in extras
