"""Tests for Ollama provider adapter."""

from __future__ import annotations

import json

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
    async def test_ollama_reasoning_tokens_zero_documented_limitation(self) -> None:
        # Ollama exposes a single ``eval_count`` for all output tokens
        # without splitting thinking vs visible. ``reasoning_tokens`` thus
        # always reports 0 — documented limitation. Cost is unaffected
        # (local models are free).
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}], model="phi4")
        assert result.usage.reasoning_tokens == 0
        assert result.usage.billable_completion_tokens == result.usage.completion_tokens
        assert result.reasoning_content is None
        await provider.close()

    @respx.mock
    async def test_ollama_message_thinking_populates_reasoning_content(self) -> None:
        # Ollama 0.9+ separates the trace into ``message.thinking`` when
        # the caller requested thinking via the ``think`` request param.
        # The adapter must surface it on ``reasoning_content`` with the
        # visible answer kept clean on ``content``.
        thinking_response = {
            "model": "deepseek-r1",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me work through this step by step...",
            },
            "prompt_eval_count": 15,
            "eval_count": 30,
            "done_reason": "stop",
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=thinking_response)
        )
        provider = OllamaProvider()
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="deepseek-r1"
        )
        assert result.content == "The answer is 42."
        assert result.reasoning_content == "Let me work through this step by step..."
        await provider.close()

    @respx.mock
    async def test_ollama_inline_think_tags_stripped_from_content(self) -> None:
        # Older models (or calls without ``think=true``) leak the trace
        # inline as ``<think>...</think>`` in ``message.content``. The
        # adapter strips the wrapper and surfaces the captured group on
        # ``reasoning_content``.
        legacy_response = {
            "model": "qwen3",
            "message": {
                "role": "assistant",
                "content": "<think>Working it out...</think>The answer is 42.",
            },
            "prompt_eval_count": 15,
            "eval_count": 30,
            "done_reason": "stop",
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=legacy_response)
        )
        provider = OllamaProvider()
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="qwen3"
        )
        assert result.content == "The answer is 42."
        assert result.reasoning_content == "Working it out..."
        await provider.close()

    @respx.mock
    async def test_ollama_think_param_passed_to_payload(self) -> None:
        # ``ThinkingConfig.enabled=True`` must surface as the top-level
        # ``think`` request param so Ollama 0.9+ activates thinking mode.
        # When ``level`` is set, it overrides the bool — GPT-OSS via
        # Ollama accepts ``"low"|"medium"|"high"`` directly.
        from agentloom.core.models import ThinkingConfig

        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider()
        cfg = ThinkingConfig(enabled=True, level="high")
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-oss",
            thinking_config=cfg,
        )
        sent = json.loads(route.calls.last.request.content)
        assert sent["think"] == "high"
        await provider.close()

    @respx.mock
    async def test_ollama_think_param_bool_when_no_level(self) -> None:
        # Without ``level``, the adapter sends ``think: true`` so reasoning
        # models (DeepSeek-R1, Qwen3) activate their default thinking mode.
        from agentloom.core.models import ThinkingConfig

        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )
        provider = OllamaProvider()
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-r1",
            thinking_config=ThinkingConfig(enabled=True),
        )
        sent = json.loads(route.calls.last.request.content)
        assert sent["think"] is True
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
    async def test_streaming_yields_chunks(self) -> None:
        ndjson = (
            '{"model":"phi4","message":{"role":"assistant","content":"Local"},"done":false}\n'
            '{"model":"phi4","message":{"role":"assistant","content":" response"},"done":false}\n'
            '{"model":"phi4","message":{"role":"assistant","content":""},"done":true,'
            '"done_reason":"stop","prompt_eval_count":15,"eval_count":10}\n'
        )
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, content=ndjson.encode())
        )
        provider = OllamaProvider()
        sr = await provider.stream(messages=[{"role": "user", "content": "hi"}], model="phi4")
        chunks = [chunk async for chunk in sr]
        assert chunks == ["Local", " response"]
        assert sr.content == "Local response"
        assert sr.usage.prompt_tokens == 15
        assert sr.usage.completion_tokens == 10
        assert sr.finish_reason == "stop"
        assert sr.cost_usd == 0.0
        await provider.close()

    @respx.mock
    async def test_streaming_api_error(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(404, text='{"error":"model not found"}')
        )
        provider = OllamaProvider()
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="nonexistent"
        )
        with pytest.raises(ProviderError, match="404"):
            async for _ in sr:
                pass
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


class TestOllamaBaseURLResolution:
    def test_base_url_from_env(self, monkeypatch) -> None:
        from agentloom.providers.ollama import OllamaProvider

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:9000")
        p = OllamaProvider()
        assert p.base_url == "http://ollama.internal:9000"

    def test_base_url_explicit_wins_over_env(self, monkeypatch) -> None:
        from agentloom.providers.ollama import OllamaProvider

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env.example")
        p = OllamaProvider(base_url="http://explicit.example")
        assert p.base_url == "http://explicit.example"

    def test_base_url_defaults_to_localhost(self, monkeypatch) -> None:
        from agentloom.providers.ollama import OllamaProvider

        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        p = OllamaProvider()
        assert p.base_url == "http://localhost:11434"

    @respx.mock
    async def test_streaming_options_built_from_temperature_max_tokens_extras(self) -> None:
        """Stream path must apply the same options/top-level layout as ``complete()``."""
        captured: dict[str, object] = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            ndjson = (
                '{"model":"phi4","message":{"role":"assistant","content":"x"},"done":false}\n'
                '{"model":"phi4","message":{"role":"assistant","content":""},"done":true,'
                '"done_reason":"stop","prompt_eval_count":1,"eval_count":1}\n'
            )
            return httpx.Response(200, content=ndjson.encode())

        respx.post("http://localhost:11434/api/chat").mock(side_effect=_capture)
        provider = OllamaProvider()
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="phi4",
            temperature=0.3,
            max_tokens=32,
            top_p=0.8,
            seed=99,
            format="json",
        )
        async for _ in sr:
            pass
        opts = captured["options"]
        assert opts["temperature"] == 0.3
        assert opts["num_predict"] == 32
        assert opts["top_p"] == 0.8
        assert opts["seed"] == 99
        assert captured["format"] == "json"
        await provider.close()

    @respx.mock
    async def test_complete_wraps_httpx_error_as_provider_error(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        provider = OllamaProvider()
        with pytest.raises(ProviderError, match="HTTP error"):
            await provider.complete(messages=[{"role": "user", "content": "x"}], model="phi4")
        await provider.close()

    @respx.mock
    async def test_options_built_from_temperature_max_tokens_extras(self) -> None:
        """Verify temperature, max_tokens (→num_predict), and allowlisted
        extras (top_p, seed) all land in the request ``options`` block, while
        ``format`` and ``tools`` sit at the top level."""
        captured: dict[str, object] = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json=MOCK_RESPONSE)

        respx.post("http://localhost:11434/api/chat").mock(side_effect=_capture)
        provider = OllamaProvider()
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="phi4",
            temperature=0.5,
            max_tokens=64,
            top_p=0.9,
            seed=7,
            format="json",
            tools=[{"name": "calc"}],
        )
        opts = captured["options"]
        assert opts["temperature"] == 0.5
        assert opts["num_predict"] == 64
        assert opts["top_p"] == 0.9
        assert opts["seed"] == 7
        assert captured["format"] == "json"
        assert captured["tools"] == [{"name": "calc"}]
