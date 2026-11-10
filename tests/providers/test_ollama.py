"""Tests for Ollama provider adapter."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx
import pytest
import respx

from agentloom.exceptions import ProviderError, UnsupportedRequestError
from agentloom.providers.base import StreamResponse
from agentloom.providers.ollama import OllamaProvider

MOCK_RESPONSE = {
    "model": "phi4",
    "message": {"role": "assistant", "content": "Local response"},
    "prompt_eval_count": 15,
    "eval_count": 10,
    "done_reason": "stop",
}


async def _stream_pieces(pieces: list[str], **kwargs: Any) -> tuple[list[str], StreamResponse]:
    """Stream a reply whose ``content`` arrives as ``pieces``; return its chunks and response."""
    lines = [
        json.dumps(
            {"model": "qwen3", "message": {"role": "assistant", "content": p}, "done": False}
        )
        for p in pieces
    ]
    lines.append(json.dumps({"model": "qwen3", "message": {"content": ""}, "done": True}))
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, content="\n".join(lines).encode())
    )
    provider = OllamaProvider()
    sr = await provider.stream(
        messages=[{"role": "user", "content": "hi"}], model="qwen3", **kwargs
    )
    chunks = [chunk async for chunk in sr]
    await provider.close()
    return chunks, sr


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
        # Ollama has one ``eval_count`` for all output, thinking included: nothing to split.
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
        # With ``think`` requested, Ollama 0.9+ returns the trace in ``message.thinking``.
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
        # Without ``think=true``, or on older models, the trace comes inline in ``<think>`` tags.
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
        # ``level`` overrides the bool: GPT-OSS on Ollama takes ``"low"|"medium"|"high"`` as is.
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
        # A bare ``think: true`` starts the default thinking mode of DeepSeek-R1 or Qwen3.
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
    async def test_ollama_capture_reasoning_false_suppresses_trace(self) -> None:
        from agentloom.core.models import ThinkingConfig

        legacy_with_inline = {
            "model": "qwen3",
            "message": {
                "role": "assistant",
                "content": "<think>internal trace</think>The answer is 42.",
                "thinking": "explicit trace",
            },
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done_reason": "stop",
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=legacy_with_inline)
        )
        provider = OllamaProvider()
        result = await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="qwen3",
            thinking_config=ThinkingConfig(enabled=True, capture_reasoning=False),
        )
        assert result.reasoning_content is None
        assert result.content == "The answer is 42."  # tags still stripped
        await provider.close()

    @respx.mock
    async def test_ollama_stream_thinking_chunks_concat_into_reasoning_content(self) -> None:
        # The ``think`` request param must reach the wire on the streaming path too.
        from agentloom.core.models import ThinkingConfig

        ndjson = (
            '{"model":"deepseek-r1","message":{"role":"assistant","content":"",'
            '"thinking":"step 1 "},"done":false}\n'
            '{"model":"deepseek-r1","message":{"role":"assistant","content":"",'
            '"thinking":"step 2 "},"done":false}\n'
            '{"model":"deepseek-r1","message":{"role":"assistant","content":"42"},'
            '"done":false}\n'
            '{"model":"deepseek-r1","message":{"role":"assistant","content":""},'
            '"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":3}\n'
        )
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, content=ndjson.encode())
        )
        provider = OllamaProvider()
        sr = await provider.stream(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-r1",
            thinking_config=ThinkingConfig(enabled=True),
        )
        chunks: list[str] = []
        async for chunk in sr:
            chunks.append(chunk)
        sent = json.loads(route.calls.last.request.content)
        assert sent["think"] is True
        assert "".join(chunks) == "42"
        assert sr.reasoning_content == "step 1 step 2 "
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
        with pytest.raises(UnsupportedRequestError, match="does not support URL passthrough"):
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
        with pytest.raises(UnsupportedRequestError, match="does not support document"):
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
        with pytest.raises(UnsupportedRequestError, match="does not support audio"):
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
    @pytest.mark.parametrize(
        "pieces",
        [
            ["<think>trace</think>answer"],
            ["<think>tr", "ace</think>  ", "answer"],
            ["<thi", "nk>trace</th", "ink>answer"],
        ],
        ids=["one-chunk", "trace-split", "tags-split"],
    )
    async def test_streaming_drops_inline_think_tags(self, pieces: list[str]) -> None:
        chunks, sr = await _stream_pieces(pieces)
        assert "".join(chunks) == "answer"
        assert sr.reasoning_content == "trace"

    @respx.mock
    async def test_streaming_keeps_the_trace_out_unless_captured(self) -> None:
        from agentloom.core.models import ThinkingConfig

        chunks, sr = await _stream_pieces(
            ["<think>trace</think>answer"],
            thinking_config=ThinkingConfig(enabled=True, capture_reasoning=False),
        )
        assert "".join(chunks) == "answer"
        assert sr.reasoning_content is None

    @respx.mock
    async def test_streaming_counts_an_unclosed_think_as_trace(self) -> None:
        chunks, sr = await _stream_pieces(["<think>cut short by", " num_predict"])
        assert chunks == []
        assert sr.reasoning_content == "cut short by num_predict"

    @respx.mock
    async def test_streaming_yields_a_lone_angle_bracket(self) -> None:
        chunks, sr = await _stream_pieces(["1 <", " 2 and 3 <"])
        assert "".join(chunks) == "1 < 2 and 3 <"
        assert sr.reasoning_content is None

    @respx.mock
    async def test_complete_counts_an_unclosed_think_as_trace(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={**MOCK_RESPONSE, "message": {"role": "assistant", "content": "<think>cut"}},
            )
        )
        provider = OllamaProvider()
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}], model="phi4")
        assert result.content == ""
        assert result.reasoning_content == "cut"
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
        # The message leads with the class name: ``str(exc)`` can be empty (httpx ``ReadTimeout``).
        with pytest.raises(ProviderError, match="ConnectError"):
            await provider.complete(messages=[{"role": "user", "content": "x"}], model="phi4")
        await provider.close()

    @respx.mock
    async def test_options_built_from_temperature_max_tokens_extras(self) -> None:
        """Sampling and extras go in ``options``; ``format`` and ``tools`` stay top level."""
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


class TestEmbedDimensions:
    @respx.mock
    async def test_dimensions_are_sent(self) -> None:
        route = respx.post("http://localhost:11434/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        )
        provider = OllamaProvider()
        await provider.embed(inputs=["a"], model="nomic-embed-text", dimensions=2)
        assert json.loads(route.calls[0].request.content)["dimensions"] == 2
        await provider.close()

    @respx.mock
    async def test_without_dimensions_the_key_stays_off(self) -> None:
        route = respx.post("http://localhost:11434/api/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        )
        provider = OllamaProvider()
        await provider.embed(inputs=["a"], model="nomic-embed-text")
        assert "dimensions" not in json.loads(route.calls[0].request.content)
        await provider.close()


class TestWireFailures:
    _HI: ClassVar[list[dict[str, Any]]] = [{"role": "user", "content": "hi"}]

    @respx.mock
    async def test_a_200_that_is_not_json_is_a_provider_error(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="<html/>")
        )
        provider = OllamaProvider()
        with pytest.raises(ProviderError, match="not valid JSON") as exc:
            await provider.complete(self._HI, "phi4")
        assert exc.value.status_code is None
        await provider.close()

    @respx.mock
    async def test_embeddings_that_are_not_json_are_a_provider_error(self) -> None:
        respx.post("http://localhost:11434/api/embed").mock(
            return_value=httpx.Response(200, text="oops")
        )
        provider = OllamaProvider()
        with pytest.raises(ProviderError, match="not valid JSON"):
            await provider.embed(inputs=["a"], model="nomic-embed-text")
        await provider.close()

    @respx.mock
    async def test_a_network_error_in_a_stream_is_a_provider_error(self) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("refused")
        )
        provider = OllamaProvider()
        sr = await provider.stream(self._HI, "phi4")
        with pytest.raises(ProviderError, match="ConnectError") as exc:
            [c async for c in sr]
        assert exc.value.status_code is None
        await provider.close()
