"""Each adapter reports the sampling it sent and what came back besides the answer (#166)."""

from __future__ import annotations

from typing import Any

import httpx
import respx

from agentloom.providers.anthropic import AnthropicProvider
from agentloom.providers.base import BaseProvider
from agentloom.providers.google import GoogleProvider
from agentloom.providers.ollama import OllamaProvider
from agentloom.providers.openai import OpenAIProvider

ASK = [{"role": "user", "content": "hi"}]
OPENAI = "https://api.openai.com/v1/chat/completions"
ANTHROPIC = "https://api.anthropic.com/v1/messages"
GEMINI = r".*/models/gemini.*"
OLLAMA = "http://localhost:11434/api/chat"
# Headers that no adapter may copy into the metadata.
OTHER_HEADERS = {"set-cookie": "s=1", "x-trace": "t"}
OPENAI_REPLY = {
    "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_1",
}
GEMINI_REPLY = {
    "candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
    "modelVersion": "gemini-2.5-flash-001",
    "responseId": "resp_1",
}


def _keyed(provider: type[BaseProvider]) -> BaseProvider:
    """``provider`` with a placeholder key, so it never reads one from the environment."""
    return provider(api_key="YOUR_API_KEY_HERE")


def _sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode()


async def _meta(provider: BaseProvider, model: str, stream: bool, **kwargs: Any) -> dict[str, Any]:
    """The metadata of one call, as the recording keeps it, without the arrival time."""
    if stream:
        sr = await provider.stream(ASK, model, **kwargs)
        async for _ in sr:
            pass
        response = sr.to_provider_response()
    else:
        response = await provider.complete(ASK, model, **kwargs)
    await provider.close()
    assert response.meta is not None
    assert response.meta.received_at is not None
    assert response.meta.received_at.utcoffset() is not None
    return response.meta.model_dump(exclude_defaults=True, exclude={"received_at"})


@respx.mock
async def test_openai_adapter_exposes_system_fingerprint() -> None:
    headers = {
        "x-request-id": "req_1",
        "x-ratelimit-remaining-requests": "99",
        "retry-after": "1",
        **OTHER_HEADERS,
    }
    respx.post(OPENAI).mock(return_value=httpx.Response(200, json=OPENAI_REPLY, headers=headers))
    sampling = {"temperature": 0.2, "top_p": 0.9, "seed": 7}
    meta = await _meta(_keyed(OpenAIProvider), "gpt-4o-mini", False, **sampling)
    assert meta == {
        **sampling,
        "system_fingerprint": "fp_1",
        "request_id": "req_1",
        "rate_limit": {"x-ratelimit-remaining-requests": "99", "retry-after": "1"},
    }


@respx.mock
async def test_openai_stream_takes_the_fingerprint_from_its_chunks() -> None:
    body = _sse(
        '{"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"system_fingerprint":"fp_2"}',
        "[DONE]",
    )
    respx.post(OPENAI).mock(
        return_value=httpx.Response(200, content=body, headers={"x-request-id": "req_2"})
    )
    meta = await _meta(_keyed(OpenAIProvider), "gpt-4o-mini", True, seed=3)
    assert meta == {"seed": 3, "system_fingerprint": "fp_2", "request_id": "req_2"}


@respx.mock
async def test_anthropic_adapter_reports_its_request_id_and_rate_limits() -> None:
    reply = {"content": [{"type": "text", "text": "Hi"}], "model": "claude-haiku-4-5"}
    headers = {
        "request-id": "req_3",
        "anthropic-ratelimit-requests-remaining": "49",
        **OTHER_HEADERS,
    }
    respx.post(ANTHROPIC).mock(return_value=httpx.Response(200, json=reply, headers=headers))
    sampling = {"temperature": 0.3, "top_p": 0.8}
    meta = await _meta(_keyed(AnthropicProvider), "claude-haiku-4-5", False, **sampling)
    assert meta == {
        **sampling,
        "request_id": "req_3",
        "rate_limit": {"anthropic-ratelimit-requests-remaining": "49"},
    }


@respx.mock
async def test_anthropic_stream_reports_its_request_id() -> None:
    body = _sse('{"type":"content_block_delta","delta":{"text":"Hi"}}')
    respx.post(ANTHROPIC).mock(
        return_value=httpx.Response(200, content=body, headers={"request-id": "req_4"})
    )
    meta = await _meta(_keyed(AnthropicProvider), "claude-haiku-4-5", True)
    assert meta == {"request_id": "req_4"}


@respx.mock
async def test_gemini_reports_its_model_version_and_response_id() -> None:
    respx.post(url__regex=GEMINI).mock(return_value=httpx.Response(200, json=GEMINI_REPLY))
    sampling = {"temperature": 0.4, "top_p": 0.7, "seed": 5}
    meta = await _meta(_keyed(GoogleProvider), "gemini-2.5-flash", False, **sampling)
    assert meta == {**sampling, "model_version": "gemini-2.5-flash-001", "request_id": "resp_1"}


@respx.mock
async def test_gemini_stream_takes_them_from_its_chunks() -> None:
    chunk = '{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"modelVersion":"v1",'
    chunk += '"responseId":"r1"}'
    respx.post(url__regex=GEMINI).mock(return_value=httpx.Response(200, content=_sse(chunk)))
    meta = await _meta(_keyed(GoogleProvider), "gemini-2.5-flash", True, top_p=0.5)
    assert meta == {"top_p": 0.5, "model_version": "v1", "request_id": "r1"}


@respx.mock
async def test_ollama_reports_the_sampling_options_it_sent() -> None:
    reply = {"message": {"content": "Hi"}, "model": "llama3.1:8b", "done": True}
    respx.post(OLLAMA).mock(return_value=httpx.Response(200, json=reply))
    sampling = {"temperature": 0.5, "top_p": 0.6, "seed": 4}
    meta = await _meta(OllamaProvider(), "llama3.1:8b", False, **sampling)
    assert meta == sampling


@respx.mock
async def test_ollama_stream_reports_them_too() -> None:
    body = b'{"message":{"content":"Hi"},"done":false}\n{"done":true,"done_reason":"stop"}\n'
    respx.post(OLLAMA).mock(return_value=httpx.Response(200, content=body))
    meta = await _meta(OllamaProvider(), "llama3.1:8b", True, seed=9)
    assert meta == {"seed": 9}
