"""Ollama provider adapter for local/LAN model inference."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError
from agentloom.providers._http import format_http_error, raise_for_status, validate_extra_kwargs
from agentloom.providers._meta import response_meta
from agentloom.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    ProviderResponse,
    StreamResponse,
)
from agentloom.providers.multimodal import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    ImageURLBlock,
    TextBlock,
)

logger = logging.getLogger("agentloom.providers.ollama")

# Inline tags from reasoning models on Ollama < 0.9, or when ``think`` is not requested.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tag_len(text: str, tag: str) -> int:
    """Length of the longest end of ``text`` that is a start of ``tag``."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class _InlineThinkFilter:
    """Split text fed in chunks into the visible reply and its inline ``<think>`` trace.

    A tag may arrive split across chunks, so the end of a chunk that could start one is held
    until the next chunk settles it. Whitespace after ``</think>`` is dropped, and a ``<think>``
    still open when the text ends counts as trace.
    """

    def __init__(self) -> None:
        self.trace: list[str] = []
        self._pending = ""
        self._inside = False
        self._after_close = False

    def feed(self, chunk: str) -> str:
        text, self._pending = self._pending + chunk, ""
        out: list[str] = []
        while text:
            tag = _THINK_CLOSE if self._inside else _THINK_OPEN
            at = text.find(tag)
            if at == -1:
                held = _partial_tag_len(text, tag)
                self._pending = text[len(text) - held :]
                self._emit(text[: len(text) - held], out)
                break
            self._emit(text[:at], out)
            text = text[at + len(tag) :]
            self._inside = not self._inside
            self._after_close = not self._inside
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        rest, self._pending = self._pending, ""
        self._emit(rest, out)
        return "".join(out)

    def _emit(self, text: str, out: list[str]) -> None:
        if self._inside:
            if text:
                self.trace.append(text)
            return
        if self._after_close:
            text = text.lstrip()
            if not text:
                return
            self._after_close = False
        out.append(text)


def _split_inline_think_tags(text: str) -> tuple[str, str | None]:
    """Strip ``<think>...</think>`` from ``text``; return (clean, trace)."""
    inline = _InlineThinkFilter()
    clean = inline.feed(text) + inline.flush()
    return clean, "".join(inline.trace) or None


def _pop_thinking_config(extras: dict[str, Any]) -> tuple[bool | str | None, bool]:
    """Pop ``thinking_config`` from ``extras`` and translate it.

    Returns ``(think_param, capture_reasoning)``:

    - ``think_param`` — the value to send under Ollama's top-level ``think``
      key, or ``None`` if reasoning was not requested. Levels override the
      bool form (GPT-OSS via Ollama accepts ``"low"|"medium"|"high"``).
    - ``capture_reasoning`` — whether to surface the trace via
      ``ProviderResponse.reasoning_content``; ``True`` for callers that
      pass no ``ThinkingConfig``. When ``False``, ``complete()`` still
      strips inline ``<think>...</think>`` tags from the visible
      ``content`` but drops the trace itself.
    """
    cfg = extras.pop("thinking_config", None)
    capture = True if cfg is None else bool(getattr(cfg, "capture_reasoning", True))
    if cfg is None or not getattr(cfg, "enabled", False):
        return None, capture
    level = getattr(cfg, "level", None)
    if level is not None:
        return str(level), capture
    return True, capture


# Keys mapped into Ollama's ``options`` bag (model-level generation params).
_OLLAMA_OPTION_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "stop",
        "seed",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    }
)
# Top-level Ollama request keys (outside ``options``).
_OLLAMA_TOPLEVEL_KEYS = frozenset({"format", "tools", "keep_alive"})
_OLLAMA_EXTRA_PAYLOAD_KEYS = _OLLAMA_OPTION_KEYS | _OLLAMA_TOPLEVEL_KEYS | {"thinking_config"}


class OllamaProvider(BaseProvider):
    """Ollama API adapter — for local or LAN-hosted models.

    Connects to an Ollama server on localhost or LAN, no API keys needed.
    """

    name = "ollama"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        resolved = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        super().__init__(api_key=api_key, base_url=resolved, **kwargs)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,  # Local models can be slow
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal content blocks to Ollama's images format.

        Ollama's chat API has no ``name`` field. Multi-agent conversation
        turns carry the speaker inline (``"[<name>] …"``) so the model
        still sees who's talking; the field is silently dropped otherwise.
        """
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            # Tool-loop messages already use the OpenAI-compatible wire shape: pass them through.
            if "tool_calls" in msg or msg.get("role") == "tool":
                formatted.append(msg)
                continue
            name = msg.get("name")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = f"[{name}] {content}" if name else content
                formatted.append({"role": msg["role"], "content": text})
            else:
                text_parts: list[str] = []
                images: list[str] = []
                for block in content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ImageBlock):
                        images.append(block.data)
                    elif isinstance(block, (DocumentBlock, AudioBlock)):
                        raise ProviderError(
                            "ollama",
                            f"Ollama does not support {block.type} attachments.",
                        )
                    elif isinstance(block, ImageURLBlock):
                        raise ProviderError(
                            "ollama",
                            "Ollama does not support URL passthrough for images. "
                            "Use fetch: local instead.",
                        )
                joined = " ".join(text_parts)
                if name:
                    joined = f"[{name}] {joined}" if joined else f"[{name}]"
                entry: dict[str, Any] = {"role": msg["role"], "content": joined}
                if images:
                    entry["images"] = images
                formatted.append(entry)
        return formatted

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        agentloom_tools = kwargs.pop("agentloom_tools", None)
        agentloom_tool_choice = kwargs.pop("agentloom_tool_choice", None)
        agentloom_response_schema = kwargs.pop("agentloom_response_schema", None)
        agentloom_step_id = kwargs.pop("agentloom_step_id", "") or ""
        extras = validate_extra_kwargs("ollama", "complete", kwargs, _OLLAMA_EXTRA_PAYLOAD_KEYS)
        think_param, capture_reasoning = _pop_thinking_config(extras)
        if agentloom_response_schema is not None:
            from agentloom.steps._structured import translate_for_ollama

            extras["format"] = translate_for_ollama(agentloom_response_schema, agentloom_step_id)
        if agentloom_tools:
            from agentloom.steps._tools import translate_tools_for_ollama

            extras["tools"] = translate_tools_for_ollama(agentloom_tools)
            # Ollama has no tool_choice: log the ignored value as a debugging hint.
            if agentloom_tool_choice not in (None, "auto"):
                logger.debug(
                    "Ollama ignores tool_choice=%r; only model-side support matters.",
                    agentloom_tool_choice,
                )
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": False,
        }
        if think_param is not None:
            payload["think"] = think_param
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        for k in _OLLAMA_OPTION_KEYS & extras.keys():
            options[k] = extras[k]
        if options:
            payload["options"] = options
        for k in _OLLAMA_TOPLEVEL_KEYS & extras.keys():
            payload[k] = extras[k]

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("ollama", format_http_error(e)) from e

        raise_for_status("ollama", response)

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
        # The trace comes in ``message.thinking`` (0.9+ with ``think``) or inline <think> tags,
        # stripped even when not captured.
        thinking_field: str | None = message.get("thinking") or None
        content, inline_trace = _split_inline_think_tags(content)
        reasoning_content = (thinking_field or inline_trace) if capture_reasoning else None

        # eval_count does not split out thinking, so ``reasoning_tokens`` stays 0.
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        from agentloom.steps._tools import parse_tool_calls_from_openai

        tool_calls = parse_tool_calls_from_openai(data.get("message", {}) or {})

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            provider="ollama",
            usage=usage,
            cost_usd=0.0,  # Local models are free
            reasoning_content=reasoning_content,
            raw_response=data,
            finish_reason=data.get("done_reason"),
            tool_calls=tool_calls,
            meta=response_meta(options, response.headers),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        # As in complete(): pop the agentloom_* kwargs, or validation rejects them.
        agentloom_tools = kwargs.pop("agentloom_tools", None)
        agentloom_tool_choice = kwargs.pop("agentloom_tool_choice", None)
        agentloom_response_schema = kwargs.pop("agentloom_response_schema", None)
        agentloom_step_id = kwargs.pop("agentloom_step_id", "") or ""
        extras = validate_extra_kwargs("ollama", "stream", kwargs, _OLLAMA_EXTRA_PAYLOAD_KEYS)
        think_param, capture_reasoning = _pop_thinking_config(extras)
        if agentloom_response_schema is not None:
            from agentloom.steps._structured import translate_for_ollama

            extras["format"] = translate_for_ollama(agentloom_response_schema, agentloom_step_id)
        if agentloom_tools:
            from agentloom.steps._tools import translate_tools_for_ollama

            extras["tools"] = translate_tools_for_ollama(agentloom_tools)
            if agentloom_tool_choice not in (None, "auto"):
                logger.debug(
                    "Ollama ignores tool_choice=%r; only model-side support matters.",
                    agentloom_tool_choice,
                )
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
        }
        if think_param is not None:
            payload["think"] = think_param
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        for k in _OLLAMA_OPTION_KEYS & extras.keys():
            options[k] = extras[k]
        if options:
            payload["options"] = options
        for k in _OLLAMA_TOPLEVEL_KEYS & extras.keys():
            payload[k] = extras[k]

        sr = StreamResponse(model=model, provider="ollama")
        thinking_buffer: list[str] = []
        inline = _InlineThinkFilter()

        async def _generate() -> AsyncIterator[str]:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status("ollama", resp)
                sr.meta = response_meta(options, resp.headers)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Malformed NDJSON chunk, skipping: %s", line[:200])
                        continue
                    if data.get("done"):
                        sr.finish_reason = data.get("done_reason")
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        completion_tokens = data.get("eval_count", 0)
                        sr.usage = TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        sr.model = data.get("model", model)
                        break
                    message = data.get("message", {})
                    thinking_chunk = message.get("thinking")
                    if thinking_chunk and capture_reasoning:
                        thinking_buffer.append(thinking_chunk)
                    text = inline.feed(message.get("content", ""))
                    if text:
                        yield text
                tail = inline.flush()
                trace = "".join(thinking_buffer) or "".join(inline.trace)
                if trace and capture_reasoning:
                    sr.reasoning_content = trace
                if tail:
                    yield tail

        sr._set_iterator(_generate())
        return sr

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        kwargs.pop("agentloom_step_id", None)
        # /api/embed (0.1.29+) takes a list; an older server fails loudly, no single-input fallback.
        payload: dict[str, Any] = {"model": model, "input": inputs}
        try:
            response = await self._client.post("/api/embed", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("ollama", format_http_error(e)) from e
        raise_for_status("ollama", response)
        data = response.json()
        vectors: list[list[float]] = data.get("embeddings") or []
        prompt_tokens = data.get("prompt_eval_count", 0)
        return EmbeddingResponse(
            embeddings=vectors,
            model=data.get("model", model),
            provider="ollama",
            usage=TokenUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
            cost_usd=0.0,
            raw_response=data,
        )

    def supports_model(self, model: str) -> bool:
        # Any name is accepted (a model Ollama lacks fails at request time), so it matches every
        # model in candidate lookup: give it a higher priority number than the cloud providers.
        return True

    async def close(self) -> None:
        await self._client.aclose()
