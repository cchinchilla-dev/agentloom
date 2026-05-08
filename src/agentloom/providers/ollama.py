"""Ollama provider adapter for local/LAN model inference."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError
from agentloom.providers._http import raise_for_status, validate_extra_kwargs
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.multimodal import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    ImageURLBlock,
    TextBlock,
)

logger = logging.getLogger("agentloom.providers.ollama")

# Inline ``<think>...</think>`` tags emitted by reasoning models on Ollama
# < 0.9 or when ``think`` is not requested at all. Capture group keeps the
# trace; substitution removes the wrapper from the visible answer.
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_inline_think_tags(text: str) -> tuple[str, str | None]:
    """Strip ``<think>...</think>`` from ``text``; return (clean, trace).

    Returns ``(text, None)`` unchanged when no tags are present so the
    common case stays a no-op.
    """
    matches = _THINK_TAG_RE.findall(text)
    if not matches:
        return text, None
    cleaned = _THINK_TAG_RE.sub("", text)
    return cleaned, "".join(matches)


def _pop_thinking_config(extras: dict[str, Any]) -> tuple[bool | str | None, bool]:
    """Pop ``thinking_config`` from ``extras`` and translate it.

    Returns ``(think_param, capture_reasoning)``:

    - ``think_param`` — the value to send under Ollama's top-level ``think``
      key, or ``None`` if reasoning was not requested. Levels override the
      bool form (GPT-OSS via Ollama accepts ``"low"|"medium"|"high"``).
    - ``capture_reasoning`` — whether to surface the trace via
      ``ProviderResponse.reasoning_content``. Defaults to ``True`` for
      callers that don't pass a ``ThinkingConfig`` so existing workflows
      keep their previous behaviour. When set to ``False``, inline
      ``<think>...</think>`` tags still get stripped from the visible
      ``content`` (kept clean) but the trace itself is dropped.
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
        """Convert internal content blocks to Ollama's images format."""
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            # Tool-loop messages (assistant with ``tool_calls``, role=``tool``
            # with ``tool_call_id``) ship in OpenAI-compatible wire shape —
            # pass them through verbatim so iteration 2+ keeps the call
            # context intact.
            if "tool_calls" in msg or msg.get("role") == "tool":
                formatted.append(msg)
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "content": content})
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
                entry: dict[str, Any] = {
                    "role": msg["role"],
                    "content": " ".join(text_parts),
                }
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
        extras = validate_extra_kwargs("ollama", "complete", kwargs, _OLLAMA_EXTRA_PAYLOAD_KEYS)
        think_param, capture_reasoning = _pop_thinking_config(extras)
        if agentloom_tools:
            from agentloom.steps._tools import translate_tools_for_ollama

            extras["tools"] = translate_tools_for_ollama(agentloom_tools)
            # Ollama doesn't expose tool_choice; only model-side support
            # decides whether the call is honored. Surface the silent drop
            # via debug log so users debugging "why doesn't my tool fire"
            # have a hint.
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
            raise ProviderError("ollama", f"HTTP error: {e}") from e

        raise_for_status("ollama", response)

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
        # Ollama 0.9+ separates the trace into ``message.thinking`` when
        # ``think`` was requested. Older models / un-flagged calls leak
        # the trace inline as ``<think>...</think>``. Strip the tags from
        # ``content`` regardless (the visible answer must always be clean);
        # surface the captured trace only when the caller opted in via
        # ``ThinkingConfig.capture_reasoning``.
        thinking_field: str | None = message.get("thinking") or None
        content, inline_trace = _split_inline_think_tags(content)
        reasoning_content = (thinking_field or inline_trace) if capture_reasoning else None

        # Ollama does not split eval_count between thinking and visible
        # tokens, so ``reasoning_tokens`` stays 0 — the docs flag this as
        # a known limitation. Cost is 0 anyway for local models.
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
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        extras = validate_extra_kwargs("ollama", "stream", kwargs, _OLLAMA_EXTRA_PAYLOAD_KEYS)
        think_param, capture_reasoning = _pop_thinking_config(extras)
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

        async def _generate() -> AsyncIterator[str]:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status("ollama", resp)
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
                        if thinking_buffer and capture_reasoning:
                            sr.reasoning_content = "".join(thinking_buffer)
                        break
                    message = data.get("message", {})
                    thinking_chunk = message.get("thinking")
                    if thinking_chunk and capture_reasoning:
                        # Buffer only when the caller opted in. When opted
                        # out we still drain the field implicitly via the
                        # parent loop without storing it.
                        thinking_buffer.append(thinking_chunk)
                    text = message.get("content", "")
                    if text:
                        yield text

        sr._set_iterator(_generate())
        return sr

    def supports_model(self, model: str) -> bool:
        # Ollama accepts any model name — it downloads on demand if not present.
        # Side effect: Ollama matches all models in gateway candidate lookup,
        # so priority must be set higher (numerically) than cloud providers.
        return True

    async def close(self) -> None:
        await self._client.aclose()
