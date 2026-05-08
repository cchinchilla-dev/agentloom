"""Anthropic provider adapter using httpx."""

from __future__ import annotations

import json
import logging
import os
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
from agentloom.providers.pricing import calculate_cost

logger = logging.getLogger("agentloom.providers.anthropic")

_ANTHROPIC_EXTRA_PAYLOAD_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
        "tools",
        "tool_choice",
        # Raw passthrough for advanced callers (full Anthropic shape).
        "thinking",
        # Provider-uniform high-level config from ``llm_call``; we translate
        # it here to the wire-format ``thinking`` payload.
        "thinking_config",
    }
)


def _translate_thinking_config(extras: dict[str, Any]) -> bool:
    """Convert a ``ThinkingConfig`` extra into Anthropic's ``thinking`` payload.

    Mutates ``extras`` in place and returns whether the caller wants the
    chain-of-thought trace exposed via ``ProviderResponse.reasoning_content``.
    Defaults to ``True`` so callers that don't pass a ``ThinkingConfig`` keep
    receiving thinking blocks when the provider returns them. If the caller
    already provided a raw ``thinking`` dict, it wins — we don't clobber the
    explicit override.
    """
    cfg = extras.pop("thinking_config", None)
    capture = True if cfg is None else bool(getattr(cfg, "capture_reasoning", True))
    if cfg is None or not getattr(cfg, "enabled", False):
        return capture
    if "thinking" in extras:
        return capture
    payload: dict[str, Any] = {"type": "enabled"}
    budget = getattr(cfg, "budget_tokens", None)
    if budget is not None:
        payload["budget_tokens"] = budget
    extras["thinking"] = payload
    return capture


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API adapter via httpx."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.anthropic.com/v1",
        **kwargs: Any,
    ) -> None:
        # Normalize base_url: SDK-style URLs (without /v1) need the suffix
        # for our direct httpx calls (e.g. ANTHROPIC_BASE_URL from Claude Desktop).
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @staticmethod
    def _format_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system prompt and convert content blocks to Anthropic format."""
        system_prompt: str | None = None
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content", "")
                system_prompt = content if isinstance(content, str) else str(content)
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "content": content})
            else:
                parts: list[dict[str, Any]] = []
                for block in content:
                    # Tool-calling messages build pure-dict wire-format
                    # blocks (``tool_use``, ``tool_result``); pass those
                    # through verbatim. Multimodal Pydantic blocks below need
                    # translation to Anthropic's specific keys.
                    if isinstance(block, dict):
                        parts.append(block)
                        continue
                    if isinstance(block, TextBlock):
                        parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageBlock):
                        parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.media_type,
                                    "data": block.data,
                                },
                            }
                        )
                    elif isinstance(block, ImageURLBlock):
                        parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": block.url,
                                },
                            }
                        )
                    elif isinstance(block, DocumentBlock):
                        parts.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.media_type,
                                    "data": block.data,
                                },
                            }
                        )
                    elif isinstance(block, AudioBlock):
                        raise ProviderError(
                            "anthropic",
                            "Anthropic does not support audio attachments.",
                        )
                formatted.append({"role": msg["role"], "content": parts})
        return system_prompt, formatted

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
        extras = validate_extra_kwargs(
            "anthropic", "complete", kwargs, _ANTHROPIC_EXTRA_PAYLOAD_KEYS
        )
        capture_reasoning = _translate_thinking_config(extras)
        if agentloom_tools:
            from agentloom.steps._tools import (
                translate_tool_choice_for_anthropic,
                translate_tools_for_anthropic,
            )

            extras["tools"] = translate_tools_for_anthropic(agentloom_tools)
            mapped_choice = translate_tool_choice_for_anthropic(agentloom_tool_choice or "auto")
            if mapped_choice is not None:
                extras["tool_choice"] = mapped_choice
        system_prompt, filtered_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or 4096,
            **extras,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = await self._client.post("/messages", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("anthropic", f"HTTP error: {e}") from e

        raise_for_status("anthropic", response)

        data = response.json()

        content_blocks = data.get("content", [])
        content = ""
        reasoning_parts: list[str] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                content += block.get("text", "")
            elif block_type == "thinking":
                # Extended-thinking trace. Captured whenever present; callers
                # that don't want it simply ignore ``reasoning_content``.
                reasoning_parts.append(block.get("thinking", "") or block.get("text", ""))

        usage_data = data.get("usage", {})
        # Anthropic rolls extended-thinking tokens into ``output_tokens``
        # rather than exposing a separate field — the official SDK's
        # ``Usage`` model has no ``thinking_tokens``/``reasoning_tokens``
        # entry, and the docs state explicitly that thinking tokens are
        # billed within ``output_tokens``. Cost is therefore correct
        # already; ``TokenUsage.reasoning_tokens`` stays 0 as a documented
        # limitation (same shape as Ollama).
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        usage = TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        from agentloom.steps._tools import parse_tool_calls_from_anthropic

        tool_calls = parse_tool_calls_from_anthropic(content_blocks)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            provider="anthropic",
            usage=usage,
            cost_usd=cost,
            reasoning_content=(
                "".join(reasoning_parts) if reasoning_parts and capture_reasoning else None
            ),
            tool_calls=tool_calls,
            raw_response=data,
            finish_reason=data.get("stop_reason"),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        extras = validate_extra_kwargs("anthropic", "stream", kwargs, _ANTHROPIC_EXTRA_PAYLOAD_KEYS)
        # Stream does not yet capture ``thinking`` deltas (separate work);
        # the return value is intentionally discarded here.
        _translate_thinking_config(extras)
        system_prompt, filtered_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or 4096,
            "stream": True,
            **extras,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature

        sr = StreamResponse(model=model, provider="anthropic")
        prompt_tokens = 0
        completion_tokens = 0

        async def _generate() -> AsyncIterator[str]:
            nonlocal prompt_tokens, completion_tokens
            async with self._client.stream("POST", "/messages", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status("anthropic", resp)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        logger.warning("Malformed SSE chunk, skipping: %s", line[:200])
                        continue
                    event_type = data.get("type", "")
                    if event_type == "message_start":
                        usage = data.get("message", {}).get("usage", {})
                        prompt_tokens = usage.get("input_tokens", 0)
                        sr.model = data.get("message", {}).get("model", model)
                    elif event_type == "content_block_delta":
                        text = data.get("delta", {}).get("text", "")
                        if text:
                            yield text
                    elif event_type == "message_delta":
                        delta = data.get("delta", {})
                        sr.finish_reason = delta.get("stop_reason")
                        usage = data.get("usage", {})
                        completion_tokens = usage.get("output_tokens", 0)
                        # Set usage immediately so the gateway wrapper sees
                        # populated values before StopAsyncIteration propagates.
                        sr.usage = TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        sr.cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

        sr._set_iterator(_generate())
        return sr

    def supports_model(self, model: str) -> bool:
        return "claude" in model

    async def close(self) -> None:
        await self._client.aclose()
