"""Anthropic provider adapter using httpx."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError, UnsupportedRequestError
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
        # Provider-uniform config from ``llm_call``, translated to the ``thinking`` payload.
        "thinking_config",
    }
)


def _translate_thinking_config(extras: dict[str, Any]) -> bool:
    """Convert a ``ThinkingConfig`` extra into Anthropic's ``thinking`` payload.

    Mutates ``extras`` in place and returns whether the caller wants the
    chain-of-thought trace exposed via ``ProviderResponse.reasoning_content``.
    Defaults to ``True`` so callers that don't pass a ``ThinkingConfig`` keep
    receiving thinking blocks when the provider returns them. A raw
    ``thinking`` dict the caller already provided wins over the config.
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
        # ANTHROPIC_BASE_URL-style URLs omit /v1 (the SDK appends /v1/messages), so add it here.
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
        """Extract system prompt and convert content blocks to Anthropic format.

        Anthropic's Messages API has no native ``name`` field on turns;
        when a message carries one (multi-agent conversations), the
        speaker gets prepended to the content as ``"[<name>] …"`` so the
        model still sees who spoke instead of losing the attribution.
        System-turn names are ignored — there is no ambiguity there.
        """
        # Every system turn applies, in order: a speaker's prompt and a conversation summary both
        systems: list[str] = []
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content", "")
                systems.append(content if isinstance(content, str) else str(content))
                continue
            name = msg.get("name")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = f"[{name}] {content}" if name else content
                formatted.append({"role": msg["role"], "content": text})
            else:
                parts: list[dict[str, Any]] = []
                for block in content:
                    # Tool-loop blocks are already wire-format dicts: pass them through.
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
                        raise UnsupportedRequestError(
                            "anthropic",
                            "Anthropic does not support audio attachments.",
                        )
                if name:
                    parts.insert(0, {"type": "text", "text": f"[{name}]"})
                formatted.append({"role": msg["role"], "content": parts})
        return "\n\n".join(systems) or None, formatted

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
        extras = validate_extra_kwargs(
            "anthropic", "complete", kwargs, _ANTHROPIC_EXTRA_PAYLOAD_KEYS
        )
        capture_reasoning = _translate_thinking_config(extras)
        if agentloom_tools and agentloom_tool_choice != "none":
            # "none" leaves ``tools`` off the request, so the model cannot call them.
            from agentloom.steps._tools import (
                translate_tool_choice_for_anthropic,
                translate_tools_for_anthropic,
            )

            extras["tools"] = translate_tools_for_anthropic(agentloom_tools)
            mapped_choice = translate_tool_choice_for_anthropic(agentloom_tool_choice or "auto")
            if mapped_choice is not None:
                extras["tool_choice"] = mapped_choice
        system_prompt, filtered_messages = self._format_messages(messages)

        # No native ``response_format``: prefill ``"{"`` so the model continues the JSON.
        if agentloom_response_schema is not None:
            from agentloom.steps._structured import anthropic_system_prefix

            system_prompt = (system_prompt or "") + anthropic_system_prefix(
                agentloom_response_schema, agentloom_step_id
            )
            filtered_messages = [*filtered_messages, {"role": "assistant", "content": "{"}]

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
            raise ProviderError("anthropic", format_http_error(e)) from e

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
                # Extended-thinking trace, returned only when ``capture_reasoning`` is on.
                reasoning_parts.append(block.get("thinking", "") or block.get("text", ""))

        # Reattach the prefilled ``"{"`` — Anthropic only echoes the continuation.
        if agentloom_response_schema is not None and not content.lstrip().startswith("{"):
            content = "{" + content

        usage_data = data.get("usage", {})
        # Thinking bills inside ``output_tokens`` with no separate count: reasoning_tokens stays 0.
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
            meta=response_meta(
                payload, response.headers, request_id=response.headers.get("request-id")
            ),
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
        extras = validate_extra_kwargs("anthropic", "stream", kwargs, _ANTHROPIC_EXTRA_PAYLOAD_KEYS)
        # stream() ignores thinking deltas, so the capture flag goes unused.
        _translate_thinking_config(extras)
        if agentloom_tools and agentloom_tool_choice != "none":
            from agentloom.steps._tools import (
                translate_tool_choice_for_anthropic,
                translate_tools_for_anthropic,
            )

            extras["tools"] = translate_tools_for_anthropic(agentloom_tools)
            mapped_choice = translate_tool_choice_for_anthropic(agentloom_tool_choice or "auto")
            if mapped_choice is not None:
                extras["tool_choice"] = mapped_choice
        system_prompt, filtered_messages = self._format_messages(messages)

        if agentloom_response_schema is not None:
            from agentloom.steps._structured import anthropic_system_prefix

            system_prompt = (system_prompt or "") + anthropic_system_prefix(
                agentloom_response_schema, agentloom_step_id
            )
            filtered_messages = [*filtered_messages, {"role": "assistant", "content": "{"}]

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
        prefill_json = agentloom_response_schema is not None
        prompt_tokens = 0
        completion_tokens = 0

        async def _generate() -> AsyncIterator[str]:
            nonlocal prompt_tokens, completion_tokens
            first_text_chunk = True
            async with self._client.stream("POST", "/messages", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status("anthropic", resp)
                sr.meta = response_meta(
                    payload, resp.headers, request_id=resp.headers.get("request-id")
                )
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
                            if (
                                prefill_json
                                and first_text_chunk
                                and not text.lstrip().startswith("{")
                            ):
                                text = "{" + text
                            first_text_chunk = False
                            yield text
                    elif event_type == "message_delta":
                        delta = data.get("delta", {})
                        sr.finish_reason = delta.get("stop_reason")
                        usage = data.get("usage", {})
                        completion_tokens = usage.get("output_tokens", 0)
                        # Set now: the gateway wrapper reads usage as soon as the iterator ends.
                        sr.usage = TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        sr.cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

        sr._set_iterator(_generate())
        return sr

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        # No first-party embeddings API: the gateway skips to the next candidate on this error.
        raise NotImplementedError(
            "Anthropic does not offer a native embeddings API. Use OpenAI "
            "'text-embedding-3-*', Google 'gemini-embedding-001', a local "
            "Ollama embedding model, or Voyage AI (Anthropic's recommended "
            "third-party provider)."
        )

    def supports_model(self, model: str) -> bool:
        return "claude" in model

    async def close(self) -> None:
        await self._client.aclose()
