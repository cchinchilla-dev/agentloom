"""OpenAI provider adapter using httpx."""

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

logger = logging.getLogger("agentloom.providers.openai")

# Keys forwarded to the OpenAI HTTP payload in addition to model/messages/
# temperature/max_tokens. Unknown kwargs raise TypeError so silent parameter
# drops become loud failures.
_OPENAI_EXTRA_PAYLOAD_KEYS = frozenset(
    {
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "response_format",
        "logit_bias",
        "n",
        "user",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        # Accepted but ignored on the chat-completions endpoint — o-series
        # reasoning is implicit in the model name. Allowing the kwarg keeps
        # ``ThinkingConfig`` provider-uniform at the step layer.
        "thinking_config",
    }
)


class OpenAIProvider(BaseProvider):
    """OpenAI API adapter via httpx (no SDK dependency)."""

    name = "openai"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        # Normalize base_url: SDK-style URLs (without /v1) need the suffix.
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal content blocks to OpenAI's vision/audio format."""
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted.append({"role": msg["role"], "content": content})
            else:
                parts: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, TextBlock):
                        parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageBlock):
                        data_url = f"data:{block.media_type};base64,{block.data}"
                        parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    elif isinstance(block, ImageURLBlock):
                        parts.append({"type": "image_url", "image_url": {"url": block.url}})
                    elif isinstance(block, AudioBlock):
                        if "wav" in block.media_type:
                            fmt = "wav"
                        elif "mp3" in block.media_type or "mpeg" in block.media_type:
                            fmt = "mp3"
                        else:
                            raise ProviderError(
                                "openai",
                                f"OpenAI only supports WAV and MP3 audio, "
                                f"got '{block.media_type}'.",
                            )
                        parts.append(
                            {
                                "type": "input_audio",
                                "input_audio": {"data": block.data, "format": fmt},
                            }
                        )
                    elif isinstance(block, DocumentBlock):
                        raise ProviderError(
                            "openai",
                            "OpenAI does not support PDF attachments in chat completions.",
                        )
                formatted.append({"role": msg["role"], "content": parts})
        return formatted

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        # Tool definitions arrive as ``ToolDefinition`` Pydantic instances —
        # translate to OpenAI's wire format before forwarding so callers
        # don't have to know provider-specific shapes.
        agentloom_tools = kwargs.pop("agentloom_tools", None)
        agentloom_tool_choice = kwargs.pop("agentloom_tool_choice", None)
        extras = validate_extra_kwargs("openai", "complete", kwargs, _OPENAI_EXTRA_PAYLOAD_KEYS)
        # ``thinking_config`` is accepted at the step layer for YAML
        # uniformity but has no chat-completions equivalent — drop it
        # before splatting extras into the request body.
        extras.pop("thinking_config", None)
        if agentloom_tools:
            from agentloom.steps._tools import (
                translate_tool_choice_for_openai,
                translate_tools_for_openai,
            )

            extras["tools"] = translate_tools_for_openai(agentloom_tools)
            if agentloom_tool_choice is not None and agentloom_tool_choice != "none":
                extras["tool_choice"] = translate_tool_choice_for_openai(agentloom_tool_choice)
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            **extras,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("openai", f"HTTP error: {e}") from e

        raise_for_status("openai", response)

        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        usage_data = data.get("usage", {})
        # o-series returns reasoning tokens under completion_tokens_details;
        # ordinary gpt-* responses omit the field and resolve to 0.
        details = usage_data.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens", 0) or 0
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            reasoning_tokens=reasoning_tokens,
        )
        cost = calculate_cost(
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )

        from agentloom.steps._tools import parse_tool_calls_from_openai

        tool_calls = parse_tool_calls_from_openai(message)


        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            provider="openai",
            usage=usage,
            cost_usd=cost,
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason"),
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
        extras = validate_extra_kwargs("openai", "stream", kwargs, _OPENAI_EXTRA_PAYLOAD_KEYS)
        extras.pop("thinking_config", None)
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **extras,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        sr = StreamResponse(model=model, provider="openai")

        async def _generate() -> AsyncIterator[str]:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status("openai", resp)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Malformed SSE chunk, skipping: %s", data_str[:200])
                        continue
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            yield text
                        fr = choices[0].get("finish_reason")
                        if fr:
                            sr.finish_reason = fr
                    usage_data = data.get("usage")
                    if usage_data:
                        details = usage_data.get("completion_tokens_details") or {}
                        reasoning_tokens = details.get("reasoning_tokens", 0) or 0
                        sr.usage = TokenUsage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0),
                            reasoning_tokens=reasoning_tokens,
                        )
                        sr.cost_usd = calculate_cost(
                            model,
                            sr.usage.prompt_tokens,
                            sr.usage.completion_tokens,
                            reasoning_tokens=sr.usage.reasoning_tokens,
                        )

        sr._set_iterator(_generate())
        return sr

    # Prefixes this adapter is known to handle. Checked longest-first so a
    # more specific prefix wins over a generic one — avoids the old ``o3``
    # prefix also claiming ``o3-mini`` when both are registered upstream.
    _SUPPORTED_PREFIXES: tuple[str, ...] = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-4",
        "gpt-3.5",
        "o4-",
        "o3-",
        "o1-",
        "gpt-",
        "o4",
        "o3",
        "o1",
    )

    def supports_model(self, model: str) -> bool:
        return any(model.startswith(p) for p in self._SUPPORTED_PREFIXES)

    async def close(self) -> None:
        await self._client.aclose()
