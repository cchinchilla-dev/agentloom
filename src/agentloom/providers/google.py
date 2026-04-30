"""Google Gemini provider adapter using httpx."""

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

logger = logging.getLogger("agentloom.providers.google")

# Keys that become entries under ``generationConfig``. Everything else is
# rejected so silent drops surface early. ``tools`` is top-level on Google's
# Generative Language API, not inside generationConfig — keep the two sets
# separate.
_GOOGLE_GEN_CONFIG_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "stop_sequences",
        "response_mime_type",
        "response_schema",
        "candidate_count",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    }
)
_GOOGLE_TOPLEVEL_KEYS = frozenset({"tools", "tool_config", "safety_settings"})
_GOOGLE_EXTRA_PAYLOAD_KEYS = _GOOGLE_GEN_CONFIG_KEYS | _GOOGLE_TOPLEVEL_KEYS


class GoogleProvider(BaseProvider):
    """Google Gemini API adapter via httpx."""

    name = "google"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0,
        )

    @staticmethod
    def _format_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Gemini contents + system instruction."""
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content", "")
                system_instruction = content if isinstance(content, str) else str(content)
                continue
            role = "user" if msg["role"] == "user" else "model"
            content = msg.get("content", "")
            if isinstance(content, str):
                contents.append({"role": role, "parts": [{"text": content}]})
            else:
                parts: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, TextBlock):
                        parts.append({"text": block.text})
                    elif isinstance(block, (ImageBlock, DocumentBlock, AudioBlock)):
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": block.media_type,
                                    "data": block.data,
                                },
                            }
                        )
                    elif isinstance(block, ImageURLBlock):
                        raise ProviderError(
                            "google",
                            "Google Gemini does not support URL passthrough for images. "
                            "Use fetch: local instead.",
                        )
                contents.append({"role": role, "parts": parts})
        return system_instruction, contents

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        extras = validate_extra_kwargs("google", "complete", kwargs, _GOOGLE_EXTRA_PAYLOAD_KEYS)
        system_instruction, contents = self._format_messages(messages)

        payload: dict[str, Any] = {"contents": contents}

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        for k in _GOOGLE_GEN_CONFIG_KEYS & extras.keys():
            generation_config[k] = extras[k]
        if generation_config:
            payload["generationConfig"] = generation_config
        for k in _GOOGLE_TOPLEVEL_KEYS & extras.keys():
            payload[k] = extras[k]

        url = f"/models/{model}:generateContent?key={self.api_key}"

        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("google", f"HTTP error: {e}") from e

        raise_for_status("google", response)

        data = response.json()

        candidates = data.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        usage_data = data.get("usageMetadata", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )
        cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        finish_reason = None
        if candidates:
            finish_reason = candidates[0].get("finishReason")

        return ProviderResponse(
            content=content,
            model=model,
            provider="google",
            usage=usage,
            cost_usd=cost,
            raw_response=data,
            finish_reason=finish_reason,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        extras = validate_extra_kwargs("google", "stream", kwargs, _GOOGLE_EXTRA_PAYLOAD_KEYS)
        system_instruction, contents = self._format_messages(messages)

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        for k in _GOOGLE_GEN_CONFIG_KEYS & extras.keys():
            generation_config[k] = extras[k]
        if generation_config:
            payload["generationConfig"] = generation_config
        for k in _GOOGLE_TOPLEVEL_KEYS & extras.keys():
            payload[k] = extras[k]

        url = f"/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        sr = StreamResponse(model=model, provider="google")

        async def _generate() -> AsyncIterator[str]:
            async with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status("google", resp)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        logger.warning("Malformed SSE chunk, skipping: %s", line[:200])
                        continue
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)
                        if text:
                            yield text
                        fr = candidates[0].get("finishReason")
                        if fr:
                            sr.finish_reason = fr
                    usage_data = data.get("usageMetadata")
                    if usage_data:
                        sr.usage = TokenUsage(
                            prompt_tokens=usage_data.get("promptTokenCount", 0),
                            completion_tokens=usage_data.get("candidatesTokenCount", 0),
                            total_tokens=usage_data.get("totalTokenCount", 0),
                        )
                        sr.cost_usd = calculate_cost(
                            model, sr.usage.prompt_tokens, sr.usage.completion_tokens
                        )

            if sr.usage.total_tokens == 0:
                # Gemini has historically streamed usage per-chunk, but a
                # regression or model variant could drop it. Surface the gap
                # so cost reporting doesn't silently read zero.
                logger.warning(
                    "Google stream for model '%s' completed without usageMetadata; "
                    "cost will be reported as 0.",
                    model,
                )

        sr._set_iterator(_generate())
        return sr

    def supports_model(self, model: str) -> bool:
        return "gemini" in model

    async def close(self) -> None:
        await self._client.aclose()
