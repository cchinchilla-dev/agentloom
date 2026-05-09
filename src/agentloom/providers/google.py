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
#
# We accept snake_case at the public API surface (consistent with the other
# adapters) and translate to the camelCase shape Gemini's REST endpoint
# expects when serializing into the request body.
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
# ``thinking_config`` is the provider-uniform reasoning kwarg from ``llm_call``;
# it doesn't follow the GEN_CONFIG / TOPLEVEL split because we translate it to
# ``generationConfig.thinkingConfig`` directly with a per-field rename.
_GOOGLE_EXTRA_PAYLOAD_KEYS = _GOOGLE_GEN_CONFIG_KEYS | _GOOGLE_TOPLEVEL_KEYS | {"thinking_config"}

# snake_case (public) → camelCase (Gemini wire format).
_GOOGLE_KEY_REMAP = {
    "top_p": "topP",
    "top_k": "topK",
    "stop_sequences": "stopSequences",
    "response_mime_type": "responseMimeType",
    "response_schema": "responseSchema",
    "candidate_count": "candidateCount",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "tool_config": "toolConfig",
    "safety_settings": "safetySettings",
    # ``seed`` and ``tools`` are already valid Gemini field names.
}


def _to_gemini_key(key: str) -> str:
    return _GOOGLE_KEY_REMAP.get(key, key)


def _build_thinking_config_payload(cfg: Any) -> dict[str, Any] | None:
    """Translate a ``ThinkingConfig`` into Gemini's ``thinkingConfig`` block.

    Returns ``None`` when the config is missing or disabled so the caller
    can skip insertion entirely. ``includeThoughts`` is taken from
    ``capture_reasoning`` — Gemini has historically returned thought
    summaries (not the full trace) and only when this flag is set.
    """
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    payload: dict[str, Any] = {}
    budget = getattr(cfg, "budget_tokens", None)
    if budget is not None:
        payload["thinkingBudget"] = budget
    level = getattr(cfg, "level", None)
    if level is not None:
        payload["thinkingLevel"] = level
    if getattr(cfg, "capture_reasoning", False):
        payload["includeThoughts"] = True
    return payload


def _parse_gemini_content_parts(parts: list[dict[str, Any]]) -> tuple[str, str]:
    """Split Gemini content parts into (visible answer, reasoning trace).

    Parts carrying ``thought=true`` are thought summaries surfaced when
    ``includeThoughts`` is enabled — concatenate them into the reasoning
    trace and keep them out of the visible answer. All other parts are
    treated as final-answer text.
    """
    answer_chunks: list[str] = []
    thought_chunks: list[str] = []
    for part in parts:
        text = part.get("text", "")
        if not text:
            continue
        if part.get("thought"):
            thought_chunks.append(text)
        else:
            answer_chunks.append(text)
    return "".join(answer_chunks), "".join(thought_chunks)


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
            # Tool-loop messages already in Gemini wire shape (role=``model``
            # with ``functionCall`` parts, role=``function`` with
            # ``functionResponse`` parts) — pass them through verbatim so
            # iteration 2+ preserves the call context. Identified by the
            # presence of ``parts`` instead of ``content``.
            if "parts" in msg:
                contents.append(msg)
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
        agentloom_tools = kwargs.pop("agentloom_tools", None)
        agentloom_tool_choice = kwargs.pop("agentloom_tool_choice", None)
        extras = validate_extra_kwargs("google", "complete", kwargs, _GOOGLE_EXTRA_PAYLOAD_KEYS)
        thinking_payload = _build_thinking_config_payload(extras.pop("thinking_config", None))
        if agentloom_tools:
            from agentloom.steps._tools import translate_tools_for_google

            extras["tools"] = translate_tools_for_google(agentloom_tools)
            # ``{"name": "fn"}`` selects a specific function via Gemini's
            # ANY mode + ``allowedFunctionNames``. Plain strings map to
            # AUTO / ANY / NONE; anything else falls back to AUTO.
            choice = agentloom_tool_choice or "auto"
            fn_config: dict[str, Any]
            if isinstance(choice, dict) and "name" in choice:
                fn_config = {"mode": "ANY", "allowedFunctionNames": [choice["name"]]}
            else:
                mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(
                    choice if isinstance(choice, str) else "auto", "AUTO"
                )
                fn_config = {"mode": mode}
            extras["tool_config"] = {"functionCallingConfig": fn_config}
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
            generation_config[_to_gemini_key(k)] = extras[k]
        if thinking_payload is not None:
            generation_config["thinkingConfig"] = thinking_payload
        if generation_config:
            payload["generationConfig"] = generation_config
        for k in _GOOGLE_TOPLEVEL_KEYS & extras.keys():
            payload[_to_gemini_key(k)] = extras[k]

        url = f"/models/{model}:generateContent?key={self.api_key}"

        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("google", f"HTTP error: {e}") from e

        raise_for_status("google", response)

        data = response.json()

        candidates = data.get("candidates", [])
        content = ""
        reasoning_content: str | None = None
        content_parts: list[dict[str, Any]] = []
        if candidates:
            content_parts = candidates[0].get("content", {}).get("parts", []) or []
            content, reasoning_trace = _parse_gemini_content_parts(content_parts)
            if reasoning_trace:
                reasoning_content = reasoning_trace

        usage_data = data.get("usageMetadata", {})
        # ``thoughtsTokenCount`` was added with Gemini 2.5 thinking and is
        # intermittently absent on ``gemini-3-flash-preview`` even when
        # thinking is enabled — default to 0 instead of raising.
        reasoning_tokens = usage_data.get("thoughtsTokenCount", 0) or 0
        prompt_tokens = usage_data.get("promptTokenCount", 0)
        candidates_tokens = usage_data.get("candidatesTokenCount", 0)
        total_tokens = usage_data.get("totalTokenCount", 0)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=candidates_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        cost = calculate_cost(
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )

        finish_reason = None
        if candidates:
            finish_reason = candidates[0].get("finishReason")

        from agentloom.steps._tools import parse_tool_calls_from_google

        tool_calls = parse_tool_calls_from_google(content_parts)

        return ProviderResponse(
            content=content,
            model=model,
            provider="google",
            usage=usage,
            cost_usd=cost,
            reasoning_content=reasoning_content,
            raw_response=data,
            finish_reason=finish_reason,
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
        extras = validate_extra_kwargs("google", "stream", kwargs, _GOOGLE_EXTRA_PAYLOAD_KEYS)
        thinking_payload = _build_thinking_config_payload(extras.pop("thinking_config", None))
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
            generation_config[_to_gemini_key(k)] = extras[k]
        if thinking_payload is not None:
            generation_config["thinkingConfig"] = thinking_payload
        if generation_config:
            payload["generationConfig"] = generation_config
        for k in _GOOGLE_TOPLEVEL_KEYS & extras.keys():
            payload[_to_gemini_key(k)] = extras[k]

        url = f"/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        sr = StreamResponse(model=model, provider="google")
        reasoning_buffer: list[str] = []

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
                        answer_chunk, thought_chunk = _parse_gemini_content_parts(parts)
                        if thought_chunk:
                            reasoning_buffer.append(thought_chunk)
                        if answer_chunk:
                            yield answer_chunk
                        fr = candidates[0].get("finishReason")
                        if fr:
                            sr.finish_reason = fr
                    usage_data = data.get("usageMetadata")
                    if usage_data:
                        reasoning_tokens = usage_data.get("thoughtsTokenCount", 0) or 0
                        sr.usage = TokenUsage(
                            prompt_tokens=usage_data.get("promptTokenCount", 0),
                            completion_tokens=usage_data.get("candidatesTokenCount", 0),
                            total_tokens=usage_data.get("totalTokenCount", 0),
                            reasoning_tokens=reasoning_tokens,
                        )
                        sr.cost_usd = calculate_cost(
                            model,
                            sr.usage.prompt_tokens,
                            sr.usage.completion_tokens,
                            reasoning_tokens=sr.usage.reasoning_tokens,
                        )
            if reasoning_buffer:
                sr.reasoning_content = "".join(reasoning_buffer)

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
