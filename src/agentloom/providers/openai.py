"""OpenAI provider adapter using httpx."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError, UnsupportedRequestError
from agentloom.providers._http import (
    format_http_error,
    json_body,
    raise_for_status,
    validate_extra_kwargs,
    wrap_stream_errors,
)
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

logger = logging.getLogger("agentloom.providers.openai")


def _normalize_base_url(raw: str) -> str:
    """Append ``/v1`` only when *raw* points at a bare host.

    Any non-root path is kept verbatim, so a gateway that serves another
    API version (``https://gw.example.com/v2``) or nests the version deeper
    (``https://gw.example.com/api/v1/foo``) keeps its URL; only a bare host
    (``https://x``, ``https://x/``) gets the suffix.
    """
    if not raw:
        return raw
    trimmed = raw.rstrip("/")
    parsed = urlparse(trimmed)
    if parsed.path and parsed.path != "/":
        return trimmed
    return trimmed + "/v1"


# Payload keys beyond model/messages/temperature/max_tokens; any other kwarg raises TypeError.
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
        # Provider-uniform config from ``llm_call``, translated to ``reasoning_effort``.
        "thinking_config",
    }
)


# pricing.yaml names the models but not which ones reason, so this goes by family.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_CHAT_ONLY_PREFIXES = ("gpt-5-chat",)  # in the family by name, but takes the ordinary request
_NO_EFFORT_PREFIXES = ("o1-mini",)  # reasons, and refuses ``reasoning_effort``


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES) and not model.startswith(_CHAT_ONLY_PREFIXES)


def _translate_thinking_config(extras: dict[str, Any], model: str) -> None:
    """Replace a ``ThinkingConfig`` extra with the ``reasoning_effort`` of its ``level``."""
    cfg = extras.pop("thinking_config", None)
    level = getattr(cfg, "level", None) if getattr(cfg, "enabled", False) else None
    takes_effort = _is_reasoning_model(model) and not model.startswith(_NO_EFFORT_PREFIXES)
    if level is not None and takes_effort:
        extras["reasoning_effort"] = level


def _add_sampling(
    payload: dict[str, Any], model: str, temperature: float | None, max_tokens: int | None
) -> None:
    """Add the limits a model takes: reasoning models refuse ``temperature`` and ``max_tokens``."""
    reasoning = _is_reasoning_model(model)
    if temperature is not None and not reasoning:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_completion_tokens" if reasoning else "max_tokens"] = max_tokens


def _usage(usage_data: dict[str, Any]) -> TokenUsage:
    """The usage of a response, its reasoning tokens apart from the visible ones.

    o-series and gpt-5 models report their reasoning under
    ``completion_tokens_details``, and count it in ``completion_tokens`` too;
    ``TokenUsage`` keeps the two apart, as cost adds them. Ordinary
    gpt-* responses omit the field, which resolves to 0.
    """
    details = usage_data.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens", 0) or 0
    return TokenUsage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=max(0, (usage_data.get("completion_tokens", 0) or 0) - reasoning),
        total_tokens=usage_data.get("total_tokens", 0),
        reasoning_tokens=reasoning,
    )


class OpenAIProvider(BaseProvider):
    """OpenAI API adapter via httpx (no SDK dependency)."""

    name = "openai"
    # Read when no ``api_key`` is passed; ``None`` reads no variable.
    _API_KEY_ENV: str | None = "OPENAI_API_KEY"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        base_url = _normalize_base_url(base_url)
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        env_key = os.environ.get(self._API_KEY_ENV, "") if self._API_KEY_ENV else ""
        self.api_key = api_key or env_key
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=60.0)

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal content blocks to OpenAI's vision/audio format.

        ``name`` on a message is forwarded verbatim — OpenAI's chat
        completions endpoint accepts ``name`` on any role for multi-agent
        threads and rejects only characters outside ``[A-Za-z0-9_-]``.
        Agent names are validated against that set at parse time; any other
        name the API rejects surfaces as a request-time 400 rather than
        being silently stripped here.
        """
        from agentloom.steps._tools import wire_seeded_tool_turns

        messages = wire_seeded_tool_turns("openai", messages)
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            # Tool-loop messages already use the OpenAI wire shape: pass them through verbatim.
            if "tool_calls" in msg or msg.get("role") == "tool":
                formatted.append(msg)
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                entry: dict[str, Any] = {"role": msg["role"], "content": content}
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
                            raise UnsupportedRequestError(
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
                        raise UnsupportedRequestError(
                            "openai",
                            "OpenAI does not support PDF attachments in chat completions.",
                        )
                entry = {"role": msg["role"], "content": parts}
            if msg.get("name"):
                entry["name"] = msg["name"]
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
        # Provider-neutral agentloom_* kwargs: popped before validation, translated below.
        agentloom_tools = kwargs.pop("agentloom_tools", None)
        agentloom_tool_choice = kwargs.pop("agentloom_tool_choice", None)
        agentloom_response_schema = kwargs.pop("agentloom_response_schema", None)
        agentloom_step_id = kwargs.pop("agentloom_step_id", "") or ""
        extras = validate_extra_kwargs(self.name, "complete", kwargs, _OPENAI_EXTRA_PAYLOAD_KEYS)
        _translate_thinking_config(extras, model)
        if agentloom_response_schema is not None:
            from agentloom.steps._structured import translate_for_openai

            extras["response_format"] = translate_for_openai(
                agentloom_response_schema, agentloom_step_id
            )
        if agentloom_tools:
            from agentloom.steps._tools import (
                translate_tool_choice_for_openai,
                translate_tools_for_openai,
            )

            extras["tools"] = translate_tools_for_openai(agentloom_tools)
            # Forward "none" explicitly: with ``tools`` set, OpenAI defaults to "auto".
            if agentloom_tool_choice is not None:
                extras["tool_choice"] = (
                    "none"
                    if agentloom_tool_choice == "none"
                    else translate_tool_choice_for_openai(agentloom_tool_choice)
                )
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            **extras,
        }
        _add_sampling(payload, model, temperature, max_tokens)

        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError(self.name, format_http_error(e)) from e

        raise_for_status(self.name, response)

        data = json_body(self.name, response)
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        usage = _usage(data.get("usage") or {})  # null from some compatible servers
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
            provider=self.name,
            usage=usage,
            cost_usd=cost,
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason"),
            tool_calls=tool_calls,
            meta=response_meta(
                payload,
                response.headers,
                request_id=response.headers.get("x-request-id"),
                system_fingerprint=data.get("system_fingerprint"),
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
        extras = validate_extra_kwargs(self.name, "stream", kwargs, _OPENAI_EXTRA_PAYLOAD_KEYS)
        _translate_thinking_config(extras, model)
        if agentloom_response_schema is not None:
            from agentloom.steps._structured import translate_for_openai

            extras["response_format"] = translate_for_openai(
                agentloom_response_schema, agentloom_step_id
            )
        if agentloom_tools:
            from agentloom.steps._tools import (
                translate_tool_choice_for_openai,
                translate_tools_for_openai,
            )

            extras["tools"] = translate_tools_for_openai(agentloom_tools)
            if agentloom_tool_choice is not None:
                extras["tool_choice"] = (
                    "none"
                    if agentloom_tool_choice == "none"
                    else translate_tool_choice_for_openai(agentloom_tool_choice)
                )
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **extras,
        }
        _add_sampling(payload, model, temperature, max_tokens)

        sr = StreamResponse(model=model, provider=self.name)

        async def _generate() -> AsyncIterator[str]:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise_for_status(self.name, resp)
                meta = response_meta(
                    payload, resp.headers, request_id=resp.headers.get("x-request-id")
                )
                sr.meta = meta
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
                    if data.get("system_fingerprint"):
                        meta.system_fingerprint = data["system_fingerprint"]
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
                        sr.usage = _usage(usage_data)
                        sr.cost_usd = calculate_cost(
                            model,
                            sr.usage.prompt_tokens,
                            sr.usage.completion_tokens,
                            reasoning_tokens=sr.usage.reasoning_tokens,
                        )

        sr._set_iterator(wrap_stream_errors(self.name, _generate()))
        return sr

    # Model-name prefixes this adapter serves; any match counts, so order does not matter.
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

    # Inputs per request; OpenAI accepts up to 2048, so this stays well below the cap.
    _EMBED_BATCH_SIZE: int = 100

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        kwargs.pop("agentloom_step_id", None)
        extras = validate_extra_kwargs(
            self.name, "embed", kwargs, frozenset({"encoding_format", "user"})
        )
        all_vectors: list[list[float]] = []
        prompt_tokens = 0
        total_tokens = 0
        cost = 0.0
        last_raw: dict[str, Any] = {}
        response_model = model
        for start in range(0, len(inputs), self._EMBED_BATCH_SIZE):
            chunk = inputs[start : start + self._EMBED_BATCH_SIZE]
            payload: dict[str, Any] = {"model": model, "input": chunk, **extras}
            if dimensions is not None:
                payload["dimensions"] = dimensions
            try:
                response = await self._client.post("/embeddings", json=payload)
            except httpx.HTTPError as e:
                raise ProviderError(self.name, format_http_error(e)) from e
            raise_for_status(self.name, response)
            data = json_body(self.name, response)
            # Sort by ``index`` so vectors stay aligned with inputs whatever order the API uses.
            for entry in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
                all_vectors.append(entry["embedding"])
            usage_data = data.get("usage", {})
            prompt_tokens += usage_data.get("prompt_tokens", 0)
            total_tokens += usage_data.get("total_tokens", usage_data.get("prompt_tokens", 0))
            cost += calculate_cost(model, usage_data.get("prompt_tokens", 0), 0)
            last_raw = data
            response_model = data.get("model", model)
        return EmbeddingResponse(
            embeddings=all_vectors,
            model=response_model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
            cost_usd=cost,
            raw_response=last_raw,
        )

    def supports_model(self, model: str) -> bool:
        # Narrow match: Google's ``text-embedding-004`` must not route here.
        if model.startswith(("text-embedding-3-", "text-embedding-ada-")):
            return True
        return any(model.startswith(p) for p in self._SUPPORTED_PREFIXES)

    async def close(self) -> None:
        await self._client.aclose()
