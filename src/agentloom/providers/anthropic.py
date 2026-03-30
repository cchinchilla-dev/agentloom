"""Anthropic provider adapter using httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.multimodal import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    ImageURLBlock,
    TextBlock,
)
from agentloom.providers.pricing import calculate_cost


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
                    if isinstance(block, TextBlock):
                        parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageBlock):
                        parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.media_type,
                                "data": block.data,
                            },
                        })
                    elif isinstance(block, ImageURLBlock):
                        parts.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": block.url,
                            },
                        })
                    elif isinstance(block, DocumentBlock):
                        parts.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": block.media_type,
                                "data": block.data,
                            },
                        })
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
        system_prompt, filtered_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = await self._client.post("/messages", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("anthropic", f"HTTP error: {e}") from e

        if response.status_code != 200:
            raise ProviderError(
                "anthropic",
                f"API error {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()

        # Extract content from content blocks
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)),
        )
        cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            provider="anthropic",
            usage=usage,
            cost_usd=cost,
            raw_response=data,
            finish_reason=data.get("stop_reason"),
        )

    def supports_model(self, model: str) -> bool:
        return "claude" in model

    async def close(self) -> None:
        await self._client.aclose()
