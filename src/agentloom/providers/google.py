"""Google Gemini provider adapter using httpx."""

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
                        parts.append({
                            "inline_data": {
                                "mime_type": block.media_type,
                                "data": block.data,
                            },
                        })
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
        system_instruction, contents = self._format_messages(messages)

        payload: dict[str, Any] = {"contents": contents}

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        url = f"/models/{model}:generateContent?key={self.api_key}"

        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("google", f"HTTP error: {e}") from e

        if response.status_code != 200:
            raise ProviderError(
                "google",
                f"API error {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()

        # Extract content
        candidates = data.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        # Extract usage
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

    def supports_model(self, model: str) -> bool:
        return "gemini" in model

    async def close(self) -> None:
        await self._client.aclose()
