"""Ollama provider adapter for local/LAN model inference."""

from __future__ import annotations

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


class OllamaProvider(BaseProvider):
    """Ollama API adapter — for local or LAN-hosted models.

    Connects to an Ollama server on localhost or LAN, no API keys needed.
    """

    name = "ollama"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,  # Local models can be slow
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal content blocks to Ollama's images format."""
        formatted: list[dict[str, Any]] = []
        for msg in messages:
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": False,
        }
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("ollama", f"HTTP error: {e}") from e

        if response.status_code != 200:
            raise ProviderError(
                "ollama",
                f"API error {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        content = data.get("message", {}).get("content", "")

        # Ollama provides token counts in eval_count and prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            provider="ollama",
            usage=usage,
            cost_usd=0.0,  # Local models are free
            raw_response=data,
            finish_reason=data.get("done_reason"),
        )

    def supports_model(self, model: str) -> bool:
        # Ollama accepts any model name — it downloads on demand if not present.
        # Side effect: Ollama matches all models in gateway candidate lookup,
        # so priority must be set higher (numerically) than cloud providers.
        return True

    async def close(self) -> None:
        await self._client.aclose()
