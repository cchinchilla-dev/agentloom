"""Ollama provider adapter for local/LAN model inference."""

from __future__ import annotations

from typing import Any

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse


class OllamaProvider(BaseProvider):
    """Ollama API adapter — for local or LAN-hosted models.

    Ideal for the Luckfox board: point to an Ollama server on the LAN
    to avoid needing API keys or internet access for inference.
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

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
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
        # Ollama can potentially serve any model
        return True

    async def close(self) -> None:
        await self._client.aclose()
