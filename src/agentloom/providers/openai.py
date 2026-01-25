"""OpenAI provider adapter using httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.pricing import calculate_cost


class OpenAIProvider(BaseProvider):
    """OpenAI API adapter via httpx (no SDK dependency)."""

    name = "openai"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
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
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError("openai", f"HTTP error: {e}") from e

        if response.status_code != 200:
            raise ProviderError(
                "openai",
                f"API error {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        return ProviderResponse(
            content=content,
            model=data.get("model", model),
            provider="openai",
            usage=usage,
            cost_usd=cost,
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    def supports_model(self, model: str) -> bool:
        # HACK: prefix matching means "o1" matches "o1-mini" too — good enough for now
        return model.startswith(("gpt-", "o1", "o3"))

    async def close(self) -> None:
        await self._client.aclose()
