"""Base provider interface and response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from agentloom.core.results import TokenUsage


class ProviderResponse(BaseModel):
    """Unified response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class BaseProvider(ABC):
    """Abstract base class for LLM provider adapters."""

    name: str = "base"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a completion request to the provider.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model identifier (e.g., 'gpt-4o-mini').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Unified ProviderResponse.
        """
        ...

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion response. Default: fall back to complete()."""
        response = await self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        yield response.content

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a given model. Override in subclasses."""
        return True

    async def close(self) -> None:
        """Clean up resources. Override if the provider uses persistent connections."""
