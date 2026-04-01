"""Base provider interface and response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError


class ProviderResponse(BaseModel):
    """Unified response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class StreamResponse:
    """Accumulates streamed text chunks and final metadata from a provider.

    Usage::

        sr = await provider.stream(messages, model)
        async for chunk in sr:
            print(chunk, end="")
        response = sr.to_provider_response()
    """

    #: Safety limit for accumulated content (10 MB).
    MAX_ACCUMULATED_BYTES: int = 10 * 1024 * 1024

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        self.usage: TokenUsage = TokenUsage()
        self.cost_usd: float = 0.0
        self.finish_reason: str | None = None
        self._chunks: list[str] = []
        self._accumulated_bytes: int = 0
        self._iterator: AsyncIterator[str] | None = None

    def _set_iterator(self, iterator: AsyncIterator[str]) -> None:
        self._iterator = iterator

    def __aiter__(self) -> StreamResponse:
        return self

    async def __anext__(self) -> str:
        if self._iterator is None:
            raise StopAsyncIteration
        chunk = await anext(self._iterator)
        self._accumulated_bytes += len(chunk.encode("utf-8"))
        if self._accumulated_bytes > self.MAX_ACCUMULATED_BYTES:
            raise ProviderError(
                self.provider,
                f"Stream exceeded {self.MAX_ACCUMULATED_BYTES} byte limit",
            )
        self._chunks.append(chunk)
        return chunk

    @property
    def content(self) -> str:
        """Full accumulated text after iteration."""
        return "".join(self._chunks)

    def to_provider_response(self) -> ProviderResponse:
        """Convert to a ProviderResponse after the stream is exhausted."""
        return ProviderResponse(
            content=self.content,
            model=self.model,
            provider=self.provider,
            usage=self.usage,
            cost_usd=self.cost_usd,
            finish_reason=self.finish_reason,
        )


class BaseProvider(ABC):
    """Abstract base class for LLM provider adapters."""

    name: str = "base"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        """Stream a completion response. Default: fall back to complete()."""
        response = await self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        sr = StreamResponse(model=response.model, provider=response.provider)
        sr.usage = response.usage
        sr.cost_usd = response.cost_usd
        sr.finish_reason = response.finish_reason

        async def _single_chunk() -> AsyncIterator[str]:
            yield response.content

        sr._set_iterator(_single_chunk())
        return sr

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a given model. Override in subclasses."""
        return True

    async def close(self) -> None:
        """Clean up resources. Override if the provider uses persistent connections."""
