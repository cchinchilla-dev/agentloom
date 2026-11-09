"""Base provider interface and response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from agentloom.contracts.stable import ProviderResponse as ProviderResponse
from agentloom.contracts.stable import ResponseMeta as ResponseMeta
from agentloom.contracts.stable import ToolCall as ToolCall
from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError


class StreamEvent(BaseModel):
    """Base type for typed stream events surfaced via ``StreamResponse.events()``.

    ``async for chunk in sr`` yields the plain text deltas; ``async for evt in
    sr.events()`` yields typed events instead, to switch on by subclass.
    """


class TextDelta(StreamEvent):
    """A chunk of text content from the model."""

    chunk: str


class ToolCallDelta(StreamEvent):
    """A partial tool-call observation while the model assembles arguments.

    ``index`` matches across deltas of the same call; ``name`` arrives on
    the first delta and is ``None`` on subsequent argument-fragment
    deltas; ``arguments_chunk`` accumulates the JSON-encoded args.
    """

    index: int
    name: str | None = None
    arguments_chunk: str = ""


class ToolCallComplete(StreamEvent):
    """A fully-assembled ``ToolCall`` ready to dispatch."""

    tool_call: ToolCall


class StreamDone(StreamEvent):
    """Emitted when the stream finishes; carries the provider's stop reason."""

    finish_reason: str = ""


class EmbeddingResponse(BaseModel):
    """Unified embedding response from any provider.

    ``embeddings`` is one vector per input, in the same order as the
    request. ``usage.completion_tokens`` is always ``0`` — embedding APIs
    only bill input tokens.
    """

    embeddings: list[list[float]]
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)


class StreamResponse:
    """Streamed text chunks and final metadata of a provider response.

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
        # Chain-of-thought trace, from the streaming adapters that surface one (Gemini, Ollama).
        self.reasoning_content: str | None = None
        # Only the complete()-based fallback fills it: no adapter's own stream() parses tool calls.
        self.tool_calls: list[ToolCall] = []
        # Set by the adapter once the response headers arrive.
        self.meta: ResponseMeta | None = None
        self.raw_response: dict[str, Any] = {}
        # Set by the gateway, which waits for the first chunk before handing the stream over.
        self.time_to_first_token_ms: float | None = None
        self._chunks: list[str] = []
        self._accumulated_bytes: int = 0
        self._iterator: AsyncIterator[str] | None = None
        self._event_iterator: AsyncIterator[StreamEvent] | None = None

    def _set_iterator(self, iterator: AsyncIterator[str]) -> None:
        self._iterator = iterator

    def _set_event_iterator(self, iterator: AsyncIterator[StreamEvent]) -> None:
        """Register the iterator of an adapter that emits typed events.

        When unset, ``events()`` synthesizes ``TextDelta`` events from
        the plain-string iterator plus a final ``StreamDone`` so callers
        always get a working typed surface.
        """
        self._event_iterator = iterator

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Iterate typed stream events.

        An iterator set with ``_set_event_iterator`` is used as is; no built-in
        adapter sets one. Otherwise wrap the plain-string iterator: each chunk
        → ``TextDelta``, then a single ``StreamDone`` once it is exhausted.
        """
        if self._event_iterator is not None:
            async for event in self._event_iterator:
                yield event
            return
        async for chunk in self:
            yield TextDelta(chunk=chunk)
        yield StreamDone(finish_reason=self.finish_reason or "")

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
            reasoning_content=self.reasoning_content,
            tool_calls=list(self.tool_calls),
            meta=self.meta,
            raw_response=self.raw_response,
        )


class BaseProvider(ABC):
    """Abstract base class for LLM provider adapters."""

    name: str = "base"

    # True makes the gateway forward ``step_id``: the mock and recording providers key by it.
    accepts_step_id: bool = False

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
        """Send a completion request to the provider."""
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
        # Copy every field, so a provider streamed through this fallback loses no tool call.
        sr.reasoning_content = response.reasoning_content
        sr.tool_calls = list(response.tool_calls)
        sr.meta = response.meta
        sr.raw_response = response.raw_response

        async def _single_chunk() -> AsyncIterator[str]:
            yield response.content

        sr._set_iterator(_single_chunk())
        return sr

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Embed a batch of texts. Default: refuse — providers opt in."""
        raise NotImplementedError(f"Provider {self.name!r} does not implement embeddings.")

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a given model. Override in subclasses."""
        return True

    async def close(self) -> None:
        """Clean up resources. Override if the provider uses persistent connections."""
