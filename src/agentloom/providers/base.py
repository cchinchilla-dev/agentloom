"""Base provider interface and response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from agentloom.core.results import TokenUsage
from agentloom.exceptions import ProviderError


class ToolCall(BaseModel):
    """A function-call decision returned by the model.

    ``id`` round-trips on the follow-up tool-result message (OpenAI:
    ``tool_call_id``, Anthropic: ``tool_use_id``). ``arguments`` is
    already JSON-decoded — adapters parse before constructing this.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    """Base type for typed stream events surfaced via ``StreamResponse.events()``.

    Backwards-compat: ``async for chunk in sr`` keeps yielding plain text
    strings (the deltas only). Callers that need to react to tool-call
    decisions mid-stream use ``async for evt in sr.events()`` and switch
    on the concrete subclass.
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


class ProviderResponse(BaseModel):
    """Unified response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
    # Reasoning / chain-of-thought text. Populated when the provider
    # exposes the trace and the caller did not opt out via
    # ``ThinkingConfig.capture_reasoning``. OpenAI o-series keeps the
    # trace server-side, so this stays ``None``; Anthropic concatenates
    # ``type="thinking"`` blocks; Gemini surfaces ``thought=true`` parts;
    # Ollama returns ``message.thinking`` (or strips inline
    # ``<think>...</think>`` tags as a fallback).
    reasoning_content: str | None = None
    # Empty when the model didn't pick a tool. When non-empty, ``content``
    # may be empty — the LLM step dispatches each call and re-prompts.
    tool_calls: list[ToolCall] = Field(default_factory=list)


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
        # Populated by adapters that surface a chain-of-thought trace
        # alongside the streamed answer (Gemini ``thought=true`` parts,
        # Anthropic ``thinking`` deltas, Ollama ``message.thinking``).
        self.reasoning_content: str | None = None
        # Tool calls accumulated by the adapter while streaming. Adapters
        # that emit ``ToolCallDelta`` events through the typed event API
        # also populate this list with the assembled calls so non-event
        # consumers can read them after the stream is exhausted.
        self.tool_calls: list[ToolCall] = []
        self._chunks: list[str] = []
        self._accumulated_bytes: int = 0
        self._iterator: AsyncIterator[str] | None = None
        self._event_iterator: AsyncIterator[StreamEvent] | None = None

    def _set_iterator(self, iterator: AsyncIterator[str]) -> None:
        self._iterator = iterator

    def _set_event_iterator(self, iterator: AsyncIterator[StreamEvent]) -> None:
        """Adapters that emit typed events register a separate iterator here.

        When unset, ``events()`` synthesises ``TextDelta`` events from
        the plain-string iterator plus a final ``StreamDone`` so callers
        always get a working typed surface.
        """
        self._event_iterator = iterator

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Iterate typed stream events (text + tool-call deltas + done).

        Adapters that wired ``_set_event_iterator`` get full fidelity
        (per-provider tool-call streaming). The default fallback wraps
        the plain-string iterator: each chunk → ``TextDelta``, then a
        single ``StreamDone`` once the underlying iterator is exhausted.
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
        # Propagate reasoning trace + tool calls so callers that read
        # ``sr.to_provider_response()`` see the same fields they'd get
        # from a non-streaming call. Without this, providers that rely on
        # the fallback silently lose tool decisions when streamed.
        sr.reasoning_content = response.reasoning_content
        sr.tool_calls = list(response.tool_calls)

        async def _single_chunk() -> AsyncIterator[str]:
            yield response.content

        sr._set_iterator(_single_chunk())
        return sr

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a given model. Override in subclasses."""
        return True

    async def close(self) -> None:
        """Clean up resources. Override if the provider uses persistent connections."""
