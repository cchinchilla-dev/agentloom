"""Recording provider — wraps a real provider and captures responses to JSON.

The captured file is directly loadable by :class:`MockProvider` for
deterministic replay.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.mock import prompt_hash


@runtime_checkable
class RecordingObserver(Protocol):
    """Minimal observer interface for RecordingProvider capture events."""

    def on_recording_capture(
        self, step_id: str, provider: str, model: str, latency_s: float
    ) -> None: ...


class RecordingProvider(BaseProvider):
    """Delegates to a wrapped provider and records each completion.

    Responses are keyed by ``step_id`` when present, otherwise by the SHA-256
    hash of the serialized messages. The file is flushed on every call so a
    crashed workflow still leaves a partial recording on disk.
    """

    def __init__(
        self,
        wrapped: BaseProvider,
        output_path: str | Path,
        observer: RecordingObserver | None = None,
    ) -> None:
        super().__init__(api_key=wrapped.api_key, base_url=wrapped.base_url)
        self.name = wrapped.name
        self._wrapped = wrapped
        self.output_path = Path(output_path)
        self._observer = observer
        self._recorded: dict[str, dict[str, Any]] = {}
        if self.output_path.exists():
            try:
                existing = json.loads(self.output_path.read_text())
                if isinstance(existing, dict):
                    self._recorded = existing
            except json.JSONDecodeError:
                pass

    def _key(self, step_id: str | None, messages: list[dict[str, Any]]) -> str:
        return step_id if step_id else prompt_hash(messages)

    def _flush(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge with any existing on-disk content so concurrent recorders
        # (e.g. wrapping multiple providers in a gateway) don't clobber
        # each other on close().
        merged: dict[str, dict[str, Any]] = {}
        if self.output_path.exists():
            try:
                existing = json.loads(self.output_path.read_text())
                if isinstance(existing, dict):
                    merged.update(existing)
            except json.JSONDecodeError:
                pass
        merged.update(self._recorded)
        self.output_path.write_text(json.dumps(merged, indent=2, default=str))

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        step_id = kwargs.get("step_id")
        start = time.perf_counter()
        response = await self._wrapped.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        self._recorded[self._key(step_id, messages)] = {
            "content": response.content,
            "model": response.model,
            "usage": response.usage.model_dump(),
            "cost_usd": response.cost_usd,
            "latency_ms": latency_ms,
            "finish_reason": response.finish_reason,
        }
        self._flush()
        if self._observer is not None:
            # observer must never break capture
            with contextlib.suppress(Exception):  # pragma: no cover
                self._observer.on_recording_capture(
                    step_id or "", response.provider, response.model, latency_ms / 1000.0
                )
        return response

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        return await self._wrapped.stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def supports_model(self, model: str) -> bool:
        return self._wrapped.supports_model(model)

    async def close(self) -> None:
        self._flush()
        await self._wrapped.close()
