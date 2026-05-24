"""Recording provider — wraps a real provider and captures responses to JSON.

The captured file is directly loadable by :class:`MockProvider` for
deterministic replay.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio

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

    # The recorder keys captures by step id — the gateway must forward
    # it (the wrapped HTTP adapter never sees it; the recorder consumes
    # it before delegating).
    accepts_step_id = True

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
        # Serializes both mutations of ``_recorded`` and the flush-to-disk
        # path so concurrent ``complete()`` calls from parallel DAG layers
        # neither drop entries (read-modify-write race across threads) nor
        # raise ``RuntimeError: dictionary changed size during iteration``.
        self._write_lock = anyio.Lock()
        if self.output_path.exists():
            try:
                existing = json.loads(self.output_path.read_text())
                if isinstance(existing, dict):
                    # Strip the metadata envelope when reading v2+ files.
                    self._recorded = {k: v for k, v in existing.items() if not k.startswith("_")}
            except json.JSONDecodeError:
                pass

    def _key(
        self,
        step_id: str | None,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if step_id:
            return step_id
        return prompt_hash(messages, model, temperature, max_tokens, extra)

    def _wrapped_kwargs(self, extra_kwargs: dict[str, Any], step_id: str | None) -> dict[str, Any]:
        """Build the kwargs forwarded to the wrapped provider.

        ``step_id`` is added back only when the wrapped provider accepts
        it (another mock / recorder) — an HTTP adapter rejects unknown
        parameters, so it receives ``extra_kwargs`` unchanged.
        """
        if step_id is not None and getattr(self._wrapped, "accepts_step_id", False):
            return {**extra_kwargs, "step_id": step_id}
        return dict(extra_kwargs)

    async def _flush(self) -> None:
        async with self._write_lock:
            snapshot = dict(self._recorded)
            await anyio.to_thread.run_sync(self._flush_sync, snapshot)

    def _flush_sync(self, snapshot: dict[str, dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge with any existing on-disk content so concurrent recorders
        # (e.g. wrapping multiple providers in a gateway) don't clobber
        # each other on close().
        merged: dict[str, Any] = {"_version": 2}
        if self.output_path.exists():
            try:
                existing = json.loads(self.output_path.read_text())
                if isinstance(existing, dict):
                    merged.update(existing)
            except json.JSONDecodeError:
                pass
        merged.update(snapshot)
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
        extra_kwargs = {k: v for k, v in kwargs.items() if k != "step_id"}
        start = time.perf_counter()
        # Forward ``step_id`` to the wrapped provider only when it
        # consumes it (another mock / recorder). An HTTP adapter would
        # reject the unknown parameter, so it gets ``extra_kwargs`` only.
        response = await self._wrapped.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **self._wrapped_kwargs(extra_kwargs, step_id),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        key = self._key(step_id, messages, model, temperature, max_tokens, extra_kwargs)
        entry = {
            "content": response.content,
            "model": response.model,
            "usage": response.usage.model_dump(),
            "cost_usd": response.cost_usd,
            "latency_ms": latency_ms,
            "finish_reason": response.finish_reason,
            # The full request hash rides on every entry — even when the
            # entry is keyed by ``step_id`` — so a strict-mode replay can
            # detect a prompt that drifted from its recording (F28).
            "request_hash": prompt_hash(messages, model, temperature, max_tokens, extra_kwargs),
        }
        async with self._write_lock:
            self._recorded[key] = entry
        await self._flush()
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
        step_id = kwargs.get("step_id")
        extra_kwargs = {k: v for k, v in kwargs.items() if k != "step_id"}
        start = time.perf_counter()
        # Forward ``step_id`` only to a wrapped provider that consumes
        # it — see ``complete`` above.
        inner_sr = await self._wrapped.stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **self._wrapped_kwargs(extra_kwargs, step_id),
        )

        outer_sr = StreamResponse(model=inner_sr.model, provider=inner_sr.provider)
        buffer: list[str] = []
        recorder = self

        async def _tap() -> AsyncIterator[str]:
            raw = inner_sr._iterator
            if raw is None:
                return
            try:
                async for chunk in raw:
                    buffer.append(chunk)
                    yield chunk
            finally:
                outer_sr.usage = inner_sr.usage
                outer_sr.cost_usd = inner_sr.cost_usd
                outer_sr.finish_reason = inner_sr.finish_reason
                outer_sr.model = inner_sr.model

            latency_ms = (time.perf_counter() - start) * 1000.0
            key = recorder._key(step_id, messages, model, temperature, max_tokens, extra_kwargs)
            entry = {
                "content": "".join(buffer),
                "model": inner_sr.model,
                "usage": inner_sr.usage.model_dump(),
                "cost_usd": inner_sr.cost_usd,
                "latency_ms": latency_ms,
                "finish_reason": inner_sr.finish_reason,
                "request_hash": prompt_hash(messages, model, temperature, max_tokens, extra_kwargs),
            }
            async with recorder._write_lock:
                recorder._recorded[key] = entry
            await recorder._flush()
            if recorder._observer is not None:
                with contextlib.suppress(Exception):  # pragma: no cover
                    recorder._observer.on_recording_capture(
                        step_id or "",
                        inner_sr.provider,
                        inner_sr.model,
                        latency_ms / 1000.0,
                    )

        outer_sr._set_iterator(_tap())
        return outer_sr

    def supports_model(self, model: str) -> bool:
        return self._wrapped.supports_model(model)

    async def close(self) -> None:
        await self._flush()
        await self._wrapped.close()
