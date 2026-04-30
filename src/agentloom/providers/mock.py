"""Mock provider — deterministic replay for tests and offline evaluation.

Responses are loaded from a JSON file keyed by either ``step_id`` or the
SHA-256 hash of the serialized messages. Latency is simulated via
``latency_model`` (``constant``, ``normal``, ``replay``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio

from agentloom.core.results import TokenUsage
from agentloom.providers.base import BaseProvider, ProviderResponse


@runtime_checkable
class MockObserver(Protocol):
    """Minimal observer interface for MockProvider replay events."""

    def on_mock_replay(self, workflow_name: str, step_id: str, matched_by: str) -> None: ...


def _canonical_default(obj: Any) -> Any:
    """JSON ``default`` that handles Pydantic models stably across versions.

    ``json.dumps(..., default=str)`` serialized Pydantic instances via ``repr``
    which changes across minor versions and breaks recorded fixtures on upgrade.
    Prefer ``model_dump()`` when available so the canonical payload depends on
    the model's public field shape only.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def prompt_hash(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable SHA-256 hash of a completion request for response keying.

    The hash covers every field that can change the model's response:
    messages, model, temperature, max_tokens, and an optional ``extra`` bag
    for forwarded kwargs (e.g. ``response_format``). Previous versions keyed
    on messages only, which caused cross-model collisions.
    """
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": extra or {},
    }
    serialized = json.dumps(payload, sort_keys=True, default=_canonical_default).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class MockProvider(BaseProvider):
    """Deterministic provider that returns pre-recorded responses.

    Response file format::

        {
          "<key>": {
            "content": "...",
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "cost_usd": 0.001,
            "latency_ms": 120.0,
            "finish_reason": "stop"
          }
        }

    ``<key>`` is either a step_id (if the caller passes ``step_id=`` through
    kwargs) or the SHA-256 hash of the serialized messages.
    """

    name = "mock"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        responses_file: str | Path | None = None,
        latency_model: str = "constant",
        latency_ms: float = 0.0,
        default_response: str = "Mock response",
        seed: int | None = None,
        observer: MockObserver | None = None,
        workflow_name: str = "unknown",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self.responses_file = Path(responses_file) if responses_file else None
        self.latency_model = latency_model
        self.latency_ms = float(latency_ms)
        self.default_response = default_response
        self._rng = random.Random(seed)
        self._observer = observer
        self._workflow_name = workflow_name
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[str, dict[str, Any]] = {}
        if self.responses_file and self.responses_file.exists():
            raw = json.loads(self.responses_file.read_text())
            if not isinstance(raw, dict):
                raise ValueError(f"responses_file {self.responses_file} must contain a JSON object")
            self._responses = raw

    def _lookup(
        self,
        step_id: str | None,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if step_id and step_id in self._responses:
            return self._responses[step_id]
        key = prompt_hash(messages, model, temperature, max_tokens, extra)
        return self._responses.get(key)

    async def _sleep(self, recorded_ms: float | None) -> None:
        if self.latency_model == "replay" and recorded_ms is not None:
            delay = recorded_ms
        elif self.latency_model == "normal":
            sigma = max(self.latency_ms * 0.1, 1.0)
            delay = max(0.0, self._rng.gauss(self.latency_ms, sigma))
        else:  # constant
            delay = self.latency_ms
        if delay > 0:
            await anyio.sleep(delay / 1000.0)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        step_id = kwargs.get("step_id")
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in ("step_id",)}
        entry = self._lookup(step_id, messages, model, temperature, max_tokens, extra_kwargs)
        if entry is None:
            matched_by = "default"
        elif step_id and step_id in self._responses:
            matched_by = "step_id"
        else:
            matched_by = "prompt_hash"
        self.calls.append(
            {
                "step_id": step_id,
                "model": model,
                "messages": messages,
                "matched": entry is not None,
                "matched_by": matched_by,
            }
        )
        if self._observer is not None:
            # observer must never break replay
            with contextlib.suppress(Exception):  # pragma: no cover
                self._observer.on_mock_replay(self._workflow_name, step_id or "", matched_by)
        recorded_latency = entry.get("latency_ms") if entry else None
        await self._sleep(recorded_latency if isinstance(recorded_latency, int | float) else None)

        if entry is None:
            return ProviderResponse(
                content=self.default_response,
                model=model,
                provider=self.name,
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                cost_usd=0.0,
                finish_reason="stop",
            )

        usage_data = entry.get("usage", {}) or {}
        return ProviderResponse(
            content=str(entry.get("content", "")),
            model=str(entry.get("model", model)),
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
                completion_tokens=int(usage_data.get("completion_tokens", 0)),
                total_tokens=int(usage_data.get("total_tokens", 0)),
            ),
            cost_usd=float(entry.get("cost_usd", 0.0)),
            finish_reason=entry.get("finish_reason", "stop"),
        )

    def supports_model(self, model: str) -> bool:
        return True
