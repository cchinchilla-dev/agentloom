"""Mock provider — deterministic replay for tests and offline evaluation.

Responses are loaded from a JSON file keyed by either ``step_id`` or the
SHA-256 hash of the serialized messages. Latency is simulated via
``latency_model`` (``constant``, ``normal``, ``replay``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio

from agentloom.core.results import TokenUsage
from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.base import BaseProvider, EmbeddingResponse, ProviderResponse

logger = logging.getLogger("agentloom.providers.mock")

# Recording-file format version this runtime reads/writes. v1 keyed
# responses by ``step_id`` or a messages-only hash; v2 keys by the full
# request hash and carries a ``request_hash`` on every entry so replay
# can detect a prompt that drifted from its recording.
RECORDING_FORMAT_VERSION = 2


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


def validate_recording_schema(raw: object, source: str) -> dict[str, Any]:
    """Validate a parsed recording file against the canonical schema.

    A recording is a JSON object whose ``_``-prefixed keys are metadata
    (``_version``) and whose remaining keys map a request key to either a
    single response object or a list of response objects (multi-turn
    tool loops). Pre-0.5.0 ``MockProvider`` only checked the top level
    was a dict, so a corrupt file like ``{"not": "valid"}`` loaded
    silently and every lookup fell through to ``default_response`` —
    a replay could pass green against garbage.

    Raises:
        ValueError: On any structural deviation, with the offending key
            named. ``_version`` below :data:`RECORDING_FORMAT_VERSION`
            gets an explicit re-record hint.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Recording {source} must be a JSON object, got {type(raw).__name__}")
    version = raw.get("_version")
    if version is not None:
        if not isinstance(version, int) or version < RECORDING_FORMAT_VERSION:
            raise ValueError(
                f"Recording {source} has _version={version!r}; this runtime needs "
                f"v{RECORDING_FORMAT_VERSION}+. Re-record the fixture with "
                f"`agentloom run --record`."
            )
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, list) and all(isinstance(turn, dict) for turn in value):
            continue
        raise ValueError(
            f"Recording {source} entry {key!r} must be a response object or a "
            f"list of response objects, got {type(value).__name__}"
        )
    return raw


def _pseudo_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding derived from the input hash.

    Not a real embedding — same input always yields the same vector so
    tests can assert on shape without mocking the wire. Values live in
    ``[-1, 0.9921875]`` (asymmetric because ``(255-128)/128`` is not
    exactly 1). The SHA-256 digest is 32 bytes; asking for ``dim > 32``
    cycles the same bytes so downstream cosine-similarity code sees only
    32 unique values regardless of dim — fine for shape checks, not for
    benchmarking against real vectors.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] - 128) / 128.0 for i in range(dim)]


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
    # Replay keys responses by step id — the gateway must forward it.
    accepts_step_id = True

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
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self.responses_file = Path(responses_file) if responses_file else None
        self.latency_model = latency_model
        self.latency_ms = float(latency_ms)
        self.default_response = default_response
        # ``strict`` is the determinism gate. ``agentloom replay`` sets it
        # so a request that misses the recording — or matches a step
        # whose prompt drifted since capture — raises
        # ``RecordingMismatchError`` instead of silently returning
        # ``default_response``. ``agentloom run --provider mock`` keeps
        # it off so ad-hoc mock runs stay frictionless (one-line warning
        # on each miss).
        self.strict = strict
        self._rng = random.Random(seed)
        self._observer = observer
        self._workflow_name = workflow_name
        self.calls: list[dict[str, Any]] = []
        # Each value is either a single response dict OR a list of turns
        # to play in order (for tool-calling loops where one step issues
        # multiple complete() calls). The cursor below tracks which turn
        # the next call should emit per step_id.
        self._responses: dict[str, Any] = {}
        self._turn_cursor: dict[str, int] = {}
        if self.responses_file and self.responses_file.exists():
            try:
                raw = json.loads(self.responses_file.read_text())
            except json.JSONDecodeError as e:
                # A malformed recording file is always an error — pre-0.5.0
                # the JSON parse failure surfaced raw; now it carries the
                # file path so the operator knows which fixture to fix.
                raise ValueError(f"Recording {self.responses_file} is not valid JSON: {e}") from e
            self._responses = validate_recording_schema(raw, str(self.responses_file))

    def _lookup(
        self,
        step_id: str | None,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        request_hash = prompt_hash(messages, model, temperature, max_tokens, extra)
        if step_id and step_id in self._responses:
            entry = self._responses[step_id]
            # List form: pop the next turn (clamp at last so excess
            # iterations replay the final response — saner than
            # raising mid-loop). Multi-turn tool loops carry a distinct
            # prompt per turn, so the drift check below is skipped for
            # lists — the step_id + cursor is the contract there.
            if isinstance(entry, list):
                if not entry:
                    return None
                idx = self._turn_cursor.get(step_id, 0)
                turn = entry[min(idx, len(entry) - 1)]
                self._turn_cursor[step_id] = idx + 1
                return turn if isinstance(turn, dict) else None
            if not isinstance(entry, dict):
                return None
            # Drift check: an entry recorded under a step_id carries the
            # ``request_hash`` it was captured against. In strict (replay)
            # mode a mismatch means the workflow's prompt / system prompt
            # / model / tools spec changed since the recording — refuse
            # rather than answer with stale data (F28). Recordings made
            # before this field existed have no ``request_hash`` and are
            # matched by step_id alone, unchanged.
            recorded_hash = entry.get("request_hash")
            if self.strict and isinstance(recorded_hash, str) and recorded_hash != request_hash:
                raise RecordingMismatchError(
                    f"Step {step_id!r}: the replayed request (hash "
                    f"{request_hash[:16]}) does not match the recording "
                    f"(hash {recorded_hash[:16]}). The prompt, system prompt, "
                    f"model, or tools spec changed since capture — re-record "
                    f"the fixture with `agentloom run --record`."
                )
            return entry
        return self._responses.get(request_hash)

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
        # Strip identifiers that aren't part of the request hash.
        extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("step_id", "agentloom_step_id")
        }
        has_response_schema = "agentloom_response_schema" in extra_kwargs
        entry = self._lookup(step_id, messages, model, temperature, max_tokens, extra_kwargs)
        if entry is None:
            # Strict (replay) mode: a miss is a hard error. Pre-0.5.0 the
            # mock fell through to ``default_response`` even under
            # ``agentloom replay``, so a recording with no entry for a
            # step replayed green with the literal "Mock response" text
            # (F29 / F41).
            if self.strict:
                raise RecordingMismatchError(
                    f"No recorded response for step {step_id!r} (model {model!r}). "
                    f"The recording has no entry matching this request. Re-record "
                    f"the fixture with `agentloom run --record`, or pass "
                    f"`--allow-default-fallback` to replay with the placeholder "
                    f"response."
                )
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
            # Non-strict miss: keep the development-convenience fallback,
            # but emit a one-line warning so a CI assertion passing on
            # the literal placeholder text is at least visible in logs.
            logger.warning(
                "MockProvider: no recorded response for step %r (model %r); "
                "returning the placeholder default. Pass strict=True (the "
                "`agentloom replay` default) to make this an error.",
                step_id,
                model,
            )
            # Schema-bound steps need JSON-shaped fallback.
            fallback_content = "{}" if has_response_schema else self.default_response
            return ProviderResponse(
                content=fallback_content,
                model=model,
                provider=self.name,
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                cost_usd=0.0,
                finish_reason="stop",
            )

        usage_data = entry.get("usage", {}) or {}
        # Hydrate ``tool_calls`` from the recording so replay drives the
        # tool-iteration loop. Each turn carries its own ``tool_calls``.
        tool_calls: list[Any] = []
        recorded_tool_calls = entry.get("tool_calls") or []
        if recorded_tool_calls:
            from agentloom.providers.base import ToolCall

            for tc in recorded_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=str(tc.get("id", "")),
                        name=str(tc.get("name", "")),
                        arguments=tc.get("arguments", {}) or {},
                    )
                )

        return ProviderResponse(
            content=str(entry.get("content", "")),
            model=str(entry.get("model", model)),
            provider=str(entry.get("provider", self.name)),
            usage=TokenUsage(
                prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
                completion_tokens=int(usage_data.get("completion_tokens", 0)),
                total_tokens=int(usage_data.get("total_tokens", 0)),
            ),
            cost_usd=float(entry.get("cost_usd", 0.0)),
            finish_reason=entry.get("finish_reason", "stop"),
            tool_calls=tool_calls,
        )

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Serve embeddings from a recording or synthesize deterministic vectors.

        Recording shape: an entry keyed by ``step_id`` (or the batch hash)
        with an ``embeddings: list[list[float]]`` field. When missing and
        not in strict mode, we synthesize hash-derived vectors so tests
        stay reproducible without needing a fixture per prompt.
        """
        step_id = kwargs.get("step_id")
        extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("step_id", "agentloom_step_id")
        }
        entry: dict[str, Any] | None = None
        if step_id and step_id in self._responses:
            candidate = self._responses[step_id]
            if isinstance(candidate, dict) and "embeddings" in candidate:
                entry = candidate
        if entry is None:
            request_hash = prompt_hash(
                [{"embed_inputs": inputs, "model": model, "dimensions": dimensions}],
                model,
                None,
                None,
                extra_kwargs,
            )
            candidate = self._responses.get(request_hash)
            if isinstance(candidate, dict) and "embeddings" in candidate:
                entry = candidate
        if entry is None and self.strict:
            raise RecordingMismatchError(
                f"No embed recording for step {step_id!r} (model {model!r}). "
                f"Re-record with `agentloom run --record`, or pass "
                f"`--allow-default-fallback` for pseudo-vectors."
            )
        if entry is not None:
            vectors = [list(v) for v in entry["embeddings"]]
            usage_data = entry.get("usage", {}) or {}
            prompt_tokens = int(usage_data.get("prompt_tokens", 0))
            return EmbeddingResponse(
                embeddings=vectors,
                model=str(entry.get("model", model)),
                provider=str(entry.get("provider", self.name)),
                usage=TokenUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
                cost_usd=float(entry.get("cost_usd", 0.0)),
                raw_response=entry,
            )
        # Deterministic fallback: hash-derived pseudo-vector per input.
        dim = dimensions or 8
        vectors = [_pseudo_vector(text, dim) for text in inputs]
        return EmbeddingResponse(
            embeddings=vectors,
            model=model,
            provider=self.name,
            usage=TokenUsage(),
            cost_usd=0.0,
        )

    def supports_model(self, model: str) -> bool:
        return True
