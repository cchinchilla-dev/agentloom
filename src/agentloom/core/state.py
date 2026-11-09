"""State management for workflow execution."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import anyio
from pydantic import BaseModel

from agentloom.core.inter_agent import log_write
from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import StateWriteError


def _json_default(obj: Any) -> Any:
    """JSON ``default`` that survives Pydantic models on checkpoint write.

    A model goes through ``model_dump()``, not its ``repr``, so a :class:`Conversation`
    or :class:`Message` stored under a state key round-trips across save and load.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return str(obj)


_SEGMENT_RE = re.compile(r"^([^\[]*)((?:\[-?\d+\])*)$")
_INDEX_RE = re.compile(r"\[(-?\d+)\]")

# A state value can be ``None``, so ``key_exists`` needs a distinct absent marker
_KEY_ABSENT = object()


def _parse_path(key: str) -> list[str | int]:
    """Split a dotted key with optional array indices into access operations.

    Examples::

        "items[0].name"  -> ["items", 0, "name"]
        "matrix[0][1]"   -> ["matrix", 0, 1]
        "a.b.c"          -> ["a", "b", "c"]
    """
    if not key:
        raise ValueError("State path must not be empty")
    parts: list[str | int] = []
    for segment in key.split("."):
        if not segment:
            raise ValueError(f"Empty segment in state path '{key}'")
        m = _SEGMENT_RE.match(segment)
        if m:
            name, indices = m.group(1), m.group(2)
            if name:
                parts.append(name)
            if indices:
                parts.extend(int(i) for i in _INDEX_RE.findall(indices))
        else:
            # Fallback: treat the whole segment as a dict key
            parts.append(segment)
    return parts


class StateManager:
    """Shared state and step results of a workflow run.

    Task-safe via ``anyio.Lock`` for concurrent steps.
    Serializable to/from JSON for checkpointing.
    """

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        # Deep: a dotted set mutates nested dicts, which would otherwise be the caller's.
        self._state: dict[str, Any] = copy.deepcopy(initial_state or {})
        self._step_results: dict[str, StepResult] = {}
        self._step_status: dict[str, StepStatus] = {}
        self._lock = anyio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the workflow state."""
        async with self._lock:
            return self._resolve_key(self._state, key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value in the workflow state."""
        async with self._lock:
            self._set_nested(self._state, key, value)
        log_write(key, value)

    async def update(self, key: str, fn: Callable[[Any], Any]) -> Any:
        """Atomic read-modify-write on a single state key.

        Holds the state lock across ``fn(current)``, so a compound update ("read the
        counter, add one, write it back") cannot interleave with other writers. A
        ``get`` + ``set`` pair drops the lock between its awaits: 50 parallel
        ``update(key, +1)`` calls end at 50, the equivalent pairs at 1.

        ``fn`` must be synchronous and side-effect-free: it runs under the lock, so a
        blocking call stalls every other state operation. For an async transformation,
        compute the value outside the lock and pass a lambda that returns it; that still
        races when the value depends on the current state, so prefer a sync ``fn`` then.

        Returns the new value (also what ``fn`` returned). If the key does
        not exist yet, ``fn`` receives ``None``; the caller chooses whether
        to treat that as the seed or refuse it.
        """
        async with self._lock:
            current = self._resolve_key(self._state, key, None)
            new_value = fn(current)
            self._set_nested(self._state, key, new_value)
        log_write(key, new_value)
        return new_value

    async def get_state_snapshot(self) -> dict[str, Any]:
        """Return a copy of the current state."""
        async with self._lock:
            return copy.deepcopy(self._state)

    async def set_step_result(self, step_id: str, result: StepResult) -> None:
        """Record a step's execution result."""
        async with self._lock:
            self._step_results[step_id] = result
            self._step_status[step_id] = result.status
            # Only a success lands in state: a failed subworkflow would leak its child state there
            if result.status == StepStatus.SUCCESS and result.output is not None:
                self._state.setdefault("steps", {})[step_id] = {
                    "output": result.output,
                    "status": result.status.value,
                }

    async def get_step_result(self, step_id: str) -> StepResult | None:
        """Get a step's execution result."""
        async with self._lock:
            return self._step_results.get(step_id)

    async def get_step_status(self, step_id: str) -> StepStatus | None:
        """Get a step's execution status."""
        async with self._lock:
            return self._step_status.get(step_id)

    async def all_step_results(self) -> dict[str, StepResult]:
        """Return all step results."""
        async with self._lock:
            return dict(self._step_results)

    # ---- internal-only synchronous helpers --------------------------------
    # Test helpers that bypass ``self._lock``: never call them while steps run concurrently

    def _get_sync_unsafe(self, key: str, default: Any = None) -> Any:
        return self._resolve_key(self._state, key, default)

    def _set_sync_unsafe(self, key: str, value: Any) -> None:
        self._set_nested(self._state, key, value)

    @property
    def state(self) -> dict[str, Any]:
        """The live internal state dict, not a snapshot; for internal use only.

        Mutating it bypasses ``self._lock`` and races with concurrent readers and
        writers, and it can change across ``await`` points. Prefer
        ``await self.get_state_snapshot()`` for a defensive copy.
        """
        return self._state

    async def save_checkpoint(self, path: str | Path) -> None:
        """Save current state to a JSON file."""
        async with self._lock:
            data = {
                "state": copy.deepcopy(self._state),
                "step_results": {k: v.model_dump() for k, v in self._step_results.items()},
            }

        def _write(p: Path, payload: dict[str, Any]) -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2, default=_json_default))

        await anyio.to_thread.run_sync(partial(_write, Path(path), data))

    @classmethod
    async def from_checkpoint(cls, path: str | Path) -> StateManager:
        """Restore state from a JSON checkpoint file."""

        def _read(p: Path) -> dict[str, Any]:
            return json.loads(p.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

        data = await anyio.to_thread.run_sync(partial(_read, Path(path)))
        manager = cls(initial_state=data.get("state", {}))
        for step_id, result_data in data.get("step_results", {}).items():
            manager._step_results[step_id] = StepResult.model_validate(result_data)
            manager._step_status[step_id] = manager._step_results[step_id].status
        return manager

    @staticmethod
    def _resolve_key(data: dict[str, Any], key: str, default: Any = None) -> Any:
        """Resolve a dotted key like 'user.name' or 'items[0].name' from nested data."""
        parts = _parse_path(key)
        current: Any = data
        for part in parts:
            if isinstance(part, int):
                if isinstance(current, list) and -len(current) <= part < len(current):
                    current = current[part]
                else:
                    return default
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @staticmethod
    def key_exists(data: dict[str, Any], key: str) -> bool:
        """Return ``True`` when the dotted *key* path resolves to a value.

        Distinct from ``_resolve_key(...) is None``: a state value can
        legitimately be ``None``, so callers that need "is this path
        present" — e.g. ``tool_step`` rejecting a ``state.X`` reference
        to a key that does not exist — must ask this rather than test
        the resolved value for falsiness.
        """
        return StateManager._resolve_key(data, key, _KEY_ABSENT) is not _KEY_ABSENT

    @staticmethod
    def _set_nested(data: dict[str, Any], key: str, value: Any) -> None:
        """Set a value at a dotted key path.

        Creates intermediate dicts for missing string segments, but does not
        create or resize lists.  For integer path segments the list and indexed
        element must already exist; otherwise ``IndexError`` is raised.

        Refuses to overwrite a scalar with a dict: writing ``user.name`` when
        ``state.user`` is ``"alice"`` raises ``StateWriteError``, so the workflow author
        picks another output key or overwrites the scalar at the parent path first.
        """
        parts = _parse_path(key)
        current: Any = data
        for i, part in enumerate(parts[:-1]):
            if isinstance(part, int):
                if isinstance(current, list) and -len(current) <= part < len(current):
                    current = current[part]
                else:
                    raise IndexError(
                        f"List index {part} out of range in path '{key}'. "
                        f"Lists are not auto-expanded by set(); pre-allocate the list "
                        f"(e.g. via an initial value containing {part + 1} elements) "
                        f"before writing to indexed paths."
                    )
            else:
                if not isinstance(current, dict):
                    # A list under a string segment: the same refusal as the scalar case below
                    traversed = ".".join(str(p) for p in parts[:i])
                    raise StateWriteError(
                        f"Cannot write to {key!r}: intermediate {traversed!r} is a "
                        f"{type(current).__name__}, not a dict. Use an integer "
                        f"index to traverse a list, or rebuild the parent path "
                        f"explicitly."
                    )
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], (dict, list)):
                    traversed = ".".join(str(p) for p in parts[: i + 1])
                    raise StateWriteError(
                        f"Cannot write to {key!r}: intermediate {traversed!r} is a "
                        f"{type(current[part]).__name__} (value={current[part]!r}), "
                        f"not a dict. Refusing to silently overwrite the scalar."
                    )
                current = current[part]
            if not isinstance(current, (dict, list)):
                raise TypeError(
                    f"Expected dict or list at '{part}' in path '{key}', "
                    f"got {type(current).__name__}"
                )
        last = parts[-1]
        if isinstance(last, int):
            if isinstance(current, list) and -len(current) <= last < len(current):
                current[last] = value
            else:
                raise IndexError(
                    f"List index {last} out of range in path '{key}'. "
                    f"Lists are not auto-expanded by set(); pre-allocate the list "
                    f"before writing to indexed paths."
                )
        else:
            if not isinstance(current, dict):
                # A list parent under a final string segment: refused as in the loop
                traversed = ".".join(str(p) for p in parts[:-1])
                raise StateWriteError(
                    f"Cannot write to {key!r}: parent path {traversed!r} resolves "
                    f"to a {type(current).__name__}, not a dict. Use an integer "
                    f"index to traverse a list, or rebuild the parent path first."
                )
            current[last] = value


class LayerStateView:
    """What one step of a layer reads: the state as the layer started, plus its own writes.

    Writes go to the run's :class:`StateManager`, and anything else is answered by it.
    """

    def __init__(self, run_state: StateManager, at_layer_start: dict[str, Any]) -> None:
        self._run_state = run_state
        self._seen = copy.deepcopy(at_layer_start)

    async def get(self, key: str, default: Any = None) -> Any:
        return StateManager._resolve_key(self._seen, key, default)

    async def get_state_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._seen)

    async def set(self, key: str, value: Any) -> None:
        await self._run_state.set(key, value)
        StateManager._set_nested(self._seen, key, value)

    async def update(self, key: str, fn: Callable[[Any], Any]) -> Any:
        value = await self._run_state.update(key, fn)
        StateManager._set_nested(self._seen, key, value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run_state, name)
