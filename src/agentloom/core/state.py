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

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import StateWriteError

_SEGMENT_RE = re.compile(r"^([^\[]*)((?:\[-?\d+\])*)$")
_INDEX_RE = re.compile(r"\[(-?\d+)\]")


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
    """Manages shared state and step results during workflow execution.

    Thread-safe via anyio.Lock for concurrent step execution.
    Serializable to/from JSON for checkpointing.
    """

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = dict(initial_state or {})
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

    async def update(self, key: str, fn: Callable[[Any], Any]) -> Any:
        """Atomic read-modify-write on a single state key.

        Holds the state lock across the full ``fn(current)`` invocation, so
        compound updates like "fetch the counter, add one, write it back"
        no longer collapse under concurrency. Without this primitive, the
        natural ``cur = await sm.get('counter'); await sm.set('counter',
        cur + 1)`` pattern drops the lock between the two awaits — anything
        that yields (a tool call, an ``await anyio.sleep(0)``) lets another
        writer race and the slower writer overwrites the faster one's
        result. 50 parallel ``update(key, +1)`` calls produce final 50;
        the equivalent ``get`` + ``set`` pair produces final 1 deterministically.

        ``fn`` MUST be synchronous and side-effect-free — it runs while the
        lock is held, so any blocking call would stall every other state
        operation in the workflow. For an async transformation, compute the
        new value outside the lock and pass a lambda that returns the
        already-computed value; this still races if the new value depends
        on the current state, in which case use ``update`` with a sync
        function instead.

        Returns the new value (also what ``fn`` returned). If the key does
        not exist yet, ``fn`` receives ``None``; the caller chooses whether
        to treat that as the seed or refuse it.
        """
        async with self._lock:
            current = self._resolve_key(self._state, key, None)
            new_value = fn(current)
            self._set_nested(self._state, key, new_value)
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
            # Store output in state under steps.<step_id>.output
            if result.output is not None:
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
    #
    # These bypass ``self._lock`` and must not be called from async code
    # that runs concurrently with ``set``/``get``. They exist so tests and
    # internal non-async paths (checkpoint hydration, resume bootstrap) can
    # poke at state without spinning an event loop. Callers in async step
    # handlers must use the awaitable ``get`` / ``set`` / ``get_state_snapshot``
    # variants instead — using these under concurrency produces subtle
    # last-writer-wins bugs because the updates race with in-flight locked
    # writes.

    def _get_sync_unsafe(self, key: str, default: Any = None) -> Any:
        return self._resolve_key(self._state, key, default)

    def _set_sync_unsafe(self, key: str, value: Any) -> None:
        self._set_nested(self._state, key, value)

    @property
    def state(self) -> dict[str, Any]:
        """Raw state dict — **unsafe live reference for internal use only**.

        This returns the internal state dict directly, not a snapshot.
        Mutating it bypasses ``self._lock`` and can race with concurrent
        readers/writers. Do not mutate it and do not rely on it remaining
        stable across ``await`` points. Prefer
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
            p.write_text(json.dumps(payload, indent=2, default=str))

        await anyio.to_thread.run_sync(partial(_write, Path(path), data))

    @classmethod
    async def from_checkpoint(cls, path: str | Path) -> StateManager:
        """Restore state from a JSON checkpoint file."""

        def _read(p: Path) -> dict[str, Any]:
            return json.loads(p.read_text())  # type: ignore[no-any-return]

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
    def _set_nested(data: dict[str, Any], key: str, value: Any) -> None:
        """Set a value at a dotted key path.

        Creates intermediate dicts for missing string segments, but does not
        create or resize lists.  For integer path segments the list and indexed
        element must already exist; otherwise ``IndexError`` is raised.

        Refuses to silently overwrite a scalar with a dict: writing to
        ``user.name`` when ``state.user = "alice"`` (a string) used to
        replace the string with ``{"name": ...}`` and lose the original
        value without warning. Such writes now raise ``StateWriteError``
        so the workflow author either picks a different output key or
        explicitly overwrites the scalar at the parent path first.
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
                    # The path expects a string segment here (e.g. ``user.name``)
                    # but the intermediate is a list — writing ``foo.bar`` when
                    # ``state.foo`` is ``[...]`` is the same class of "wrong-
                    # type intermediate" as the scalar-overwrite case below, so
                    # raise the same exception for a uniform contract instead
                    # of leaking a generic ``TypeError`` to the caller.
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
                # Same shape as the in-loop wrong-type intermediate check —
                # ``users.name`` where ``state.users`` is a list lands here
                # because the final segment expects a dict. Raise
                # ``StateWriteError`` for the uniform "refused write through
                # wrong-type intermediate" contract; callers that catch only
                # the dedicated exception now see this case too.
                traversed = ".".join(str(p) for p in parts[:-1])
                raise StateWriteError(
                    f"Cannot write to {key!r}: parent path {traversed!r} resolves "
                    f"to a {type(current).__name__}, not a dict. Use an integer "
                    f"index to traverse a list, or rebuild the parent path first."
                )
            current[last] = value
