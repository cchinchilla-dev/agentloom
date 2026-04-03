"""State management for workflow execution."""

from __future__ import annotations

import copy
import json
import re
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from agentloom.core.results import StepResult, StepStatus

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

    def get_sync(self, key: str, default: Any = None) -> Any:
        """Synchronous get for use in non-async contexts (e.g., template rendering)."""
        return self._resolve_key(self._state, key, default)

    def set_sync(self, key: str, value: Any) -> None:
        """Synchronous set for non-async contexts."""
        self._set_nested(self._state, key, value)

    @property
    def state(self) -> dict[str, Any]:
        """Direct access to state dict (use in sync contexts only)."""
        return self._state

    # -- Checkpointing --

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

    # -- Internal helpers --

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
        """
        parts = _parse_path(key)
        current: Any = data
        for part in parts[:-1]:
            if isinstance(part, int):
                if isinstance(current, list) and -len(current) <= part < len(current):
                    current = current[part]
                else:
                    raise IndexError(f"List index {part} out of range in path '{key}'")
            else:
                if part not in current or not isinstance(current[part], (dict, list)):
                    current[part] = {}
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
                raise IndexError(f"List index {last} out of range in path '{key}'")
        else:
            if not isinstance(current, dict):
                raise TypeError(
                    f"Cannot set key '{last}' on {type(current).__name__} in path '{key}'"
                )
            current[last] = value
