"""State management for workflow execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio

from agentloom.core.results import StepResult, StepStatus


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
            return dict(self._state)

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
                "state": self._state,
                "step_results": {k: v.model_dump() for k, v in self._step_results.items()},
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> StateManager:
        """Restore state from a JSON checkpoint file."""
        data = json.loads(Path(path).read_text())
        manager = cls(initial_state=data.get("state", {}))
        for step_id, result_data in data.get("step_results", {}).items():
            manager._step_results[step_id] = StepResult.model_validate(result_data)
            manager._step_status[step_id] = manager._step_results[step_id].status
        return manager

    # -- Internal helpers --

    @staticmethod
    def _resolve_key(data: dict[str, Any], key: str, default: Any = None) -> Any:
        """Resolve a dotted key like 'state.user_input' from nested dicts."""
        parts = key.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @staticmethod
    def _set_nested(data: dict[str, Any], key: str, value: Any) -> None:
        """Set a value at a dotted key path, creating intermediate dicts."""
        # TODO: handle array indices like state.items[0]
        parts = key.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
