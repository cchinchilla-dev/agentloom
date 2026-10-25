"""Tool results in a recording: capture them, serve them back, or check them (#164).

``agentloom run --record rec.json --record-tools`` stores the tool calls of
each step under ``_tools``, in call order::

    "_tools": {"<step_id>": [{"name": "...", "arguments": {...}, "result": "...", "success": true}]}

``agentloom replay --tools replay`` serves those results without running the
tools, and ``--tools verify`` runs them and compares. The dispatcher finds
the tape on ``ToolRegistry.tape``; the engine tells it when a step starts an
attempt, so a retried step leaves no trace of its failed attempt.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import anyio

from agentloom.core.attempts import AttemptLog
from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.mock import validate_recording_schema
from agentloom.providers.recorder import merge_into_recording

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentloom.core.attempts import Attempt
    from agentloom.providers.base import ToolCall

logger = logging.getLogger("agentloom.providers.tool_tape")


class ToolMode(StrEnum):
    """How ``agentloom replay`` treats the tool calls of a workflow."""

    LIVE = "live"
    REPLAY = "replay"
    VERIFY = "verify"


class ToolTape(Protocol):
    """What the engine and the dispatcher ask of ``ToolRegistry.tape``."""

    async def begin_attempt(self, step_id: str, failed: Attempt | None) -> None:
        """A step starts an attempt; on a retry, ``failed`` is the attempt it replaces."""

    def serve(self, step_id: str, calls: list[ToolCall]) -> list[tuple[str, bool] | None]:
        """The recorded ``(result, success)`` of each call; ``None`` runs the tool."""

    async def capture(self, step_id: str, results: list[tuple[ToolCall, str, bool]]) -> None:
        """Take the outcome of each call, in call order, once the turn has run."""


def _canonical(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ToolRecorder:
    """Writes the tool calls of this run under ``_tools`` in a recording file.

    The first capture of a step in a run replaces what an earlier run left
    under it; the other steps survive, as with the LLM entries. A retry
    drops the calls of the attempt that failed, a subworkflow's included,
    so the recording holds the calls of the attempt that finished.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._steps: dict[str, list[dict[str, Any]]] = {}
        # Calls of each step before an attempt first recorded it: a retry goes back there.
        self._attempts = AttemptLog()
        self._lock = anyio.Lock()

    async def begin_attempt(self, step_id: str, failed: Attempt | None) -> None:
        if failed is None:
            return
        async with self._lock:
            taken_back = self._attempts.take_back(failed)
            for key, start in taken_back.items():
                del self._steps[key][start:]
            if taken_back:
                await self._flush()

    def serve(self, step_id: str, calls: list[ToolCall]) -> list[tuple[str, bool] | None]:
        return [None] * len(calls)

    async def capture(self, step_id: str, results: list[tuple[ToolCall, str, bool]]) -> None:
        async with self._lock:
            self._attempts.note(step_id, len(self._steps.get(step_id, [])))
            self._steps.setdefault(step_id, []).extend(
                {"name": call.name, "arguments": call.arguments, "result": text, "success": ok}
                for call, text, ok in results
            )
            await self._flush()

    async def _flush(self) -> None:
        """Write this run's steps over the file; call with ``_lock`` held."""
        steps = {key: list(calls) for key, calls in self._steps.items()}

        def _merge(merged: dict[str, Any]) -> None:
            merged["_tools"] = {**merged.get("_tools", {}), **steps}

        await anyio.to_thread.run_sync(merge_into_recording, self.path, _merge)


class ToolReplayer:
    """Serves (``replay``) or checks (``verify``) tool results from a recording.

    A call matches the next recorded call of its step with the same tool
    name and canonical arguments, so a tool called twice gets its two
    results in order. A retry serves again the calls that its failed
    attempt took, a subworkflow's included. ``strict`` raises
    ``RecordingMismatchError`` on a missing or changed call; otherwise a
    warning is logged and the live result is used.
    """

    def __init__(
        self,
        path: str | Path,
        mode: ToolMode,
        *,
        strict: bool = True,
        live: Callable[[str], bool] | None = None,
    ) -> None:
        text = Path(path).read_text(encoding="utf-8")
        recording = validate_recording_schema(json.loads(text), str(path))
        self._recorded: dict[str, list[dict[str, Any]]] = recording.get("_tools", {})
        self.mode = mode
        self.strict = strict
        # A step that the cut of a replay runs live (#165) runs its tools live too.
        self._live = live or (lambda _step_id: False)
        self._cursor: dict[str, int] = {}
        # Each step's cursor before an attempt first moved it: a retry goes back there.
        self._attempts = AttemptLog()
        self._numbers: dict[str, int] = {}

    async def begin_attempt(self, step_id: str, failed: Attempt | None) -> None:
        if failed is not None:
            self._cursor.update(self._attempts.take_back(failed))
        self._numbers[step_id] = 0

    def _match(self, step_id: str, call: ToolCall) -> tuple[str, dict[str, Any] | None]:
        """Name the call and find its recorded entry, moving the step's cursor past it."""
        number = self._numbers.get(step_id, 0) + 1
        self._numbers[step_id] = number
        where = f"Step {step_id!r}: tool call #{number} {call.name!r}"
        entries = self._recorded.get(step_id, [])
        arguments = _canonical(call.arguments)
        for i in range(self._cursor.get(step_id, 0), len(entries)):
            entry = entries[i]
            if entry["name"] == call.name and _canonical(entry["arguments"]) == arguments:
                self._attempts.note(step_id, self._cursor.get(step_id, 0))
                self._cursor[step_id] = i + 1
                return where, entry
        return where, None

    def _refuse(self, problem: str) -> None:
        if self.strict:
            raise RecordingMismatchError(
                f"{problem}. Re-record it with `agentloom run --record <file> --record-tools`."
            )
        logger.warning("%s; using the live result.", problem)

    def serve(self, step_id: str, calls: list[ToolCall]) -> list[tuple[str, bool] | None]:
        if self.mode is not ToolMode.REPLAY or self._live(step_id):
            return [None] * len(calls)
        served: list[tuple[str, bool] | None] = []
        for call in calls:
            where, entry = self._match(step_id, call)
            if entry is None:
                self._refuse(f"{where} is not in the recording")
                served.append(None)
            else:
                served.append((entry["result"], entry["success"]))
        return served

    async def capture(self, step_id: str, results: list[tuple[ToolCall, str, bool]]) -> None:
        if self.mode is not ToolMode.VERIFY or self._live(step_id):
            return
        for call, text, ok in results:
            where, entry = self._match(step_id, call)
            if entry is None:
                self._refuse(f"{where} is not in the recording")
            elif (entry["result"], entry["success"]) != (text, ok):
                self._refuse(
                    f"{where} returned {_digest(text)} (success={ok}), the recording has "
                    f"{_digest(entry['result'])} (success={entry['success']})"
                )
