"""The step attempt that the running code belongs to (#164).

Every attempt of a step gets an id of its own, and the engine marks the running
attempt as the chain of ids from the outermost step down to it. A subworkflow's
steps run inside their parent step's attempt, so their chains extend the
parent's. Recorders note what each attempt captured, so a retry can take back
everything its failed attempt recorded, the subworkflow's steps included.
"""

from __future__ import annotations

import itertools
from contextvars import ContextVar

Attempt = tuple[int, ...]
"""The ids of an attempt and of the attempts enclosing it, outermost first."""

_RUNNING: ContextVar[Attempt] = ContextVar("agentloom_attempt", default=())
_NUMBER: ContextVar[int] = ContextVar("agentloom_attempt_number", default=1)
_STEPS: ContextVar[tuple[str, ...]] = ContextVar("agentloom_steps", default=())
_IDS = itertools.count(1)


def running_attempt() -> Attempt:
    """The attempt the running code belongs to; ``()`` outside any step."""
    return _RUNNING.get()


def attempt_number() -> int:
    """Which attempt of its step the running one is, from 1; 1 outside any step (#166)."""
    return _NUMBER.get()


def running_steps() -> tuple[str, ...]:
    """The ids of the running step and of the steps enclosing it, outermost first (#165)."""
    return _STEPS.get()


def enter_step(step_id: str) -> None:
    """Mark ``step_id`` as running inside the steps already running, for the rest of its task."""
    _STEPS.set((*_STEPS.get(), step_id))


def enter_attempt(outer: Attempt, number: int = 1) -> Attempt:
    """Start attempt ``number`` of a step inside ``outer`` and make it the running one.

    The engine runs each step in a task of its own, so the mark ends with the step.
    """
    attempt = (*outer, next(_IDS))
    _RUNNING.set(attempt)
    _NUMBER.set(number)
    return attempt


class AttemptLog:
    """Where each key stood before an attempt first moved it, to take the attempt back."""

    def __init__(self) -> None:
        self._marks: dict[Attempt, dict[str, int]] = {}

    def note(self, key: str, position: int) -> None:
        """Record ``position`` for ``key`` unless the running attempt already moved it."""
        self._marks.setdefault(running_attempt(), {}).setdefault(key, position)

    def take_back(self, failed: Attempt) -> dict[str, int]:
        """Forget ``failed`` and the attempts inside it; return where their keys stood before."""
        positions: dict[str, int] = {}
        for attempt in [a for a in self._marks if a[: len(failed)] == failed]:
            for key, position in self._marks.pop(attempt).items():
                positions[key] = min(position, positions.get(key, position))
        return positions
