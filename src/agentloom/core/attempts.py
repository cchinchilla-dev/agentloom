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
_IDS = itertools.count(1)


def running_attempt() -> Attempt:
    """The attempt the running code belongs to; ``()`` outside any step."""
    return _RUNNING.get()


def enter_attempt(outer: Attempt) -> Attempt:
    """Start a new attempt inside ``outer`` and make it the running one.

    The engine runs each step in a task of its own, so the mark ends with the step.
    """
    attempt = (*outer, next(_IDS))
    _RUNNING.set(attempt)
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
