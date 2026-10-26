"""The inter-agent event log of a run (#135).

While a log is active (``recording_events``), the engine, the tool calls and
the state manager write into it. For each step attempt: a ``task_sent`` from
every step it depends on that finished, its ``task_received``, its tool calls
and returns, and, when it succeeds, its ``response_sent`` and one
``state_transition`` per state key it wrote. An agent is the step's
``speaker``, or the step itself. A retry forgets what the failed attempt
logged, the steps of a subworkflow included, as the recorder forgets its turns.

Event ids derive from the step path, the event type and the position, so two
runs that do the same thing write the same log, and ``order_events`` sorts it
by the dependency graph, which the completion order of parallel steps does
not change.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from agentloom.contracts._inter_agent import (
    EVENT_ADAPTER,
    SCHEMA_URI,
    InterAgentEvent,
    ResponseSentEvent,
    StateTransitionEvent,
    TaskReceivedEvent,
    TaskSentEvent,
    ToolCallEvent,
    ToolReturnEvent,
)
from agentloom.core.attempts import running_steps

EVENTS_KEY = "_inter_agent_events"
"""Where a recording keeps the event log of the run it captured."""

StepPath = tuple[str, ...]
"""A step's id and the ids of the steps enclosing it, outermost first."""

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, SCHEMA_URI)
_ACTIVE: ContextVar[EventLog | None] = ContextVar("agentloom_inter_agent_log", default=None)


def payload_hash(value: Any) -> str:
    """SHA-256 of ``value`` as canonical JSON, 16-char prefix.

    Any state value or tool result hashes, so logging never fails a step: a
    mapping with keys of mixed types or a container that holds itself goes
    through ``_canonical``.
    """
    try:
        blob = _dumps(value)
    except (TypeError, ValueError):  # keys that do not sort, a circular reference
        blob = _dumps(_canonical(value, frozenset()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=_default, separators=(",", ":"))


def _default(value: Any) -> Any:
    """A set as its sorted items, as its iteration order changes between processes; else ``str``."""
    if isinstance(value, set | frozenset):
        return sorted(value, key=_dumps)
    return str(value)


def _canonical(value: Any, enclosing: frozenset[int]) -> Any:
    """``value`` as plain JSON: a mapping as sorted ``[key, value]`` pairs, cycles marked."""
    if not isinstance(value, dict | list | tuple | set | frozenset):
        return value if value is None or isinstance(value, str | int | float) else str(value)
    if id(value) in enclosing:
        return "<cycle>"
    enclosing = enclosing | {id(value)}
    if isinstance(value, dict):
        pairs = [[_canonical(k, enclosing), _canonical(v, enclosing)] for k, v in value.items()]
        return sorted(pairs, key=_dumps)  # the whole pair: keys can tie, as Path('x') and 'x' do
    items = [_canonical(item, enclosing) for item in value]
    return sorted(items, key=_dumps) if isinstance(value, set | frozenset) else items


def dump_events(events: Iterable[InterAgentEvent]) -> list[dict[str, Any]]:
    """``events`` as JSON-ready dicts, as a recording or a JSONL line holds them."""
    return [e.model_dump(mode="json") for e in events]


def load_events(raw: Iterable[Any]) -> list[InterAgentEvent]:
    """Validate the events of a recording; ``ValidationError`` on a malformed one."""
    return [EVENT_ADAPTER.validate_python(item) for item in raw]


def active_log() -> EventLog | None:
    """The log the running workflow writes into, if any."""
    return _ACTIVE.get()


def log_tools(calls: list[tuple[str, Any]], results: list[tuple[Any, bool]]) -> None:
    """Log a round of tool calls of the running step, if a log is active."""
    if (log := _ACTIVE.get()) is not None:
        log.tool_round(running_steps(), calls, results)


def log_write(key: str, value: Any) -> None:
    """Log a state write of the running step, if a log is active."""
    if (log := _ACTIVE.get()) is not None:
        log.write(running_steps(), key, value)


@contextmanager
def recording_events(log: EventLog) -> Iterator[EventLog]:
    """Make ``log`` the active one for the workflows run inside the block."""
    previous = _ACTIVE.set(log)
    try:
        yield log
    finally:
        _ACTIVE.reset(previous)


_TYPE_ORDER = (
    "task_sent",
    "task_received",
    "tool_call",
    "tool_return",
    "response_sent",
    "state_transition",
)


def order_events(events: Iterable[InterAgentEvent]) -> list[InterAgentEvent]:
    """``events`` in dependency-graph order: by causal depth, then step, type and ``event_id``.

    An event's depth is one more than its deepest parent's, so the order
    keeps every event after its parents and, at each depth, a step's events
    together; the completion order of parallel steps does not enter it.
    """
    by_id = {e.event_id: e for e in events}
    waiting = {i: {p for p in e.parent_event_ids if p in by_id} for i, e in by_id.items()}
    children: dict[uuid.UUID, list[uuid.UUID]] = {i: [] for i in by_id}
    for i, parents in waiting.items():
        for parent in parents:
            children[parent].append(i)
    depth = dict.fromkeys(by_id, 0)
    ready = [i for i, parents in waiting.items() if not parents]
    while ready:
        current = ready.pop()
        for child in children[current]:
            depth[child] = max(depth[child], depth[current] + 1)
            waiting[child].discard(current)
            if not waiting[child]:
                ready.append(child)
    return sorted(
        by_id.values(),
        key=lambda e: (depth[e.event_id], e.step_id, _TYPE_ORDER.index(e.type), str(e.event_id)),
    )


@dataclass
class _Step:
    """What one attempt of a step has logged so far."""

    agent: str
    events: list[InterAgentEvent] = field(default_factory=list)
    received: uuid.UUID | None = None
    tail: list[uuid.UUID] = field(default_factory=list)  # what the next event follows
    rounds: int = 0
    writes: dict[str, str] = field(default_factory=dict)
    output: str | None = None
    response: uuid.UUID | None = None


class EventLog:
    """The inter-agent events of one run."""

    def __init__(self) -> None:
        self._steps: dict[StepPath, _Step] = {}

    def events(self) -> list[InterAgentEvent]:
        """Every event logged, in dependency-graph order."""
        return order_events(e for step in self._steps.values() for e in step.events)

    def start_step(self, path: StepPath, agent: str, depends_on: Iterable[str]) -> None:
        """An attempt of the step at ``path`` starts after the steps it ``depends_on``."""
        for logged in [p for p in self._steps if p[: len(path)] == path]:
            del self._steps[logged]  # a retry: the failed attempt and its subworkflow steps
        step = self._steps[path] = _Step(agent)
        sent: list[TaskSentEvent] = []
        for upstream in sorted(depends_on):
            source = self._steps.get((*path[:-1], upstream))
            if source is None or source.response is None:
                continue  # skipped or failed: it handed nothing over
            sent.append(
                TaskSentEvent(
                    type="task_sent",
                    event_id=self._id(path, "task_sent", upstream),
                    agent_id=source.agent,
                    step_id=".".join((*path[:-1], upstream)),
                    recipient_agent_id=agent,
                    parent_event_ids=[source.response],
                    payload_hash=payload_hash({"output": source.output, "state": source.writes}),
                    timestamp_ns=time.monotonic_ns(),
                )
            )
        enclosing = self._steps.get(path[:-1])
        if sent:
            parents = sorted(e.event_id for e in sent)
        else:  # a root step: inside a subworkflow, it follows the subworkflow step
            parents = [enclosing.received] if enclosing and enclosing.received else []
        received = TaskReceivedEvent(
            type="task_received",
            event_id=self._id(path, "task_received"),
            agent_id=agent,
            step_id=".".join(path),
            parent_event_ids=parents,
            payload_hash=payload_hash(sorted(e.payload_hash for e in sent)),
            timestamp_ns=time.monotonic_ns(),
        )
        step.events.extend([*sent, received])
        step.received = received.event_id
        step.tail = [received.event_id]

    def tool_round(
        self, path: StepPath, calls: list[tuple[str, Any]], results: list[tuple[Any, bool]]
    ) -> None:
        """The step at ``path`` called tools at once: ``(name, arguments)`` and ``(result, ok)``."""
        step = self._steps.get(path)
        if step is None:
            return
        round_, step.rounds = step.rounds, step.rounds + 1
        returns: list[uuid.UUID] = []
        pairs = zip(calls, results, strict=True)
        for i, ((name, arguments), (result, success)) in enumerate(pairs):
            call = ToolCallEvent(
                type="tool_call",
                event_id=self._id(path, "tool_call", f"{round_}.{i}"),
                agent_id=step.agent,
                step_id=".".join(path),
                tool_name=name,
                parent_event_ids=sorted(step.tail),
                payload_hash=payload_hash(arguments),
                timestamp_ns=time.monotonic_ns(),
            )
            answer = ToolReturnEvent(
                type="tool_return",
                event_id=self._id(path, "tool_return", f"{round_}.{i}"),
                agent_id=step.agent,
                step_id=".".join(path),
                tool_name=name,
                parent_event_ids=[call.event_id],
                payload_hash=payload_hash({"result": result, "success": success}),
                timestamp_ns=time.monotonic_ns(),
            )
            step.events.extend([call, answer])
            returns.append(answer.event_id)
        step.tail = returns

    def write(self, path: StepPath, key: str, value: Any) -> None:
        """The step at ``path`` wrote ``value`` to the state ``key``."""
        step = self._steps.get(path)
        if step is not None:
            step.writes[key] = payload_hash(value)

    def finish_step(self, path: StepPath, output: Any) -> None:
        """The attempt of the step at ``path`` succeeded with ``output``."""
        step = self._steps.get(path)
        if step is None:
            return
        step.output = payload_hash(output)
        # A subworkflow step answers once its own steps have answered.
        inner = [s.response for p, s in self._steps.items() if p[:-1] == path and s.response]
        response = ResponseSentEvent(
            type="response_sent",
            event_id=self._id(path, "response_sent"),
            agent_id=step.agent,
            step_id=".".join(path),
            parent_event_ids=sorted({*step.tail, *inner}),
            payload_hash=step.output,
            timestamp_ns=time.monotonic_ns(),
        )
        step.events.append(response)
        step.response = response.event_id
        for key in sorted(step.writes):
            step.events.append(
                StateTransitionEvent(
                    type="state_transition",
                    event_id=self._id(path, "state_transition", key),
                    agent_id=step.agent,
                    step_id=".".join(path),
                    key=key,
                    parent_event_ids=[response.event_id],
                    payload_hash=step.writes[key],
                    timestamp_ns=time.monotonic_ns(),
                )
            )

    @staticmethod
    def _id(path: StepPath, kind: str, position: str = "") -> uuid.UUID:
        return uuid.uuid5(_NAMESPACE, f"{'/'.join(path)}|{kind}|{position}")
