"""The inter-agent event log (#135): its schema, its order and what a run writes into it."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from agentloom.contracts._inter_agent import SCHEMA_PATH, InterAgentEvent, event_schema
from agentloom.core.attempts import enter_step
from agentloom.core.engine import WorkflowEngine
from agentloom.core.inter_agent import (
    EventLog,
    dump_events,
    load_events,
    log_tools,
    order_events,
    payload_hash,
    recording_events,
)
from agentloom.core.models import WorkflowConfig, WorkflowDefinition
from agentloom.core.results import StepStatus, WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.providers.base import ProviderResponse
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry
from tests.conftest import MockProvider, MockTool
from tests.providers.test_tool_tape_retries import _AgentFailingOnce, _Clock, _run

CONFIG = WorkflowConfig(provider="mock", model="m")


def _comparable(events: list[InterAgentEvent]) -> list[dict[str, Any]]:
    """The events as two runs must agree on them: everything but the timestamp."""
    return [e.model_dump(mode="json", exclude={"timestamp_ns"}) for e in events]


def _kinds(events: list[InterAgentEvent]) -> list[tuple[str, str]]:
    return [(e.step_id, e.type) for e in events]


def _assert_causal(events: list[InterAgentEvent]) -> None:
    seen: set[Any] = set()
    for event in events:
        assert set(event.parent_event_ids) <= seen, event
        seen.add(event.event_id)


def _diamond_log(first: str, second: str) -> list[InterAgentEvent]:
    """``a`` hands over to ``b`` and ``c``, which hand over to ``d``; the middle in any order."""
    log = EventLog()
    log.start_step(("a",), "a", [])
    log.write(("a",), "a", "A")
    log.finish_step(("a",), "A")
    for step in (first, second):
        log.start_step((step,), step, ["a"])
    for step in (second, first):
        log.finish_step((step,), step.upper())
    log.start_step(("d",), "d", ["c", "b"])
    log.finish_step(("d",), "D")
    return log.events()


def _every_kind() -> list[dict[str, Any]]:
    """One logged event of each type, as a recording holds them."""
    log = EventLog()
    log.start_step(("a",), "alice", [])
    log.tool_round(("a",), [("search", {"q": "x"})], [("found", True)])
    log.write(("a",), "notes", "found")
    log.finish_step(("a",), "done")
    log.start_step(("b",), "bob", ["a"])
    events = dump_events(log.events())
    assert {e["type"] for e in events} == {
        "task_sent",
        "task_received",
        "tool_call",
        "tool_return",
        "response_sent",
        "state_transition",
    }
    return events


def test_jsonl_schema_validates() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    good = _every_kind()
    for event in good:
        validator.validate(event)
    sent = next(e for e in good if e["type"] == "task_sent")
    bad = [
        {**sent, "type": "task_lost"},
        {k: v for k, v in sent.items() if k != "recipient_agent_id"},
        {**sent, "payload_hash": "abc"},
        {**sent, "timestamp_ns": -1},
        {**sent, "parent_event_ids": sent["parent_event_ids"][0]},
        {**sent, "extra": 1},
    ]
    for event in bad:
        assert not validator.is_valid(event), event
        with pytest.raises(ValidationError):
            load_events([event])


def test_event_ids_must_be_uuid5() -> None:
    sent = next(e for e in _every_kind() if e["type"] == "task_sent")
    with pytest.raises(ValidationError):
        load_events([{**sent, "event_id": "6f1c9f8e-2b8e-4c3a-9f0e-0d6d5e2b7a10"}])


def test_committed_schema_matches_the_models() -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == event_schema(), "regenerate: python -m agentloom.contracts._inter_agent"


def test_dep_graph_ordering_deterministic() -> None:
    logged = _diamond_log("b", "c")
    assert _comparable(logged) == _comparable(_diamond_log("c", "b"))
    for seed in range(5):
        shuffled = list(logged)
        random.Random(seed).shuffle(shuffled)
        assert order_events(shuffled) == logged
    _assert_causal(logged)
    # By causal depth, then step, type and id: a sender's task_sent carries the sender's step.
    assert _kinds(logged) == [
        ("a", "task_received"),
        ("a", "response_sent"),
        ("a", "task_sent"),
        ("a", "task_sent"),
        ("a", "state_transition"),
        ("b", "task_received"),
        ("c", "task_received"),
        ("b", "response_sent"),
        ("c", "response_sent"),
        ("b", "task_sent"),
        ("c", "task_sent"),
        ("d", "task_received"),
        ("d", "response_sent"),
    ]


def test_a_step_that_skips_a_layer_follows_its_deepest_dependency() -> None:
    log = EventLog()
    for step, depends_on in (("a", []), ("b", ["a"]), ("c", ["b"]), ("d", ["a", "c"])):
        log.start_step((step,), step, depends_on)
        log.finish_step((step,), step)
    logged = log.events()
    _assert_causal(logged)
    for seed in range(10):
        shuffled = list(logged)
        random.Random(seed).shuffle(shuffled)
        assert order_events(shuffled) == logged


def test_a_handover_carries_the_sender_output_and_state_writes() -> None:
    events = _diamond_log("b", "c")
    by_kind = {(e.step_id, e.type, getattr(e, "recipient_agent_id", "")): e for e in events}
    sent = by_kind[("a", "task_sent", "b")]
    assert sent.agent_id == "a"
    assert sent.parent_event_ids == [by_kind[("a", "response_sent", "")].event_id]
    assert sent.payload_hash == payload_hash(
        {"output": payload_hash("A"), "state": {"a": payload_hash("A")}}
    )
    received = by_kind[("d", "task_received", "")]
    handed = [by_kind[("b", "task_sent", "d")], by_kind[("c", "task_sent", "d")]]
    assert received.parent_event_ids == sorted(e.event_id for e in handed)
    assert received.payload_hash == payload_hash(sorted(e.payload_hash for e in handed))
    assert by_kind[("a", "task_received", "")].parent_event_ids == []


def test_payload_hash_takes_any_value() -> None:
    mixed: dict[Any, Any] = {1: "a", "b": [2, (3, 4)], "p": Path("x")}
    reordered: dict[Any, Any] = {"p": Path("x"), "b": [2, (3, 4)], 1: "a"}
    assert payload_hash(mixed) == payload_hash(reordered)
    assert payload_hash({1: "a", "1": "b"}) != payload_hash({1: "b", "1": "a"})
    loop: list[Any] = []
    loop.append(loop)
    again: list[Any] = []
    again.append(again)
    assert payload_hash(loop) == payload_hash(again) == payload_hash(["<cycle>"])
    itself: dict[str, Any] = {}
    itself["self"] = itself
    assert payload_hash(itself) == payload_hash([["self", "<cycle>"]])
    shared = [1]  # twice, side by side, is no cycle
    assert payload_hash({1: shared, "x": shared}) == payload_hash({1: [1], "x": [1]})
    assert payload_hash({"b", "a"}) == payload_hash(["a", "b"])
    # Keys sort by their JSON: '"b"' before '1'
    assert payload_hash({1: 2.5, "b": None}) == payload_hash([["b", None], [1, 2.5]])
    assert payload_hash(Path("x")) == payload_hash("x")


def test_payload_hash_of_a_set_is_the_same_in_every_process() -> None:
    probe = (
        "from agentloom.core.inter_agent import payload_hash\n"
        "words = {'alpha', 'beta', 'gamma', 'delta', 'epsilon'}\n"
        "print(payload_hash(words), payload_hash({1: words, 'k': 0}))"
    )
    hashes = {
        subprocess.run(
            [sys.executable, "-c", probe],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(hashes) == 1


def test_a_failed_or_skipped_dependency_hands_nothing_over() -> None:
    log = EventLog()
    log.start_step(("a",), "a", [])  # fails: no finish_step
    log.start_step(("b",), "b", ["a", "skipped"])
    assert _kinds(log.events()) == [("a", "task_received"), ("b", "task_received")]


def test_a_retry_forgets_the_steps_of_its_subworkflow() -> None:
    log = EventLog()
    log.start_step(("sub",), "sub", [])
    log.start_step(("sub", "x"), "x", [])
    log.finish_step(("sub", "x"), "X")
    log.start_step(("sub",), "sub", [])  # the retry runs no step of the subworkflow yet
    assert _kinds(log.events()) == [("sub", "task_received")]


def test_tool_rounds_follow_one_another() -> None:
    log = EventLog()
    log.start_step(("a",), "a", [])
    log.tool_round(("a",), [("f", 1), ("g", 2)], [("F", True), ("G", True)])
    log.tool_round(("a",), [("f", 3)], [("F", True)])
    log.finish_step(("a",), "done")
    events = log.events()
    by_kind: dict[str, list[InterAgentEvent]] = {}
    for event in events:
        by_kind.setdefault(event.type, []).append(event)
    (received,) = by_kind["task_received"]
    first, second, third = by_kind["tool_call"]
    returns = {e.parent_event_ids[0]: e.event_id for e in by_kind["tool_return"]}
    assert first.parent_event_ids == second.parent_event_ids == [received.event_id]
    assert third.parent_event_ids == sorted([returns[first.event_id], returns[second.event_id]])
    assert by_kind["response_sent"][0].parent_event_ids == [returns[third.event_id]]
    assert len({e.event_id for e in events}) == len(events) == 8


async def test_an_atomic_update_is_a_state_transition() -> None:
    log = EventLog()
    state = StateManager()

    async def _step() -> None:
        enter_step("s")
        log.start_step(("s",), "s", [])
        await state.update("n", lambda n: (n or 0) + 1)
        log.finish_step(("s",), None)

    with recording_events(log):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_step)
    (transition,) = (e for e in log.events() if e.type == "state_transition")
    assert (transition.key, transition.payload_hash) == ("n", payload_hash(1))


def test_calls_outside_a_logged_step_are_ignored() -> None:
    log = EventLog()
    log.tool_round(("x",), [("t", {})], [("r", True)])
    log.write(("x",), "k", 1)
    log.finish_step(("x",), "out")
    assert log.events() == []


async def test_writes_and_tools_outside_a_step_are_not_logged() -> None:
    log = EventLog()
    with recording_events(log):
        await StateManager().set("k", 1)
        log_tools([("t", {})], [("r", True)])
    assert log.events() == []


class _Ordered(MockProvider):
    """Answers ``<prompt>!``; the prompts in ``order`` answer one after the other."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order
        self.done = {prompt: anyio.Event() for prompt in order}
        self.answered: list[str] = []

    async def complete(self, messages: list[dict[str, Any]], model: str, *_: Any, **__: Any) -> Any:
        prompt = messages[-1]["content"]
        for before in self.order[: self.order.index(prompt)] if prompt in self.order else []:
            await self.done[before].wait()
        self.answered.append(prompt)
        if prompt in self.done:
            self.done[prompt].set()
        return ProviderResponse(content=f"{prompt}!", model=model, provider="mock")


def _llm(step_id: str, *depends_on: str, **extra: Any) -> dict[str, Any]:
    step = {"id": step_id, "type": "llm_call", "prompt": step_id, "output": step_id}
    return {**step, "depends_on": list(depends_on), **extra}


async def _logged(
    workflow: WorkflowDefinition, provider: Any, registry: ToolRegistry | None = None
) -> tuple[Any, list[InterAgentEvent]]:
    gateway = ProviderGateway()
    gateway.register(provider)
    log = EventLog()
    with recording_events(log):
        engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, tool_registry=registry)
        result = await engine.run()
    await gateway.close()
    return result, log.events()


async def test_parallel_layers_write_the_same_log() -> None:
    steps = [_llm("a"), _llm("b", "a", speaker="bob"), _llm("c", "a"), _llm("d", "b", "c")]
    workflow = WorkflowDefinition(name="diamond", config=CONFIG, steps=steps)
    runs = []
    for order in (["b", "c"], ["c", "b"]):
        provider = _Ordered(order)
        result, events = await _logged(workflow, provider)
        assert result.status == WorkflowStatus.SUCCESS
        assert provider.answered[1:3] == order
        runs.append(events)
    assert _comparable(runs[0]) == _comparable(runs[1])
    _assert_causal(runs[0])
    sent = [e for e in runs[0] if e.type == "task_sent"]
    assert {(e.agent_id, e.recipient_agent_id) for e in sent} == {
        ("a", "bob"),
        ("a", "c"),
        ("bob", "d"),
        ("c", "d"),
    }
    transitions = {e.step_id: e for e in runs[0] if e.type == "state_transition"}
    assert transitions["b"].key == "b"
    assert transitions["b"].payload_hash == payload_hash("b!")


async def test_subworkflow_events_follow_their_step() -> None:
    child = {"name": "child", "config": CONFIG, "steps": [_llm("x"), _llm("y", "x")]}
    sub = {"id": "sub", "type": "subworkflow", "workflow_inline": child, "output": "sub"}
    workflow = WorkflowDefinition(
        name="parent", config=CONFIG, steps=[_llm("a"), {**sub, "depends_on": ["a"]}]
    )
    result, events = await _logged(workflow, _Ordered([]))
    assert result.status == WorkflowStatus.SUCCESS
    _assert_causal(events)
    by_kind = {(e.step_id, e.type): e for e in events if e.type != "task_sent"}
    received = by_kind[("sub", "task_received")]
    assert by_kind[("sub.x", "task_received")].parent_event_ids == [received.event_id]
    handover = next(e for e in events if e.type == "task_sent" and e.step_id == "sub.x")
    assert handover.recipient_agent_id == "y"
    answers = [by_kind[(s, "response_sent")].event_id for s in ("sub.x", "sub.y")]
    response = by_kind[("sub", "response_sent")]
    assert response.parent_event_ids == sorted([received.event_id, *answers])
    assert by_kind[("sub.y", "state_transition")].key == "y"
    assert by_kind[("sub", "state_transition")].key == "sub"


@pytest.mark.parametrize("nested", [False, True], ids=["step", "subworkflow"])
async def test_a_retry_drops_what_the_failed_attempt_logged(nested: bool) -> None:
    log = EventLog()
    with recording_events(log):
        result = await _run(_AgentFailingOnce(), _Clock(), None, nested)
    assert result.status == WorkflowStatus.SUCCESS
    events = log.events()
    _assert_causal(events)
    step = "sub.s" if nested else "s"
    assert [e.type for e in events if e.step_id == step] == [
        "task_received",
        "tool_call",
        "tool_return",
        "response_sent",
        "state_transition",
    ]
    answer = next(e for e in events if e.type == "tool_return")
    assert answer.tool_name == "clock"
    assert answer.payload_hash == payload_hash({"result": "10:02", "success": True})


class _Mixed(BaseTool):
    """Answers a mapping that JSON cannot sort."""

    name = "mixed"

    async def execute(self, **kwargs: Any) -> Any:
        return {1: "a", "b": 2}


class _Broken(BaseTool):
    name = "broken"

    async def execute(self, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("tool", "answer", "status"),
    [
        (MockTool(), {"result": "mock_result", "success": True}, StepStatus.SUCCESS),
        (_Broken(), {"result": "boom", "success": False}, StepStatus.FAILED),
        (_Mixed(), {"result": {1: "a", "b": 2}, "success": True}, StepStatus.SUCCESS),
    ],
    ids=["ok", "raises", "mixed-keys"],
)
async def test_a_tool_step_logs_its_call(tool: BaseTool, answer: Any, status: StepStatus) -> None:
    registry = ToolRegistry()
    registry.register(tool)
    step = {"id": "t", "type": "tool", "tool_name": tool.name, "tool_args": {"input": "hi"}}
    step.update(output="t", retry={"max_retries": 0})
    workflow = WorkflowDefinition(name="tools", config=CONFIG, steps=[step])
    result, events = await _logged(workflow, MockProvider(), registry)
    assert result.step_results["t"].status == status
    call, back = (e for e in events if e.type in ("tool_call", "tool_return"))
    assert call.payload_hash == payload_hash({"input": "hi"})
    assert back.parent_event_ids == [call.event_id]
    assert back.payload_hash == payload_hash(answer)
    assert (("t", "response_sent") in _kinds(events)) == (status == StepStatus.SUCCESS)
