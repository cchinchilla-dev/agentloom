"""Retries and the tool tape (#164): a failed attempt leaves no trace in recording or replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentloom.core.attempts import Attempt, enter_attempt
from agentloom.core.models import (
    RetryConfig,
    StepDefinition,
    StepType,
    ToolDefinition,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowStatus
from agentloom.exceptions import ProviderError, RecordingMismatchError
from agentloom.providers.base import ProviderResponse, ToolCall
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider
from agentloom.providers.tool_tape import ToolMode, ToolRecorder, ToolReplayer
from agentloom.steps._tools import dispatch_tool_calls
from agentloom.tools.base import BaseTool
from tests.providers.test_tool_tape import ADD, _AddTool, _entry, _recording, _registry

CLOCK = ToolDefinition(name="clock", description="time", parameters={"type": "object"})


class _Clock(BaseTool):
    """A tool whose answer changes on every call."""

    name = "clock"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: Any) -> Any:
        self.calls += 1
        return f"10:0{self.calls}"


class _AgentFailingOnce(MockProvider):
    """Asks for the clock, answers once it has the time; the second request fails, as a 503.

    Each request gets a fresh tool call id, as a real provider gives.
    """

    requests = 0

    async def complete(self, messages: list[dict[str, Any]], model: str, *_: Any, **__: Any) -> Any:
        self.requests += 1
        if self.requests == 2:
            raise ProviderError("mock", "unavailable", status_code=503)
        if messages[-1]["role"] == "tool":
            answer = f"It is {messages[-1]['content']}."
            return ProviderResponse(content=answer, model=model, provider="mock")
        call = ToolCall(id=f"call-{self.requests}", name="clock", arguments={})
        return ProviderResponse(content="", model=model, provider="mock", tool_calls=[call])


def _workflow(nested: bool) -> WorkflowDefinition:
    """Step ``s`` asks for the clock; ``nested`` runs it in a subworkflow, ``sub``."""
    config = WorkflowConfig(provider="mock", model="m")
    retry = RetryConfig(max_retries=2, backoff_max=0.0, jitter=False)
    step = {"id": "s", "type": "llm_call", "prompt": "What time is it?", "output": "out"}
    step["tools"] = [CLOCK.model_dump()]
    if not nested:
        return WorkflowDefinition(name="w", config=config, steps=[{**step, "retry": retry}])
    # The child step does not retry: its failure fails the child, and the parent retries it all.
    child = {"name": "child", "config": config, "steps": [{**step, "retry": {"max_retries": 0}}]}
    parent = StepDefinition(
        id="sub", type=StepType.SUBWORKFLOW, workflow_inline=child, output="sub", retry=retry
    )
    return WorkflowDefinition(name="w", config=config, steps=[parent])


async def _run(provider: Any, tool: _Clock, tape: Any, nested: bool = False) -> Any:
    from agentloom.core.engine import WorkflowEngine
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.tools.registry import ToolRegistry

    workflow = _workflow(nested)
    gateway = ProviderGateway()
    gateway.register(provider)
    registry = ToolRegistry()
    registry.register(tool)
    registry.tape = tape
    result = await WorkflowEngine(
        workflow=workflow, provider_gateway=gateway, tool_registry=registry
    ).run()
    await gateway.close()
    return result


def _out(result: Any, nested: bool) -> Any:
    return (result.final_state["sub"] if nested else result.final_state)["out"]


@pytest.mark.parametrize("nested", [False, True], ids=["step", "subworkflow"])
async def test_a_retry_while_recording_keeps_only_the_finished_attempt(
    tmp_path: Path, nested: bool
) -> None:
    path = tmp_path / "rec.json"
    recorder = RecordingProvider(_AgentFailingOnce(), path)
    recorded = await _run(recorder, _Clock(), ToolRecorder(path), nested)
    assert _out(recorded, nested) == "It is 10:02."
    recording = json.loads(path.read_text())
    assert recording["_tools"] == {
        "s": [{"name": "clock", "arguments": {}, "result": "10:02", "success": True}]
    }
    # The LLM turns of the failed attempt are gone too: the retry's ask and its answer remain.
    turns = [
        t["tool_calls"][0]["id"] if "tool_calls" in t else t["content"] for t in recording["s"]
    ]
    assert turns == ["call-3", "It is 10:02."]

    clock = _Clock()
    replay = MockProvider(responses_file=path, strict=True)
    replayed = await _run(replay, clock, ToolReplayer(path, ToolMode.REPLAY), nested)
    assert replayed.status == WorkflowStatus.SUCCESS
    assert _out(replayed, nested) == "It is 10:02."
    assert clock.calls == 0


async def _attempt(tape: Any, failed: Attempt | None = None, outer: Attempt = ()) -> Attempt:
    """Start an attempt of step ``s`` the way the engine does; ``failed`` makes it a retry."""
    running = enter_attempt(outer)
    await tape.begin_attempt("s", failed)
    return running


async def test_a_retry_drops_only_the_calls_of_its_own_attempt(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    recorder = ToolRecorder(path)
    await _attempt(recorder)
    await recorder.capture("s", [(ADD, "1", True)])
    second = await _attempt(recorder)  # the step runs again, in a loop
    await recorder.capture("s", [(ADD, "2", True)])
    await _attempt(recorder, second)  # that second run failed and is retried
    assert json.loads(path.read_text())["_tools"]["s"] == [_entry("1")]
    await recorder.capture("s", [(ADD, "3", True)])
    assert json.loads(path.read_text())["_tools"]["s"] == [_entry("1"), _entry("3")]


async def test_a_retried_step_replays_the_calls_of_its_failed_attempt_again(
    tmp_path: Path,
) -> None:
    recorded = {"s": [_entry("5"), _entry("6"), _entry("7")]}
    tape = ToolReplayer(_recording(tmp_path, recorded), ToolMode.REPLAY)
    registry = _registry(_AddTool(), tape)

    async def _call() -> str:
        return (await dispatch_tool_calls([ADD], registry, step_id="s"))[0][1]

    await _attempt(tape)
    first = await _call()
    retried = await _attempt(tape)  # the step runs again, in a loop
    second = await _call()
    await _attempt(tape, retried)  # that second run is retried
    assert [first, second, await _call(), await _call()] == ["5", "6", "6", "7"]
    # Past the recorded calls nothing wraps around: the next one is missing.
    with pytest.raises(RecordingMismatchError, match=r"tool call #3 'add' is not in the recording"):
        await _call()


async def test_a_retried_subworkflow_replays_its_steps_calls_again(tmp_path: Path) -> None:
    recorded = {"s": [_entry("5"), _entry("6")]}
    tape = ToolReplayer(_recording(tmp_path, recorded), ToolMode.REPLAY)
    registry = _registry(_AddTool(), tape)

    async def _call() -> str:
        return (await dispatch_tool_calls([ADD], registry, step_id="s"))[0][1]

    parent = enter_attempt(())  # the subworkflow step's attempt; ``s`` runs inside it
    await _attempt(tape, outer=parent)
    assert await _call() == "5"
    retry = enter_attempt(())  # the subworkflow step failed and is retried
    await tape.begin_attempt("sub", parent)
    await _attempt(tape, outer=retry)
    assert [await _call(), await _call()] == ["5", "6"]
