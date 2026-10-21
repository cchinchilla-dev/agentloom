"""Recording a tool loop: every turn, with its tool calls, replays in order."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    StepDefinition,
    StepType,
    ToolDefinition,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowStatus
from agentloom.providers.base import BaseProvider, StreamResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider, prompt_hash
from agentloom.providers.recorder import RecordingProvider
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry


def _turn(content: str, *calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """A recorded turn in the shape ``MockProvider`` reads."""
    turn: dict[str, Any] = {
        "content": content,
        "model": "m",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "cost_usd": 0.0,
        "finish_reason": "tool_calls" if calls else "stop",
    }
    if calls:
        turn["tool_calls"] = [{"id": i, "name": n, "arguments": a} for i, n, a in calls]
    return turn


async def _tool_loop(provider: BaseProvider, turns: int) -> list[Any]:
    """Call ``provider`` as a tool loop does: same step id, a longer history each turn."""
    responses = []
    for i in range(turns):
        tool_turns = [{"role": "tool", "content": str(j)} for j in range(i)]
        messages = [{"role": "user", "content": "go"}, *tool_turns]
        responses.append(await provider.complete(messages=messages, model="m", step_id="loop"))
    return responses


async def test_tool_loop_records_every_turn_with_its_tool_calls(tmp_path):
    src = tmp_path / "src.json"
    loop = [
        _turn("", ("c1", "search", {"q": "a"})),
        _turn("", ("c2", "search", {"q": "b"}), ("c3", "fetch", {"url": "u"})),
        _turn("done"),
    ]
    src.write_text(json.dumps({"loop": loop}))
    recording = tmp_path / "rec.json"
    live = await _tool_loop(RecordingProvider(MockProvider(responses_file=src), recording), 3)

    data = json.loads(recording.read_text())
    assert [t.get("tool_calls") for t in data["loop"]] == [t.get("tool_calls") for t in loop]
    assert {t["provider"] for t in data["loop"]} == {"mock"}

    replay = MockProvider(responses_file=recording, strict=True)
    replayed = await _tool_loop(replay, 3)
    assert [r.tool_calls for r in replayed] == [r.tool_calls for r in live]
    assert replayed[-1].content == "done"


async def test_new_run_replaces_turns_of_an_earlier_run(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"loop": [_turn("", ("c1", "t", {})), _turn("end")]}))
    recording = tmp_path / "rec.json"
    for _ in range(2):  # two runs, two turns each
        await _tool_loop(RecordingProvider(MockProvider(responses_file=src), recording), 2)
    assert len(json.loads(recording.read_text())["loop"]) == 2


async def test_single_turn_step_and_hash_keys_stay_plain_entries(tmp_path):
    recording = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(default_response="ok"), recording)
    messages = [{"role": "user", "content": "same"}]
    await recorder.complete(messages=messages, model="m", step_id="once")
    await recorder.complete(messages=messages, model="m")
    await recorder.complete(messages=messages, model="m")  # same request, no step id

    data = json.loads(recording.read_text())
    assert isinstance(data["once"], dict)
    assert "tool_calls" not in data["once"]
    assert isinstance(data[prompt_hash(messages, "m")], dict)


async def test_stream_keeps_tool_calls_for_the_caller_and_the_recording(tmp_path):
    class FakeToolStreamingProvider:
        """Streams nothing but a tool call; the recorder only touches ``stream``."""

        name, api_key, base_url = "fake", "", ""

        async def stream(self, **kwargs) -> StreamResponse:
            sr = StreamResponse(model=kwargs["model"], provider="fake")

            async def _gen() -> AsyncIterator[str]:
                yield ""
                sr.tool_calls = [ToolCall(id="c1", name="search", arguments={"q": "x"})]
                sr.reasoning_content = "thinking"
                sr.finish_reason = "tool_calls"

            sr._set_iterator(_gen())
            return sr

    recording = tmp_path / "r.json"
    recorder = RecordingProvider(FakeToolStreamingProvider(), recording)  # type: ignore[arg-type]
    sr = await recorder.stream(messages=[{"role": "user", "content": "hi"}], model="m", step_id="s")
    _ = [c async for c in sr]

    assert [c.name for c in sr.tool_calls] == ["search"]
    assert sr.reasoning_content == "thinking"
    entry = json.loads(recording.read_text())["s"]
    assert entry["tool_calls"] == [{"id": "c1", "name": "search", "arguments": {"q": "x"}}]


class _AddTool(BaseTool):
    name = "add"
    description = "Add two integers."
    parameters_schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset  # a non-zero offset stands for a tool whose output changed
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return kwargs["a"] + kwargs["b"] + self.offset


ADD_LOOP = [_turn("", ("c1", "add", {"a": 2, "b": 3})), _turn("The answer is 5.")]


def _add_workflow(**step: Any) -> WorkflowDefinition:
    add = ToolDefinition(
        name="add", description="add two integers", parameters=_AddTool.parameters_schema
    )
    ask = StepDefinition(
        id="ask",
        type=StepType.LLM_CALL,
        prompt="What is 2+3?",
        tools=[add],
        output="answer",
        **step,
    )
    return WorkflowDefinition(
        name="agent", config=WorkflowConfig(provider="mock", model="m"), state={}, steps=[ask]
    )


async def _run(workflow: WorkflowDefinition, provider: BaseProvider, tool: _AddTool) -> Any:
    gateway = ProviderGateway()
    gateway.register(provider)
    registry = ToolRegistry()
    registry.register(tool)
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, tool_registry=registry)
    result = await engine.run()
    await gateway.close()
    return result


async def test_tool_workflow_records_and_replays_offline(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"ask": ADD_LOOP}))
    recording = tmp_path / "rec.json"
    workflow, live_tool, replay_tool = _add_workflow(), _AddTool(), _AddTool()
    live = await _run(
        workflow, RecordingProvider(MockProvider(responses_file=src), recording), live_tool
    )
    replay_provider = MockProvider(responses_file=recording, strict=True)
    replayed = await _run(workflow, replay_provider, replay_tool)

    assert replayed.status == WorkflowStatus.SUCCESS
    assert replayed.final_state["answer"] == live.final_state["answer"] == "The answer is 5."
    assert replay_tool.calls == live_tool.calls == [{"a": 2, "b": 3}]
    assert len(replay_provider.calls) == 2
