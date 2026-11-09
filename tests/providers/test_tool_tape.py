"""Tool results in a recording: capture them, replay them offline, verify them."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anyio
import pytest

from agentloom.core.attempts import enter_step
from agentloom.core.engine import WorkflowEngine
from agentloom.core.redact import RedactionPolicy, is_redacted
from agentloom.core.results import StepStatus, WorkflowResult, WorkflowStatus
from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.base import BaseProvider, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider
from agentloom.providers.tool_tape import ToolMode, ToolRecorder, ToolReplayer, ToolTape
from agentloom.steps._tools import dispatch_tool_calls
from agentloom.tools.registry import ToolRegistry
from tests.providers.test_recorder_tool_loop import ADD_LOOP, _add_workflow, _AddTool

ADD = ToolCall(id="c1", name="add", arguments={"b": 3, "a": 2})


def _entry(result: str, success: bool = True) -> dict[str, Any]:
    return {"name": "add", "arguments": {"a": 2, "b": 3}, "result": result, "success": success}


def _recording(tmp_path: Path, tools: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "rec.json"
    raw: dict[str, Any] = {"_version": 2, "ask": ADD_LOOP}
    if tools is not None:
        raw["_tools"] = tools
    path.write_text(json.dumps(raw))
    return path


def _registry(tool: _AddTool, tape: ToolTape | None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    registry.tape = tape
    return registry


async def _run(provider: BaseProvider, tool: _AddTool, tape: ToolTape | None) -> WorkflowResult:
    gateway = ProviderGateway()
    gateway.register(provider)
    engine = WorkflowEngine(
        workflow=_add_workflow(), provider_gateway=gateway, tool_registry=_registry(tool, tape)
    )
    result = await engine.run()
    await gateway.close()
    return result


async def _record(tmp_path: Path) -> Path:
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"ask": ADD_LOOP}))
    path = tmp_path / "rec.json"
    provider = RecordingProvider(MockProvider(responses_file=src), path)
    result = await _run(provider, _AddTool(), ToolRecorder(path))
    assert result.status == WorkflowStatus.SUCCESS
    return path


async def _replay(path: Path, mode: ToolMode, tool: _AddTool) -> WorkflowResult:
    provider = MockProvider(responses_file=path, strict=True)
    return await _run(provider, tool, ToolReplayer(path, mode))


async def test_tool_results_round_trip(tmp_path: Path) -> None:
    path = await _record(tmp_path)
    data = json.loads(path.read_text())
    assert data["_tools"] == {"ask": [_entry("5")]}
    assert len(data["ask"]) == 2  # the LLM turns sit next to them

    tool = _AddTool()
    replayed = await _replay(path, ToolMode.REPLAY, tool)
    assert replayed.status == WorkflowStatus.SUCCESS
    assert replayed.final_state["answer"] == "The answer is 5."
    assert tool.calls == []


async def test_replay_mode_does_not_execute_tools(tmp_path: Path) -> None:
    path = await _record(tmp_path)
    drifted = _AddTool(offset=1)  # would answer 6 if it ran
    replayed = await _replay(path, ToolMode.REPLAY, drifted)
    assert replayed.status == WorkflowStatus.SUCCESS
    assert drifted.calls == []


async def test_verify_mode_runs_tools_and_passes_when_results_match(tmp_path: Path) -> None:
    path = await _record(tmp_path)
    tool = _AddTool()
    verified = await _replay(path, ToolMode.VERIFY, tool)
    assert verified.status == WorkflowStatus.SUCCESS
    assert tool.calls == [{"a": 2, "b": 3}]


async def test_verify_mode_fails_on_result_drift_and_names_the_call(tmp_path: Path) -> None:
    path = await _record(tmp_path)
    drifted = _AddTool(offset=1)
    verified = await _replay(path, ToolMode.VERIFY, drifted)
    step = verified.step_results["ask"]
    assert (step.status, step.error_classification) == (StepStatus.FAILED, "permanent")
    assert "Step 'ask': tool call #1 'add' returned" in (step.error or "")
    assert "the recording has" in (step.error or "")
    assert drifted.calls == [{"a": 2, "b": 3}]


async def test_recordings_without_tools_section_load_unchanged(tmp_path: Path) -> None:
    # A recording without a ``_tools`` section holds LLM turns only: it replays with live tools.
    tool = _AddTool()
    provider = MockProvider(responses_file=_recording(tmp_path), strict=True)
    replayed = await _run(provider, tool, None)
    assert replayed.status == WorkflowStatus.SUCCESS
    assert tool.calls == [{"a": 2, "b": 3}]


async def test_strict_replay_fails_on_a_call_missing_from_the_recording(tmp_path: Path) -> None:
    tool = _AddTool()
    replayed = await _replay(_recording(tmp_path), ToolMode.REPLAY, tool)
    step = replayed.step_results["ask"]
    assert step.status == StepStatus.FAILED
    assert "Step 'ask': tool call #1 'add' is not in the recording" in (step.error or "")
    assert tool.calls == []


async def test_lenient_replay_runs_a_missing_call_live(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tape = ToolReplayer(_recording(tmp_path, {}), ToolMode.REPLAY, strict=False)
    with caplog.at_level(logging.WARNING, logger="agentloom.providers.tool_tape"):
        results = await dispatch_tool_calls([ADD], _registry(_AddTool(), tape), step_id="ask")
    assert results == [(ADD, "5", True)]
    assert "is not in the recording; using the live result" in caplog.text


async def test_lenient_verify_keeps_the_live_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tape = ToolReplayer(_recording(tmp_path, {"s": [_entry("5")]}), ToolMode.VERIFY, strict=False)
    with caplog.at_level(logging.WARNING, logger="agentloom.providers.tool_tape"):
        results = await dispatch_tool_calls([ADD], _registry(_AddTool(offset=1), tape), step_id="s")
    assert results == [(ADD, "6", True)]
    assert "tool call #1 'add' returned" in caplog.text


async def test_verify_fails_on_a_call_missing_from_the_recording(tmp_path: Path) -> None:
    tape = ToolReplayer(_recording(tmp_path, {}), ToolMode.VERIFY)
    with pytest.raises(RecordingMismatchError, match=r"tool call #1 'add' is not in the recording"):
        await dispatch_tool_calls([ADD], _registry(_AddTool(), tape), step_id="s")


async def test_rerecording_a_step_replaces_only_its_tool_calls(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    earlier = {"old": [_entry("1")], "s": [_entry("9")]}
    path.write_text(json.dumps({"_version": 2, "llm": {"content": "x"}, "_tools": earlier}))
    recorder = ToolRecorder(path)
    assert recorder.serve("s", [ADD]) == [None]
    await recorder.capture("s", [(ADD, "5", True)])
    await recorder.capture("s", [(ADD, "boom", False)])

    data = json.loads(path.read_text())
    assert data["llm"] == {"content": "x"}
    assert data["_tools"] == {"old": [_entry("1")], "s": [_entry("5"), _entry("boom", False)]}


async def _in_step(*steps: str, call: Any) -> None:
    """Run ``call`` as the innermost of ``steps``; its own task keeps the marks to itself."""

    async def _run() -> None:
        for step in steps:
            enter_step(step)
        await call()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run)


class TestSubworkflowSteps:
    async def test_a_child_step_records_under_its_path(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.json"
        recorder = ToolRecorder(path)
        await _in_step("sub", "ask", call=lambda: recorder.capture("ask", [(ADD, "5", True)]))
        await _in_step("ask", call=lambda: recorder.capture("ask", [(ADD, "6", True)]))
        tools = json.loads(path.read_text())["_tools"]
        assert [e["result"] for e in tools["sub/ask"]] == ["5"]
        assert [e["result"] for e in tools["ask"]] == ["6"]

    async def test_a_child_step_is_served_from_its_path(self, tmp_path: Path) -> None:
        path = _recording(tmp_path, {"sub/ask": [_entry("5")], "ask": [_entry("6")]})
        replayer = ToolReplayer(path, ToolMode.REPLAY)
        served: list[Any] = []

        async def _serve() -> None:
            served.extend(replayer.serve("ask", [ADD]))

        await _in_step("sub", "ask", call=_serve)
        await _in_step("ask", call=_serve)
        assert served == [("5", True), ("6", True)]

    async def test_a_recording_with_the_bare_id_still_serves_a_child_step(
        self, tmp_path: Path
    ) -> None:
        path = _recording(tmp_path, {"ask": [_entry("5")]})
        replayer = ToolReplayer(path, ToolMode.REPLAY)
        served: list[Any] = []

        async def _serve() -> None:
            served.extend(replayer.serve("ask", [ADD]))

        await _in_step("sub", "ask", call=_serve)
        assert served == [("5", True)]


class TestRedaction:
    POLICY = RedactionPolicy(["*token*"])
    CALL = ToolCall(id="c1", name="fetch", arguments={"api_token": "dummy-secret", "url": "u"})

    async def _captured(self, tmp_path: Path, result: str) -> dict[str, Any]:
        path = tmp_path / "rec.json"
        recorder = ToolRecorder(path, redaction=self.POLICY)
        await recorder.capture("ask", [(self.CALL, result, True)])
        return json.loads(path.read_text())["_tools"]["ask"][0]

    async def test_arguments_are_masked(self, tmp_path: Path) -> None:
        entry = await self._captured(tmp_path, "done")
        assert is_redacted(entry["arguments"]["api_token"])
        assert entry["arguments"]["url"] == "u"
        assert entry["result"] == "done"

    async def test_a_json_result_is_masked(self, tmp_path: Path) -> None:
        entry = await self._captured(tmp_path, json.dumps({"api_token": "dummy-secret"}))
        assert "dummy-secret" not in entry["result"]

    async def test_a_masked_call_is_served_on_replay(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.json"
        await ToolRecorder(path, redaction=self.POLICY).capture("ask", [(self.CALL, "done", True)])
        replayed_call = self.CALL.model_copy(
            update={"arguments": json.loads(path.read_text())["_tools"]["ask"][0]["arguments"]}
        )
        replayer = ToolReplayer(path, ToolMode.REPLAY)
        assert replayer.serve("ask", [replayed_call]) == [("done", True)]

    async def test_without_a_policy_they_stay(self, tmp_path: Path) -> None:
        path = tmp_path / "rec.json"
        await ToolRecorder(path).capture("ask", [(self.CALL, "done", True)])
        assert "dummy-secret" in path.read_text()
