"""``agentloom verify-determinism`` and the event log ``--record`` writes (#135)."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from agentloom.cli.main import app
from agentloom.core.inter_agent import EVENTS_KEY, load_events
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry
from tests.cli.test_replay import FETCH, TOOL_YAML, _tool_turn
from tests.cli.test_replay_cut import _answers, _providers, _recorded, _replay, runner

WORKFLOW = """\
name: verify-cli
config:
  provider: openai
  model: gpt-4o-mini
steps:
  - id: a
    type: llm_call
    prompt: "A"
    output: a
  - id: t
    type: tool
    depends_on: [a]
    tool_name: answer
    output: t
  - id: b
    type: llm_call
    depends_on: [t]
    prompt: "B"
    output: b
"""


def _tools(answers: Iterator[Any]) -> Any:
    """Stand in for the built-in tools: one, ``answer``, that answers from ``answers``."""

    class _Answer(BaseTool):
        name = "answer"

        async def execute(self, **kwargs: Any) -> Any:
            return next(answers)

    def _registry(workflow: Any) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(_Answer())
        return registry

    return patch("agentloom.cli.run._tool_registry", side_effect=_registry)


def _record(tmp_path: Path, *flags: str, tool: str = "recorded") -> tuple[Path, Path]:
    workflow = tmp_path / "wf.yaml"
    workflow.write_text(WORKFLOW)
    recording = tmp_path / "rec.json"
    answers = _answers(tmp_path / "v1.json", "v1")
    with _providers(answers), _tools(itertools.repeat(tool)):
        argv = ["run", str(workflow), "--record", str(recording), "--lite", *flags]
        result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    return workflow, recording


def _verify(workflow: Path, recording: Path, *flags: str, tool: Iterator[Any]) -> Any:
    argv = ["verify-determinism", str(workflow), "-r", str(recording), *flags]
    with _tools(tool):
        return runner.invoke(app, argv)


def _logged(recording: Path) -> list[Any]:
    return load_events(json.loads(recording.read_text())[EVENTS_KEY])


def _untimed(event: Any) -> dict[str, Any]:
    return dict(event.model_dump(exclude={"timestamp_ns"}))


def test_run_record_writes_the_event_log(tmp_path: Path) -> None:
    _, recording = _record(tmp_path)
    events = _logged(recording)
    assert [(e.step_id, e.type) for e in events if e.type != "state_transition"] == [
        ("a", "task_received"),
        ("a", "response_sent"),
        ("a", "task_sent"),
        ("t", "task_received"),
        ("t", "tool_call"),
        ("t", "tool_return"),
        ("t", "response_sent"),
        ("t", "task_sent"),
        ("b", "task_received"),
        ("b", "response_sent"),
    ]
    assert json.loads(recording.read_text())["a"]["content"] == "v1-a"


def test_a_cut_replay_records_its_event_log(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    out = tmp_path / "out.json"
    result = _replay(tmp_path, workflow, recording, "--live-from", "b", "--record", str(out))
    assert result.exit_code == 0, result.output
    assert [e.step_id for e in _logged(out) if e.type == "response_sent"] == ["a", "b"]


def test_verify_cli_exits_zero_on_identical_runs(tmp_path: Path) -> None:
    workflow, recording = _record(tmp_path)
    events_out = tmp_path / "events.jsonl"
    result = _verify(
        workflow,
        recording,
        "--runs",
        "3",
        "--events-out",
        str(events_out),
        tool=itertools.repeat("recorded"),
    )
    assert result.exit_code == 0, result.output
    assert "Deterministic: 3 replays" in result.output
    assert "and the recorded event log (status success, 13 events)" in result.output
    lines = events_out.read_text().splitlines()
    written = load_events(json.loads(line) for line in lines)
    assert [_untimed(e) for e in written] == [_untimed(e) for e in _logged(recording)]


def test_verify_cli_exits_nonzero_on_divergence(tmp_path: Path) -> None:
    workflow, recording = _record(tmp_path)
    result = _verify(workflow, recording, tool=itertools.count())
    assert result.exit_code == 1
    assert "Replay 2 differs from replay 1:" in result.output
    assert '-      "output": 0,' in result.output
    assert '+      "output": 1,' in result.output


def test_verify_cli_compares_with_the_recorded_log(tmp_path: Path) -> None:
    workflow, recording = _record(tmp_path)
    result = _verify(workflow, recording, tool=itertools.repeat("changed"))
    assert result.exit_code == 1
    assert "The replayed event log differs from the recorded one:" in result.output
    assert '"tool_name": "answer", "type": "tool_return"' in result.output


def test_verify_cli_orders_the_recorded_log(tmp_path: Path) -> None:
    workflow, recording = _record(tmp_path)
    data = json.loads(recording.read_text())
    data[EVENTS_KEY].reverse()
    recording.write_text(json.dumps(data))
    result = _verify(workflow, recording, tool=itertools.repeat("recorded"))
    assert result.exit_code == 0, result.output


def _pages() -> Any:
    """A page fetch that answers another page on every call."""
    pages = [(f"<page {i}>", True) for i in range(4)]
    return patch("agentloom.steps._tools._execute", AsyncMock(side_effect=pages))


def test_verify_cli_replays_recorded_tools(tmp_path: Path) -> None:
    workflow = tmp_path / "wf.yaml"
    workflow.write_text(TOOL_YAML)
    answers = tmp_path / "answers.json"
    loop = [_tool_turn("", [{"id": "c1", **FETCH}]), _tool_turn("fetched", [])]
    answers.write_text(json.dumps({"ask": loop}))
    recording = tmp_path / "rec.json"
    with _providers(answers), _pages():
        argv = ["run", str(workflow), "--record", str(recording), "--record-tools", "--lite"]
        assert runner.invoke(app, argv).exit_code == 0
    verify = ["verify-determinism", str(workflow), "-r", str(recording)]
    with _pages():
        live = runner.invoke(app, verify)
    assert live.exit_code == 1  # the second replay fetches another page
    assert "Replay 2 differs from replay 1:" in live.output
    with _pages() as fetch:
        replayed = runner.invoke(app, [*verify, "--tools", "replay"])
    assert replayed.exit_code == 0, replayed.output
    fetch.assert_not_awaited()


def test_inter_agent_events_extend_recordings_without_breaking_v2_format(tmp_path: Path) -> None:
    workflow, recording = _record(tmp_path)
    with _tools(itertools.repeat("recorded")):
        replayed = runner.invoke(app, ["replay", str(workflow), "-r", str(recording)])
    assert replayed.exit_code == 0, replayed.output
    # A recording from before the event log: the replays are compared with each other only.
    data = json.loads(recording.read_text())
    del data[EVENTS_KEY]
    recording.write_text(json.dumps(data))
    result = _verify(workflow, recording, tool=itertools.repeat("recorded"))
    assert result.exit_code == 0, result.output
    assert "give the same result (status success, 13 events)" in result.output


def test_verify_cli_rejects_a_broken_recording(tmp_path: Path) -> None:
    workflow, recording = _record(tmp_path)
    cases = {
        "not json": "not a recording with a valid event log",
        json.dumps({EVENTS_KEY: [{"type": "task_lost"}]}): "not a recording with a valid event log",
        json.dumps({"a": "not a response"}): "Error: Recording",
    }
    for text, message in cases.items():
        recording.write_text(text)
        result = _verify(workflow, recording, tool=itertools.repeat("recorded"))
        assert result.exit_code == 1, text
        assert message in result.output, result.output
