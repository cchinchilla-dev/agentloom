"""``agentloom replay --live-from / --force-live`` (#165)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from agentloom.cli.main import app
from agentloom.providers.mock import MockProvider

runner = CliRunner()

WORKFLOW = """\
name: cut-cli
config:
  provider: openai
  model: gpt-4o-mini
steps:
  - id: a
    type: llm_call
    prompt: "A"
    output: a
  - id: b
    type: llm_call
    depends_on: [a]
    prompt: "B {state.a}"
    output: b
"""


def _answers(path: Path, version: str) -> Path:
    path.write_text(json.dumps({s: {"content": f"{version}-{s}"} for s in ("a", "b")}))
    return path


def _providers(answers: Path, asked: list[str] | None = None) -> Any:
    """Stand in for the workflow's providers, as `MockProvider` with fixed answers."""

    def _wire(gateway: Any, default: str) -> None:
        if asked is not None:
            asked.append(default)
        gateway.register(MockProvider(responses_file=answers), priority=0)

    return patch("agentloom.cli.run._setup_providers", side_effect=_wire)


def _recorded(tmp_path: Path) -> tuple[Path, Path]:
    workflow = tmp_path / "wf.yaml"
    workflow.write_text(WORKFLOW)
    recording = tmp_path / "rec.json"
    with _providers(_answers(tmp_path / "v1.json", "v1")):
        result = runner.invoke(app, ["run", str(workflow), "--record", str(recording), "--lite"])
    assert result.exit_code == 0, result.output
    return workflow, recording


def _replay(
    tmp_path: Path, workflow: Path, recording: Path, *flags: str, asked: list[str] | None = None
) -> Any:
    with _providers(_answers(tmp_path / "v2.json", "v2"), asked):
        return runner.invoke(app, ["replay", str(workflow), "-r", str(recording), *flags])


def test_live_from_lists_where_each_call_came_from(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    asked: list[str] = []
    result = _replay(tmp_path, workflow, recording, "--live-from", "b", "--json", asked=asked)

    assert result.exit_code == 0, result.output
    assert asked == ["openai"]  # the live part runs on the workflow's own provider
    data = json.loads(result.output[result.output.index("{") :])
    assert (data["final_state"]["a"], data["final_state"]["b"]) == ("v1-a", "v2-b")
    assert data["calls"] == [
        {"step_id": "a", "call": 1, "source": "replay", "reason": "hit"},
        {"step_id": "b", "call": 1, "source": "live", "reason": "forced"},
    ]


def test_force_live_prints_the_calls(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    result = _replay(tmp_path, workflow, recording, "--force-live", "a")

    assert result.exit_code == 0, result.output
    assert "  a #1: live (forced)" in result.output
    assert "  b #1: live (miss)" in result.output  # its prompt carries a's new answer


def test_unknown_cut_step_is_refused(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    result = _replay(tmp_path, workflow, recording, "--live-from", "x")

    assert result.exit_code == 1
    assert "No step 'x' in workflow 'cut-cli'" in result.output


def test_default_fallback_does_not_combine_with_a_cut(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    result = _replay(tmp_path, workflow, recording, "--force-live", "a", "--allow-default-fallback")

    assert result.exit_code == 1
    assert "--allow-default-fallback does not combine" in result.output


def test_a_forced_step_runs_its_tools_live(tmp_path: Path) -> None:
    from tests.cli.test_replay import FETCH, TOOL_YAML, _tool_turn

    answers = tmp_path / "answers.json"
    loop = [_tool_turn("", [{"id": "c1", **FETCH}]), _tool_turn("fetched", [])]
    answers.write_text(json.dumps({"ask": loop}))
    workflow = tmp_path / "wf.yaml"
    workflow.write_text(TOOL_YAML)
    recording = tmp_path / "rec.json"
    record = ["run", str(workflow), "--record", str(recording), "--record-tools", "--lite"]
    with _providers(answers), patch("agentloom.steps._tools._execute", AsyncMock()) as page:
        page.return_value = ("<html>", True)
        assert runner.invoke(app, record).exit_code == 0

    replay = ["replay", str(workflow), "-r", str(recording), "--tools", "replay"]
    with _providers(answers), patch("agentloom.steps._tools._execute", AsyncMock()) as page:
        page.return_value = ("<html>", True)
        result = runner.invoke(app, [*replay, "--force-live", "ask"])

    assert result.exit_code == 0, result.output
    page.assert_awaited_once()  # served from the recording, it would not have run
    assert "  ask #2: live (forced)" in result.output
