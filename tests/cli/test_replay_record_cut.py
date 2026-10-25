"""``agentloom replay --live-from/--force-live --record`` (#178)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from agentloom.cli.main import app
from tests.cli.test_replay import FETCH, TOOL_YAML, _tool_turn
from tests.cli.test_replay_cut import _providers, _recorded, _replay, runner


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_record_after_cut_copies_prefix_and_captures_tail(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    source = _load(recording)
    source["a"].update(latency_ms=1234.5, usage={**source["a"]["usage"], "reasoning_tokens": 7})
    recording.write_text(json.dumps(source))
    out = tmp_path / "out.json"

    result = _replay(tmp_path, workflow, recording, "--live-from", "b", "--record", str(out))

    assert result.exit_code == 0, result.output
    recorded = _load(out)
    assert recorded["a"] == source["a"]  # copied as recorded: latency and reasoning tokens kept
    assert recorded["b"]["content"] == "v2-b"
    assert recorded["b"]["request_hash"] == source["b"]["request_hash"]  # the same request, live


def test_record_never_writes_the_source_recording(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    before = recording.read_bytes()
    link = tmp_path / "link.json"
    link.symlink_to(recording)

    for target in (recording, link):
        result = _replay(
            tmp_path, workflow, recording, "--force-live", "a", "--record", str(target)
        )
        assert result.exit_code == 1
        assert "--record must name another file" in result.output
    out = tmp_path / "out.json"
    result = _replay(tmp_path, workflow, recording, "--force-live", "a", "--record", str(out))

    assert result.exit_code == 0, result.output
    assert recording.read_bytes() == before


def test_recording_of_a_cut_run_replays_strictly(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)
    out = tmp_path / "out.json"
    cut = _replay(tmp_path, workflow, recording, "--live-from", "b", "--record", str(out))
    assert cut.exit_code == 0, cut.output

    result = runner.invoke(app, ["replay", str(workflow), "-r", str(out), "--json"])

    assert result.exit_code == 0, result.output
    state = json.loads(result.output[result.output.index("{") :])["final_state"]
    assert (state["a"], state["b"]) == ("v1-a", "v2-b")  # the replayed prefix and the live tail


def test_record_needs_a_cut_and_record_tools_needs_record(tmp_path: Path) -> None:
    workflow, recording = _recorded(tmp_path)

    result = _replay(tmp_path, workflow, recording, "--record", str(tmp_path / "out.json"))
    assert result.exit_code == 1
    assert "--record needs --live-from or --force-live" in result.output

    result = _replay(tmp_path, workflow, recording, "--force-live", "a", "--record-tools")
    assert result.exit_code == 1
    assert "--record-tools needs --record" in result.output


AFTER = """\
  - id: after
    type: llm_call
    depends_on: [ask]
    prompt: "Sum up {state.answer}"
    output: summary
"""


def _tool_run(tmp_path: Path, page: str, *args: str) -> tuple[Any, AsyncMock]:
    """Run the fetch workflow, its page fetch answering ``page``."""
    answers = tmp_path / "answers.json"
    loop = [_tool_turn("", [{"id": "c1", **FETCH}]), _tool_turn("fetched", [])]
    answers.write_text(json.dumps({"ask": loop, "after": {"content": "summed"}}))
    with _providers(answers), patch("agentloom.steps._tools._execute", AsyncMock()) as fetch:
        fetch.return_value = (page, True)
        return runner.invoke(app, list(args)), fetch


def test_record_tools_keeps_served_results_and_captures_live_ones(tmp_path: Path) -> None:
    workflow = tmp_path / "wf.yaml"
    workflow.write_text(TOOL_YAML + AFTER)
    source = tmp_path / "src.json"
    record = ["run", str(workflow), "--record", str(source), "--record-tools", "--lite"]
    assert _tool_run(tmp_path, "<html>", *record)[0].exit_code == 0
    replay = ["replay", str(workflow), "-r", str(source), "--tools", "replay", "--record-tools"]

    o1, o2 = tmp_path / "o1.json", tmp_path / "o2.json"

    served, fetch = _tool_run(
        tmp_path, "<new>", *replay, "--live-from", "after", "--record", str(o1)
    )
    assert served.exit_code == 0, served.output
    fetch.assert_not_awaited()
    assert _load(o1)["_tools"]["ask"] == _load(source)["_tools"]["ask"]

    live, fetch = _tool_run(tmp_path, "<new>", *replay, "--force-live", "ask", "--record", str(o2))
    assert live.exit_code == 0, live.output
    fetch.assert_awaited_once()
    assert _load(o2)["_tools"]["ask"][0]["result"] == "<new>"
