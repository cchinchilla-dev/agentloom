"""``ReplayCut`` on its own: which calls go live, and the tools that follow them (#165)."""

from __future__ import annotations

import json
from pathlib import Path

from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.base import ToolCall
from agentloom.providers.mock import MockProvider
from agentloom.providers.replay_or_live import ReplayCut
from agentloom.providers.tool_tape import ToolMode, ToolReplayer

CALL = {"name": "clock", "arguments": {}, "result": "12:00", "success": True}


def _cut(tmp_path: Path, entries: dict[str, object], forced: set[str] | None = None) -> ReplayCut:
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(entries))
    return ReplayCut(MockProvider(responses_file=path, strict=True), forced or set())


async def test_a_step_that_missed_stays_live(tmp_path: Path) -> None:
    cut = _cut(tmp_path, {"s": [{"content": "one"}, {"content": "two"}]})

    async def recorded() -> str:
        return "recorded"

    async def changed() -> str:
        raise RecordingMismatchError("the request changed")

    async def live() -> str:
        return "live"

    assert await cut.route("s", recorded, live) == "recorded"
    assert await cut.route("s", changed, live) == "live"
    assert (
        await cut.route("s", recorded, live) == "live"
    )  # it would match, but s left the recording
    assert [(c.call, c.source, c.reason) for c in cut.calls] == [
        (1, "replay", "hit"),
        (2, "live", "miss"),
        (3, "live", "miss"),
    ]
    assert cut.unused_turns() == {}  # s ran live, so its unserved turns are not a change


def test_extra_calls_of_a_step_follow_it(tmp_path: Path) -> None:
    cut = _cut(tmp_path, {}, {"b"})

    assert cut.is_live("b::summarize")
    assert not cut.is_live("a::summarize")


def test_tools_of_a_live_step_run_live(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({"_tools": {"a": [CALL], "b": [CALL]}}))
    replayer = ToolReplayer(path, ToolMode.REPLAY, live=lambda step_id: step_id == "b")
    calls = [ToolCall(id="1", name="clock", arguments={})]

    assert replayer.serve("a", calls) == [("12:00", True)]
    assert replayer.serve("b", calls) == [None]


async def test_verify_skips_the_tools_of_a_live_step(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({"_tools": {"b": [CALL]}}))
    replayer = ToolReplayer(path, ToolMode.VERIFY, live=lambda step_id: step_id == "b")
    call = ToolCall(id="1", name="clock", arguments={})

    await replayer.capture("b", [(call, "13:00", True)])  # a different result, not checked
