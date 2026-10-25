"""Re-recording replayed calls copies the recorded turns as they were (#178)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.providers.mock import MockProvider, prompt_hash
from agentloom.providers.recorder import RecordingProvider
from agentloom.providers.replay_or_live import ReplayCut, ReplayOrLiveProvider
from agentloom.providers.tool_tape import ToolMode, ToolRecorder, ToolReplayer
from tests.providers.test_replay_or_live import _Versioned
from tests.providers.test_tool_tape import ADD, _entry, _recording
from tests.providers.test_tool_tape_retries import _attempt


def _ask(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _turn(text: str, answer: str) -> dict[str, Any]:
    return {
        "content": answer,
        "model": "m",
        "provider": "v1",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "reasoning_tokens": 7,
        },
        "cost_usd": 0.25,
        "latency_ms": 812.5,
        "finish_reason": "stop",
        "request_hash": prompt_hash(_ask(text), "m", None, None, None),
    }


def _source(tmp_path: Path, turns: list[dict[str, Any]]) -> MockProvider:
    path = tmp_path / "src.json"
    path.write_text(json.dumps({"s": turns if len(turns) > 1 else turns[0]}))
    return MockProvider(responses_file=path, strict=True)


async def test_replay_restores_reasoning_tokens(tmp_path: Path) -> None:
    response = await _source(tmp_path, [_turn("one", "r1")]).complete(_ask("one"), "m", step_id="s")

    assert response.usage.reasoning_tokens == 7


async def test_a_replayed_stream_is_copied_as_recorded(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    recorder = RecordingProvider(_source(tmp_path, [_turn("one", "r1")]), out)

    sr = await recorder.stream(_ask("one"), "m", step_id="s")
    assert [chunk async for chunk in sr] == ["r1"]

    assert json.loads(out.read_text())["s"] == _turn("one", "r1")


async def test_a_step_that_misses_midway_records_both_turns_in_order(tmp_path: Path) -> None:
    cut = ReplayCut(_source(tmp_path, [_turn("one", "r1"), _turn("two", "r2")]))
    out = tmp_path / "out.json"
    recorder = RecordingProvider(ReplayOrLiveProvider(_Versioned("v2"), cut), out)

    await recorder.complete(_ask("one"), "m", step_id="s")
    await recorder.complete(_ask("changed"), "m", step_id="s")

    first, second = json.loads(out.read_text())["s"]
    assert first == _turn("one", "r1")
    assert (second["content"], second["provider"]) == ("v2:changed", "versioned")
    assert second["request_hash"] == prompt_hash(_ask("changed"), "m", None, None, None)


class _Echo(BaseProvider):
    """A live provider whose raw answer carries a ``request_hash`` of another request."""

    name = "echo"

    async def complete(self, messages: list[dict[str, Any]], model: str, **kw: Any) -> Any:
        raw = {"request_hash": "another request", "content": "stale"}
        return ProviderResponse(content="fresh", model=model, provider=self.name, raw_response=raw)


async def test_a_live_answer_is_recorded_from_the_response(tmp_path: Path) -> None:
    out = tmp_path / "out.json"

    await RecordingProvider(_Echo(), out).complete(_ask("one"), "m", step_id="s")

    entry = json.loads(out.read_text())["s"]
    assert (entry["content"], entry["provider"]) == ("fresh", "echo")
    assert entry["request_hash"] == prompt_hash(_ask("one"), "m", None, None, None)


async def test_tools_served_from_the_source_are_served_again_after_a_retry(tmp_path: Path) -> None:
    source = _recording(tmp_path, {"s": [_entry("5")]})
    tape = ToolRecorder(tmp_path / "out.json", serve_from=ToolReplayer(source, ToolMode.REPLAY))

    failed = await _attempt(tape)
    assert tape.serve("s", [ADD]) == [("5", True)]
    await tape.capture("s", [(ADD, "5", True)])
    await _attempt(tape, failed)  # the source's cursor goes back with the retry

    assert tape.serve("s", [ADD]) == [("5", True)]


async def test_tools_checked_against_the_source_refuse_a_changed_result(tmp_path: Path) -> None:
    source = _recording(tmp_path, {"s": [_entry("5")]})
    tape = ToolRecorder(tmp_path / "out.json", serve_from=ToolReplayer(source, ToolMode.VERIFY))
    await _attempt(tape)

    with pytest.raises(RecordingMismatchError, match="tool call #1 'add' returned"):
        await tape.capture("s", [(ADD, "6", True)])
