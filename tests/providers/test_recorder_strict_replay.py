"""Strict replay of recorded loops: only a turn recorded for the exact request is served."""

from __future__ import annotations

import json

import pytest

from agentloom.core.models import RetryConfig
from agentloom.core.results import WorkflowStatus
from agentloom.exceptions import RecordingMismatchError
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider
from tests.providers.test_recorder_tool_loop import (
    ADD_LOOP,
    _add_workflow,
    _AddTool,
    _run,
    _tool_loop,
    _turn,
)

GO = [{"role": "user", "content": "go"}]


async def _record_loop(tmp_path, turns: list, calls: int, runs: int = 1):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"loop": turns}))
    recording = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(responses_file=src), recording)
    for _ in range(runs):
        await _tool_loop(recorder, calls)
    return recording


async def test_strict_replay_detects_drift_in_a_list_turn(tmp_path):
    recording = await _record_loop(tmp_path, [_turn("", ("c1", "t", {})), _turn("end")], 2)

    replay = MockProvider(responses_file=recording, strict=True)
    await replay.complete(messages=GO, model="m", step_id="loop")
    with pytest.raises(RecordingMismatchError, match="no recorded turn matches"):
        await replay.complete(
            messages=[{"role": "user", "content": "changed"}], model="m", step_id="loop"
        )


async def test_a_retry_after_drift_fails_again_instead_of_serving_the_last_turn(tmp_path):
    recording = await _record_loop(tmp_path, [_turn("", ("c1", "t", {})), _turn("end")], 2)
    replay = MockProvider(responses_file=recording, strict=True)
    changed = [*GO, {"role": "tool", "content": "something else"}]
    for _ in range(3):  # an engine retry restarts the loop from turn 1
        await replay.complete(messages=GO, model="m", step_id="loop")
        with pytest.raises(RecordingMismatchError):
            await replay.complete(messages=changed, model="m", step_id="loop")


async def test_strict_replay_follows_a_loop_recorded_twice_in_one_run(tmp_path):
    turns = [_turn("first"), _turn("second"), _turn("third"), _turn("fourth")]
    recording = await _record_loop(tmp_path, turns, calls=2, runs=2)  # a step retried once
    assert len(json.loads(recording.read_text())["loop"]) == 4

    replay = MockProvider(responses_file=recording, strict=True)
    first = await _tool_loop(replay, 2)
    second = await _tool_loop(replay, 2)
    assert [r.content for r in first + second] == ["first", "second", "third", "fourth"]


async def test_tool_workflow_whose_tool_output_changed_fails_strict_replay(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"ask": ADD_LOOP}))
    recording = tmp_path / "rec.json"
    workflow = _add_workflow(retry=RetryConfig(max_retries=3, backoff_max=0.0, jitter=False))
    await _run(workflow, RecordingProvider(MockProvider(responses_file=src), recording), _AddTool())

    replayed = await _run(
        workflow, MockProvider(responses_file=recording, strict=True), _AddTool(1)
    )
    assert replayed.status == WorkflowStatus.FAILED


async def test_structured_output_ids_stay_out_of_the_recorded_hash(tmp_path):
    recording = tmp_path / "rec.json"
    recorder = RecordingProvider(MockProvider(default_response="{}"), recording)
    await recorder.complete(messages=GO, model="m", step_id="s", agentloom_step_id="s")

    replay = MockProvider(responses_file=recording, strict=True)
    response = await replay.complete(messages=GO, model="m", step_id="s", agentloom_step_id="s")
    assert response.content == "{}"


async def test_a_second_recorder_on_the_file_keeps_a_re_recorded_step(tmp_path):
    recording = tmp_path / "rec.json"
    recording.write_text(json.dumps({"_version": 2, "ask": {"content": "old"}}))
    first = RecordingProvider(MockProvider(default_response="new"), recording)
    second = RecordingProvider(MockProvider(default_response="unused"), recording)

    await first.complete(messages=GO, model="m", step_id="ask")
    await first.close()
    await second.close()  # closes last, as a fallback provider does in the gateway

    assert json.loads(recording.read_text())["ask"]["content"] == "new"
