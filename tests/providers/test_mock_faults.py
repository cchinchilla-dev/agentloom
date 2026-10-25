"""Deterministic faults declared in a mock recording (#123)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import (
    RetryConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.results import WorkflowStatus
from agentloom.exceptions import ProviderError, RateLimitError, RecordingMismatchError
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider, prompt_hash
from agentloom.resilience.circuit_breaker import CircuitState

HI = [{"role": "user", "content": "hi"}]


def _mock(tmp_path: Path, entries: dict[str, Any], **kwargs: Any) -> MockProvider:
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(entries))
    return MockProvider(responses_file=path, **kwargs)


def _fault(**fault: Any) -> dict[str, Any]:
    return {"fault": fault}


@pytest.mark.parametrize(
    ("fault", "error", "message", "status"),
    [
        (
            {"type": "http_error", "status_code": 500, "body": "boom"},
            ProviderError,
            "API error 500: boom",
            500,
        ),
        (
            {"type": "http_error", "status_code": 429, "retry_after_s": 30},
            RateLimitError,
            "retry_after=30",
            429,
        ),
        ({"type": "timeout", "after_ms": 250}, ProviderError, "ReadTimeout", None),
        ({"type": "connection_reset"}, ProviderError, "ConnectError", None),
    ],
)
async def test_a_fault_raises_what_a_real_adapter_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: dict[str, Any],
    error: type[ProviderError],
    message: str,
    status: int | None,
) -> None:
    slept: list[float] = []

    async def sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("agentloom.providers.mock.anyio.sleep", sleep)
    mock = _mock(tmp_path, {"s": {"fault": fault}})

    with pytest.raises(error, match=message) as info:
        await mock.complete(HI, "m", step_id="s")

    assert type(info.value) is error
    assert info.value.status_code == status
    assert slept == ([0.25] if fault["type"] == "timeout" else [])


async def test_a_429_without_retry_after_has_none(tmp_path: Path) -> None:
    mock = _mock(tmp_path, {"s": _fault(type="http_error", status_code=429)})

    with pytest.raises(RateLimitError) as info:
        await mock.complete(HI, "m", step_id="s")

    assert info.value.retry_after_s is None


async def test_a_cut_stream_delivers_its_first_words_then_fails(tmp_path: Path) -> None:
    entry = {"content": "one two three four", **_fault(type="stream_truncate", after_chunks=2)}
    mock = _mock(tmp_path, {"s": entry})

    sr = await mock.stream(HI, "m", step_id="s")
    seen: list[str] = []
    with pytest.raises(
        ProviderError, match=r"ReadError: the connection dropped after 2 chunk\(s\)"
    ):
        async for chunk in sr:
            seen.append(chunk)

    assert seen == ["one ", "two "]
    with pytest.raises(ProviderError, match="ReadError"):  # a plain call gets no text at all
        await mock.complete(HI, "m", step_id="s")


async def test_a_stream_without_a_fault_is_one_chunk(tmp_path: Path) -> None:
    sr = await _mock(tmp_path, {"s": {"content": "whole"}}).stream(HI, "m", step_id="s")

    assert [chunk async for chunk in sr] == ["whole"]


@pytest.mark.parametrize(
    "fault",
    [
        {"type": "explode"},
        {"type": "http_error"},
        {"type": "http_error", "status_code": 200},
        {"type": "timeout", "after_ms": -1},
        {"type": "connection_reset", "status_code": 500},
    ],
    ids=["unknown type", "missing status", "not an error", "negative wait", "extra field"],
)
def test_a_malformed_fault_is_refused_on_load(tmp_path: Path, fault: dict[str, Any]) -> None:
    for entry in ({"fault": fault}, [{"content": "ok"}, {"fault": fault}]):
        with pytest.raises(ValueError, match=r"entry 's' declares an invalid fault"):
            _mock(tmp_path, {"s": entry})


async def test_strict_replay_fires_a_fault_between_recorded_turns(tmp_path: Path) -> None:
    turn = {"content": "ok", "request_hash": prompt_hash(HI, "m", None, None, None)}
    mock = _mock(tmp_path, {"s": [_fault(type="connection_reset"), turn]}, strict=True)

    with pytest.raises(ProviderError, match="ConnectError"):
        await mock.complete(HI, "m", step_id="s")
    response = await mock.complete(HI, "m", step_id="s")  # the retry gets the recorded answer

    assert response.content == "ok"


async def test_a_fault_inside_a_loop_fires_on_every_attempt(tmp_path: Path) -> None:
    ask, follow = HI, [*HI, {"role": "user", "content": "more"}]
    turns = [
        {"content": "first", "request_hash": prompt_hash(ask, "m", None, None, None)},
        _fault(type="timeout"),
        {"content": "second", "request_hash": prompt_hash(follow, "m", None, None, None)},
    ]
    mock = _mock(tmp_path, {"s": turns}, strict=True)

    for _ in range(2):  # a retry starts the loop again and reaches the fault again
        assert (await mock.complete(ask, "m", step_id="s")).content == "first"
        with pytest.raises(ProviderError, match="ReadTimeout"):
            await mock.complete(follow, "m", step_id="s")


async def test_a_fault_on_a_recorded_turn_needs_its_request(tmp_path: Path) -> None:
    other = [{"role": "user", "content": "changed"}]
    turn = {"content": "ok", "request_hash": prompt_hash(HI, "m", None, None, None)}
    faulted = {**turn, **_fault(type="connection_reset")}
    mock = _mock(tmp_path, {"s": [faulted]}, strict=True)

    with pytest.raises(RecordingMismatchError):  # a changed request is a stale recording
        await mock.complete(other, "m", step_id="s")
    with pytest.raises(ProviderError, match="ConnectError"):
        await mock.complete(HI, "m", step_id="s")


async def test_a_fault_is_reported_to_the_observer(tmp_path: Path) -> None:
    observer = MagicMock()
    mock = _mock(tmp_path, {"s": _fault(type="connection_reset")}, observer=observer)
    mock._workflow_name = "wf"

    with pytest.raises(ProviderError):
        await mock.complete(HI, "m", step_id="s")

    observer.on_mock_fault.assert_called_once_with("wf", "s", "connection_reset")
    observer.on_mock_replay.assert_called_once_with("wf", "s", "step_id")  # still a lookup


async def test_an_observer_without_the_fault_hook_still_replays(tmp_path: Path) -> None:
    class ReplayOnly:  # written before on_mock_fault existed
        def __init__(self) -> None:
            self.replays: list[str] = []

        def on_mock_replay(self, workflow_name: str, step_id: str, matched_by: str) -> None:
            self.replays.append(step_id)

    observer = ReplayOnly()
    mock = _mock(tmp_path, {"s": _fault(type="connection_reset")}, observer=observer)

    with pytest.raises(ProviderError, match="ConnectError"):
        await mock.complete(HI, "m", step_id="s")

    assert observer.replays == ["s"]


def _engine(mock: MockProvider, step: StepDefinition) -> tuple[WorkflowEngine, ProviderGateway]:
    gateway = ProviderGateway()
    gateway.register(mock, circuit_fail_threshold=2)
    workflow = WorkflowDefinition(
        name="faults", config=WorkflowConfig(provider="mock", model="m"), steps=[step]
    )
    return WorkflowEngine(workflow=workflow, provider_gateway=gateway), gateway


def _step(max_retries: int) -> StepDefinition:
    retry = RetryConfig(max_retries=max_retries, backoff_max=0, jitter=False)
    return StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi", output="out", retry=retry)


@pytest.mark.parametrize("status", [500, 429])
async def test_a_fault_in_a_workflow_triggers_its_retry_policy(tmp_path: Path, status: int) -> None:
    turns = [_fault(type="http_error", status_code=status), {"content": "recovered"}]
    mock = _mock(tmp_path, {"s": turns})
    engine, _ = _engine(mock, _step(max_retries=1))

    result = await engine.run()

    assert result.status == WorkflowStatus.SUCCESS
    assert result.final_state["out"] == "recovered"
    assert len(mock.calls) == 2


async def test_a_fault_propagates_to_the_circuit_breaker(tmp_path: Path) -> None:
    mock = _mock(tmp_path, {"s": [_fault(type="http_error", status_code=503)] * 2})
    engine, gateway = _engine(mock, _step(max_retries=1))

    await engine.run()

    assert gateway._providers[0].circuit_breaker.state == CircuitState.OPEN
