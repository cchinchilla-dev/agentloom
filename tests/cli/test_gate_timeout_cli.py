"""Tests for expiring approval gates in the CLI: the callback server's sweep, resume and runs."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
from typer.testing import CliRunner

from agentloom.checkpointing.base import CheckpointData
from agentloom.checkpointing.file import FileCheckpointer
from agentloom.cli import callback_server
from agentloom.cli.main import app
from agentloom.core.models import StepDefinition, StepType, WorkflowConfig, WorkflowDefinition
from tests.conftest import MockProvider

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
DEADLINE = T0 + timedelta(seconds=60)


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    """The clock gates and the server read: set ``now[0]`` to move it."""
    current = [T0]
    monkeypatch.setattr("agentloom.steps.approval_gate._now", lambda: current[0])
    return current


@pytest.fixture(autouse=True)
def _wired() -> Iterator[None]:
    """Providers and observer of a resumed run: a mock, and none."""

    def _setup(gw: object, default: str, config_path: object = None) -> None:
        gw.register(MockProvider(), priority=0)  # type: ignore[attr-defined]

    with (
        patch("agentloom.cli.run._setup_providers", side_effect=_setup),
        patch("agentloom.cli.run._setup_observer", return_value=None),
    ):
        yield
    callback_server._resuming.clear()


def _workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="timeout-cb",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        steps=[
            StepDefinition(
                id="gate",
                type=StepType.APPROVAL_GATE,
                timeout_seconds=60,
                on_timeout="reject",
                output="decision",
            ),
        ],
    )


def _write(cp_dir: Path, run_id: str, deadline: datetime | None = DEADLINE) -> None:
    wf = _workflow()
    data = CheckpointData(
        workflow_name=wf.name,
        run_id=run_id,
        workflow_definition=wf.model_dump(),
        status="paused",
        paused_step_id="gate",
        gate_deadlines={"gate": deadline.isoformat()} if deadline else {},
        created_at=T0.isoformat(),
        updated_at=T0.isoformat(),
    )
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / f"{run_id}.json").write_text(data.model_dump_json())


class _Responses:
    def __init__(self) -> None:
        self.sent: list[tuple[int, dict[str, object]]] = []

    async def send(self, data: bytes) -> None:
        text = data.decode()
        self.sent.append((int(text.split(" ")[1]), json.loads(text[text.index("\r\n\r\n") + 4 :])))


async def _decide(cp_dir: Path, run_id: str, decision: str = "approved") -> _Responses:
    stream = _Responses()
    await callback_server._handle_decision(stream, str(cp_dir), True, run_id, decision)  # type: ignore[arg-type]
    return stream


async def _final(cp_dir: Path, run_id: str) -> CheckpointData:
    return await FileCheckpointer(checkpoint_dir=cp_dir).load(run_id)


class TestDecisions:
    async def test_pending_shows_when_the_gate_expires(self, tmp_path: Path) -> None:
        _write(tmp_path, "run-pending")
        stream = _Responses()
        await callback_server._handle_pending(stream, str(tmp_path))  # type: ignore[arg-type]
        [(status, body)] = stream.sent
        assert status == 200
        assert body["paused_runs"][0]["expires_at"] == DEADLINE.isoformat()  # type: ignore[index]

    async def test_a_late_decision_answers_202_and_the_gate_takes_on_timeout(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-late")
        now[0] = DEADLINE
        stream = await _decide(tmp_path, "run-late", "approved")

        [(status, body)] = stream.sent
        assert status == 202
        assert body["expired_at"] == DEADLINE.isoformat()
        assert "on_timeout decides" in str(body["message"])
        final = await _final(tmp_path, "run-late")
        assert final.state["decision"] == "rejected"

    async def test_a_timely_decision_answers_202_without_an_expiry(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-timely")
        stream = await _decide(tmp_path, "run-timely", "approved")
        [(status, body)] = stream.sent
        assert status == 202
        assert "expired_at" not in body
        assert (await _final(tmp_path, "run-timely")).state["decision"] == "approved"

    async def test_a_run_already_resuming_answers_409(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-busy")
        callback_server._resuming.add("run-busy")
        stream = await _decide(tmp_path, "run-busy")
        assert stream.sent == [(409, {"error": "run 'run-busy' is already resuming"})]
        assert (await _final(tmp_path, "run-busy")).status == "paused"

    async def test_the_run_is_released_after_its_resume(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-done")
        await _decide(tmp_path, "run-done")
        assert "run-done" not in callback_server._resuming

    async def test_the_run_is_released_when_provider_setup_fails(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        from agentloom.exceptions import ConfigError

        _write(tmp_path, "run-broken")
        with patch("agentloom.cli.run._setup_providers", side_effect=ConfigError("bad")):
            stream = await _decide(tmp_path, "run-broken")
        assert stream.sent == [(500, {"error": "bad"})]
        assert "run-broken" not in callback_server._resuming


class TestSweep:
    @staticmethod
    async def _sweep(cp_dir: Path) -> float:
        async with anyio.create_task_group() as tg:
            wait = await callback_server._sweep_once(str(cp_dir), True, tg)
        return wait

    async def test_it_resumes_an_expired_run_with_on_timeout(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-expired")
        now[0] = DEADLINE + timedelta(seconds=1)
        await self._sweep(tmp_path)
        final = await _final(tmp_path, "run-expired")
        assert final.status == "success"
        assert final.state["decision"] == "rejected"
        assert "run-expired" not in callback_server._resuming

    async def test_it_sleeps_until_the_next_expiry(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-soon", deadline=T0 + timedelta(seconds=12))
        _write(tmp_path, "run-later", deadline=T0 + timedelta(seconds=20))
        assert await self._sweep(tmp_path) == 12
        assert (await _final(tmp_path, "run-soon")).status == "paused"

    @pytest.mark.parametrize("deadline", [None, T0 + timedelta(hours=1)])
    async def test_it_sleeps_at_most_the_interval(
        self, tmp_path: Path, now: list[datetime], deadline: datetime | None
    ) -> None:
        _write(tmp_path, "run-far", deadline=deadline)
        assert await self._sweep(tmp_path) == callback_server._SWEEP_INTERVAL

    async def test_it_leaves_child_checkpoints_and_runs_it_is_resuming(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "parent.sub")
        _write(tmp_path, "run-busy")
        callback_server._resuming.add("run-busy")
        now[0] = DEADLINE
        await self._sweep(tmp_path)
        assert (await _final(tmp_path, "parent.sub")).status == "paused"
        assert (await _final(tmp_path, "run-busy")).status == "paused"

    async def test_a_failed_resume_is_logged_and_released(
        self, tmp_path: Path, now: list[datetime], caplog: pytest.LogCaptureFixture
    ) -> None:
        from agentloom.exceptions import ConfigError

        _write(tmp_path, "run-fails")
        now[0] = DEADLINE
        with (
            patch("agentloom.cli.run._setup_providers", side_effect=ConfigError("bad")),
            caplog.at_level(logging.WARNING, logger="agentloom.callback"),
        ):
            await self._sweep(tmp_path)
        assert "Could not resume run 'run-fails'" in caplog.text
        assert "run-fails" not in callback_server._resuming
        assert (await _final(tmp_path, "run-fails")).status == "paused"


class TestCommands:
    def test_resume_says_an_expired_gate_ignores_the_decision(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-cli")
        now[0] = DEADLINE
        result = CliRunner().invoke(
            app, ["resume", "run-cli", "--approve", "--checkpoint-dir", str(tmp_path), "--lite"]
        )
        output = result.output + (result.stderr or "")
        assert "The gate expired at" in output
        assert "--approve is ignored" in output

    def test_resume_before_the_expiry_says_nothing_of_it(
        self, tmp_path: Path, now: list[datetime]
    ) -> None:
        _write(tmp_path, "run-cli-ok")
        result = CliRunner().invoke(
            app, ["resume", "run-cli-ok", "--reject", "--checkpoint-dir", str(tmp_path), "--lite"]
        )
        assert "expired" not in result.output + (result.stderr or "")

    def test_runs_shows_when_the_gate_expires(self, tmp_path: Path) -> None:
        _write(tmp_path, "run-table")
        result = CliRunner().invoke(app, ["runs", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "GATE EXPIRES" in result.output
        assert DEADLINE.isoformat()[:19] in result.output
