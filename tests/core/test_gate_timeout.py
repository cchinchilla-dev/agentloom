"""Tests for the approval gate's timeout_seconds and on_timeout."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agentloom.checkpointing.base import CheckpointData
from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import StepDefinition, StepType, WorkflowConfig, WorkflowDefinition
from agentloom.core.results import StepStatus, WorkflowResult, WorkflowStatus

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.now = T0

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr("agentloom.steps.approval_gate._now", lambda: fake.now)
    return fake


def _gate(
    step_id: str = "approve",
    on_timeout: Literal["approve", "reject"] | None = None,
    timeout: int | None = 60,
    **extra: object,
) -> StepDefinition:
    return StepDefinition(
        id=step_id,
        type=StepType.APPROVAL_GATE,
        timeout_seconds=timeout,
        on_timeout=on_timeout,
        output="decision",
        **extra,  # type: ignore[arg-type]
    )


def _workflow(*steps: StepDefinition) -> WorkflowDefinition:
    return WorkflowDefinition(
        name="gate-timeout",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        steps=list(steps) or [_gate()],
    )


async def _pause(
    tmp_path: Path, workflow: WorkflowDefinition, run_id: str = "run"
) -> tuple[FileCheckpointer, CheckpointData]:
    checkpointer = FileCheckpointer(checkpoint_dir=tmp_path)
    result = await WorkflowEngine(workflow=workflow, checkpointer=checkpointer, run_id=run_id).run()
    assert result.status == WorkflowStatus.PAUSED
    return checkpointer, await checkpointer.load(run_id)


async def _resume(
    checkpointer: FileCheckpointer,
    data: CheckpointData,
    decision: str | None = None,
    observer: object | None = None,
) -> WorkflowResult:
    decisions = {data.paused_step_id: decision} if decision and data.paused_step_id else None
    engine = await WorkflowEngine.from_checkpoint(
        checkpoint_data=data,
        checkpointer=checkpointer,
        approval_decisions=decisions,
        observer=observer,
    )
    return await engine.run()


class TestExpiry:
    async def test_the_first_pause_records_when_the_gate_expires(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        _, data = await _pause(tmp_path, _workflow())
        expected = (T0 + timedelta(seconds=60)).isoformat()
        assert data.gate_deadlines == {"approve": expected}
        assert data.expires_at == expected

    async def test_a_gate_without_a_timeout_records_none(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        _, data = await _pause(tmp_path, _workflow(_gate(timeout=None)))
        assert data.gate_deadlines == {}
        assert data.expires_at is None

    @pytest.mark.parametrize(
        ("on_timeout", "taken"), [("reject", "rejected"), ("approve", "approved")]
    )
    async def test_an_expired_gate_takes_on_timeout_over_the_decision(
        self,
        tmp_path: Path,
        clock: _Clock,
        on_timeout: Literal["approve", "reject"],
        taken: str,
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow(_gate(on_timeout=on_timeout)))
        clock.advance(60)
        sent = "approved" if taken == "rejected" else "rejected"
        result = await _resume(checkpointer, data, sent)
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["approve"].output == taken
        assert result.final_state["decision"] == taken

    async def test_an_expired_gate_takes_on_timeout_without_a_decision(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow(_gate(on_timeout="approve")))
        clock.advance(61)
        result = await _resume(checkpointer, data)
        assert result.step_results["approve"].output == "approved"

    async def test_an_expired_gate_without_on_timeout_fails_once_with_timeout(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow())
        clock.advance(60)
        observer = MagicMock()
        result = await _resume(checkpointer, data, "approved", observer=observer)

        gate = result.step_results["approve"]
        assert result.status == WorkflowStatus.FAILED
        assert gate.status == StepStatus.TIMEOUT
        assert gate.error_classification == "permanent"
        assert gate.error is not None and "expired" in gate.error
        decisions = [call.args[2] for call in observer.on_approval_gate.call_args_list]
        assert decisions == ["timeout"]  # once: no retry decides it

    async def test_a_hook_that_raises_leaves_an_expired_gate_alone(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow())
        clock.advance(60)
        observer = MagicMock()
        observer.on_approval_gate.side_effect = RuntimeError("hook bug")

        result = await _resume(checkpointer, data, "approved", observer=observer)

        assert result.step_results["approve"].status == StepStatus.TIMEOUT

    async def test_a_gate_before_its_expiry_pauses_again_with_the_same_deadline(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow(_gate(on_timeout="reject")))
        clock.advance(59)
        result = await _resume(checkpointer, data)
        assert result.status == WorkflowStatus.PAUSED
        again = await checkpointer.load("run")
        assert again.gate_deadlines == data.gate_deadlines

    async def test_a_decision_before_the_expiry_applies(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow(_gate(on_timeout="reject")))
        clock.advance(59)
        result = await _resume(checkpointer, data, "approved")
        assert result.step_results["approve"].output == "approved"

    async def test_a_checkpoint_without_a_deadline_counts_from_the_resume(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow(_gate(on_timeout="reject")))
        data.gate_deadlines = {}  # as a checkpoint written before gates expired
        clock.advance(600)
        result = await _resume(checkpointer, data)
        assert result.status == WorkflowStatus.PAUSED
        again = await checkpointer.load("run")
        assert again.expires_at == (clock.now + timedelta(seconds=60)).isoformat()

    async def test_a_checkpoint_without_a_deadline_takes_the_decision(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        checkpointer, data = await _pause(tmp_path, _workflow(_gate(on_timeout="reject")))
        data.gate_deadlines = {}
        clock.advance(600)
        result = await _resume(checkpointer, data, "approved")
        assert result.step_results["approve"].output == "approved"

    async def test_parallel_gates_each_record_their_deadline(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        workflow = _workflow(_gate("first", timeout=60), _gate("second", timeout=120))
        _, data = await _pause(tmp_path, workflow)
        assert data.gate_deadlines == {
            "first": (T0 + timedelta(seconds=60)).isoformat(),
            "second": (T0 + timedelta(seconds=120)).isoformat(),
        }


class TestSubworkflowGates:
    @staticmethod
    def _parent(child_gate: dict[str, object], *before: StepDefinition) -> WorkflowDefinition:
        sub = StepDefinition(
            id="sub",
            type=StepType.SUBWORKFLOW,
            depends_on=[step.id for step in before],
            workflow_inline={"name": "child", "steps": [child_gate]},
        )
        return _workflow(*before, sub)

    async def test_a_childs_deadline_reaches_the_parent_and_expires_there(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        child_gate = {
            "id": "gate",
            "type": "approval_gate",
            "timeout_seconds": 60,
            "on_timeout": "reject",
            "output": "decision",
        }
        checkpointer, data = await _pause(tmp_path, self._parent(child_gate))
        assert data.paused_step_id == "sub.gate"
        assert data.expires_at == (T0 + timedelta(seconds=60)).isoformat()

        clock.advance(60)
        result = await _resume(checkpointer, data, "approved")
        assert result.status == WorkflowStatus.SUCCESS
        assert result.step_results["sub"].output["decision"] == "rejected"

    async def test_a_child_gate_does_not_take_the_parents_decision(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        child_gate = {"id": "review", "type": "approval_gate"}
        parent = self._parent(child_gate, _gate("review", timeout=None))
        checkpointer, data = await _pause(tmp_path, parent)
        assert data.paused_step_id == "review"

        result = await _resume(checkpointer, data, "approved")
        assert result.status == WorkflowStatus.PAUSED
        assert (await checkpointer.load("run")).paused_step_id == "sub.review"


class TestValidation:
    def test_on_timeout_needs_timeout_seconds(self) -> None:
        with pytest.raises(ValidationError, match="'on_timeout' needs 'timeout_seconds'"):
            _gate(on_timeout="approve", timeout=None)

    @pytest.mark.parametrize("seconds", [0, -5])
    def test_timeout_seconds_must_be_positive(self, seconds: int) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            _gate(timeout=seconds)


class TestCheckpointExpiry:
    @staticmethod
    def _data(status: str = "paused", deadline: datetime | None = T0) -> CheckpointData:
        return CheckpointData(
            workflow_name="wf",
            run_id="r",
            workflow_definition={},
            status=status,
            paused_step_id="approve",
            gate_deadlines={"approve": deadline.isoformat()} if deadline else {},
        )

    def test_the_expiry_is_reached_at_the_deadline(self) -> None:
        data = self._data()
        assert data.gate_expired(T0)
        assert not data.gate_expired(T0 - timedelta(seconds=1))

    def test_a_run_that_is_not_paused_has_no_expiry(self) -> None:
        data = self._data(status="success")
        assert data.expires_at is None
        assert not data.gate_expired(T0)

    def test_a_gate_without_a_deadline_never_expires(self) -> None:
        assert not self._data(deadline=None).gate_expired(T0 + timedelta(days=365))
