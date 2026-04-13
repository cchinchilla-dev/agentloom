"""Tests for the approval gate step executor."""

from __future__ import annotations

import pytest

from agentloom.core.models import StepDefinition, StepType
from agentloom.core.results import StepStatus
from agentloom.core.state import StateManager
from agentloom.exceptions import PauseRequestedError, StepError
from agentloom.steps.approval_gate import ApprovalGateStep
from agentloom.steps.base import StepContext


def _context(
    step_id: str = "gate",
    output: str | None = "decision",
    state: dict | None = None,
) -> StepContext:
    """Build a minimal StepContext for the approval gate."""
    return StepContext(
        step_definition=StepDefinition(
            id=step_id,
            type=StepType.APPROVAL_GATE,
            output=output,
        ),
        state_manager=StateManager(initial_state=state or {}),
    )


class TestApprovalGateStep:
    async def test_pauses_without_decision(self) -> None:
        ctx = _context()
        with pytest.raises(PauseRequestedError, match="gate"):
            await ApprovalGateStep().execute(ctx)

    async def test_returns_approved(self) -> None:
        ctx = _context(state={"_approval": {"gate": "approved"}})
        result = await ApprovalGateStep().execute(ctx)

        assert result.status == StepStatus.SUCCESS
        assert result.output == "approved"

    async def test_returns_rejected(self) -> None:
        ctx = _context(state={"_approval": {"gate": "rejected"}})
        result = await ApprovalGateStep().execute(ctx)

        assert result.status == StepStatus.SUCCESS
        assert result.output == "rejected"

    async def test_invalid_decision_raises(self) -> None:
        ctx = _context(state={"_approval": {"gate": "maybe"}})
        with pytest.raises(StepError, match="Invalid approval decision"):
            await ApprovalGateStep().execute(ctx)

    async def test_stores_output_in_state(self) -> None:
        ctx = _context(output="my_decision", state={"_approval": {"gate": "approved"}})
        await ApprovalGateStep().execute(ctx)

        value = await ctx.state_manager.get("my_decision")
        assert value == "approved"

    async def test_no_output_field(self) -> None:
        """When output is None, the decision is not stored in state."""
        ctx = _context(output=None, state={"_approval": {"gate": "rejected"}})
        result = await ApprovalGateStep().execute(ctx)

        assert result.output == "rejected"
        # No key written to state
        value = await ctx.state_manager.get("my_decision")
        assert value is None

    async def test_pause_message_on_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        ctx = _context(step_id="review")
        with pytest.raises(PauseRequestedError):
            await ApprovalGateStep().execute(ctx)

        captured = capsys.readouterr()
        assert "APPROVAL REQUIRED" in captured.err
        assert "review" in captured.err
        assert "--approve" in captured.err
        assert "--reject" in captured.err
