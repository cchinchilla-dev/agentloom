"""Approval gate step executor — pauses for human decision."""

from __future__ import annotations

import sys
import time

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import PauseRequestedError, StepError
from agentloom.steps.base import BaseStep, StepContext


class ApprovalGateStep(BaseStep):
    """Pauses the workflow until a human approves or rejects.

    On first execution the step raises ``PauseRequestedError`` so the
    engine serializes the workflow state.  When the user resumes with
    ``--approve`` or ``--reject``, the CLI injects the decision into
    state at ``_approval.<step_id>`` and the step returns it as output.
    """

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        # Check whether a decision was injected into state on resume
        decision = await context.state_manager.get(f"_approval.{step.id}")

        if decision is None:
            # First execution — print instructions and pause
            print(
                f"\n[APPROVAL REQUIRED] Step '{step.id}' is waiting for approval.\n"
                f"  Resume with: agentloom resume <run_id> --approve\n"
                f"  Or reject:   agentloom resume <run_id> --reject\n",
                file=sys.stderr,
            )
            raise PauseRequestedError(step.id, f"Approval required at step '{step.id}'")

        # Validate the decision value
        if decision not in ("approved", "rejected"):
            raise StepError(
                step.id,
                f"Invalid approval decision '{decision}'. Expected 'approved' or 'rejected'.",
            )

        duration = (time.monotonic() - start) * 1000

        # Store the decision in the output state variable
        if step.output:
            await context.state_manager.set(step.output, decision)

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=decision,
            duration_ms=duration,
        )
