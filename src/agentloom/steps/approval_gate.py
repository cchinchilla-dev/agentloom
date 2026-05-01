"""Approval gate step executor — pauses for human decision."""

from __future__ import annotations

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

    If a ``notify`` webhook is configured on the step definition, a
    POST request is sent before pausing so external systems (Slack, CI,
    dashboards) can act on the pending approval.

    The step body intentionally does **not** write to stdout/stderr — UX
    rendering is the CLI's job. Under parallel DAG execution two gates
    pausing concurrently used to interleave their prompts; surfacing the
    pause via ``PauseRequestedError`` and letting the CLI render after
    `engine.run()` returns avoids the race entirely.
    """

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        decision = await context.state_manager.get(f"_approval.{step.id}")

        if decision is None:
            if step.notify:
                await self._send_notification(context)

            if context.observer:
                hook = getattr(context.observer, "on_approval_gate", None)
                if hook:
                    hook(step.id, context.workflow_name, "pending")

            raise PauseRequestedError(step.id, f"Approval required at step '{step.id}'")

        if decision not in ("approved", "rejected"):
            raise StepError(
                step.id,
                f"Invalid approval decision '{decision}'. Expected 'approved' or 'rejected'.",
            )

        if context.observer:
            hook = getattr(context.observer, "on_approval_gate", None)
            if hook:
                hook(step.id, context.workflow_name, decision)

        duration = (time.monotonic() - start) * 1000

        if step.output:
            await context.state_manager.set(step.output, decision)

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=decision,
            duration_ms=duration,
        )

    async def _send_notification(self, context: StepContext) -> None:
        """Fire the webhook notification (best-effort, never raises)."""
        from agentloom.webhooks.sender import WebhookContext, send_webhook

        step = context.step_definition
        notify = step.notify
        if notify is None:  # pragma: no cover — caller already checks step.notify
            return

        state_snapshot = await context.state_manager.get_state_snapshot()

        wh_context = WebhookContext(
            run_id=context.run_id,
            step_id=step.id,
            workflow_name=context.workflow_name,
            state=state_snapshot,
        )
        await send_webhook(notify, wh_context, observer=context.observer)
