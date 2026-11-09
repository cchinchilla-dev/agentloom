"""Approval gate step executor — pauses for human decision."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import PauseRequestedError, StepError
from agentloom.steps.base import BaseStep, StepContext

logger = logging.getLogger("agentloom.steps")


def _now() -> datetime:
    """The current UTC time: tests replace it to move the clock."""
    return datetime.now(UTC)


class ApprovalGateStep(BaseStep):
    """Step that pauses the workflow until a human approves or rejects.

    On first execution the step raises ``PauseRequestedError`` so the
    engine serializes the workflow state.  When the run resumes with
    ``--approve`` or ``--reject``, ``WorkflowEngine.from_checkpoint`` writes
    the decision to state at ``_approval.<step_id>`` and the step returns it
    as output.

    If a ``notify`` webhook is configured on the step definition, a
    POST request is sent before pausing so external systems (Slack, CI,
    dashboards) can act on the pending approval.

    With ``timeout_seconds``, the first pause sets when the gate expires,
    which the checkpoint keeps. A gate resumed after that takes ``on_timeout``,
    whatever decision it was given, or ends as ``timeout`` without one.

    The step body does not write to stdout/stderr: two gates pausing in
    one parallel layer would interleave their prompts, so the pause
    surfaces as ``PauseRequestedError`` and the CLI renders it after
    ``engine.run()`` returns.
    """

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        decision = await context.state_manager.get(f"_approval.{step.id}")
        expires_at = context.gate_deadlines.get(step.id)

        if step.timeout_seconds is not None:
            now = _now()
            if expires_at is None:
                expires_at = (now + timedelta(seconds=step.timeout_seconds)).isoformat()
            if now >= datetime.fromisoformat(expires_at):
                if step.on_timeout is None:
                    return self._expired(context, expires_at, start)
                logger.info(
                    "Approval gate '%s' expired at %s: on_timeout=%s applies",
                    step.id,
                    expires_at,
                    step.on_timeout,
                )
                decision = "approved" if step.on_timeout == "approve" else "rejected"

        if decision is None:
            if step.notify:
                await self._send_notification(context)

            if context.observer:
                hook = getattr(context.observer, "on_approval_gate", None)
                if hook:
                    hook(step.id, context.workflow_name, "pending")

            raise PauseRequestedError(
                step.id, f"Approval required at step '{step.id}'", expires_at=expires_at
            )

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

    def _expired(self, context: StepContext, expires_at: str, start: float) -> StepResult:
        """The result of a gate that ran out of time with no ``on_timeout``: no retry decides it."""
        step = context.step_definition
        if context.observer:
            hook = getattr(context.observer, "on_approval_gate", None)
            if hook:
                hook(step.id, context.workflow_name, "timeout")
        return StepResult(
            step_id=step.id,
            status=StepStatus.TIMEOUT,
            error=f"Approval gate '{step.id}' expired at {expires_at} with no on_timeout",
            error_classification="permanent",
            duration_ms=(time.monotonic() - start) * 1000,
        )

    async def _send_notification(self, context: StepContext) -> None:
        """Fire the webhook notification (best-effort, never raises)."""
        from agentloom.tools.sandbox import ToolSandbox
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
            redaction_policy=context.redaction_policy,
        )

        # Gate ``notify.url`` with the workflow's sandbox, or it can reach internal hosts (SSRF).
        sandbox_cfg = context.sandbox_config
        sandbox = ToolSandbox(
            enabled=sandbox_cfg.enabled,
            allow_network=sandbox_cfg.allow_network,
            allowed_domains=sandbox_cfg.allowed_domains,
            allowed_schemes=sandbox_cfg.allowed_schemes,
            allow_internal_webhook_targets=sandbox_cfg.allow_internal_webhook_targets,
        )
        await send_webhook(notify, wh_context, observer=context.observer, sandbox=sandbox)
