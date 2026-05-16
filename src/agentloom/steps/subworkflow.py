"""Subworkflow step executor — runs a nested workflow."""

from __future__ import annotations

import time

from agentloom.core.results import StepResult, StepStatus, WorkflowStatus
from agentloom.exceptions import PauseRequestedError, StepError
from agentloom.steps.base import BaseStep, StepContext


class SubworkflowStep(BaseStep):
    """Executes a nested workflow as a step, passing state down and merging results up."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.parser import WorkflowParser
        from agentloom.core.state import StateManager

        if step.workflow_path:
            try:
                sub_workflow = WorkflowParser.from_yaml(step.workflow_path)
            except Exception as e:
                raise StepError(step.id, f"Failed to load subworkflow: {e}") from e
        elif step.workflow_inline:
            try:
                sub_workflow = WorkflowParser.from_dict(step.workflow_inline)
            except Exception as e:
                raise StepError(step.id, f"Invalid inline subworkflow: {e}") from e
        else:
            raise StepError(
                step.id,
                "Subworkflow step requires 'workflow_path' or 'workflow_inline'",
            )

        # State propagation contract:
        #
        # * Default (``isolated_state: False``): the child engine sees a copy
        #   of the parent's state. This was the pre-0.5.0 behaviour — kept
        #   for compatibility because several existing workflows depend on
        #   it implicitly.
        # * ``isolated_state: True``: the child sees only its own ``state:``
        #   block plus whatever the parent step explicitly passes via
        #   ``input:``. The child's own ``state:`` from the inline /
        #   referenced YAML is preserved by ``WorkflowParser``; we merge
        #   ``input:`` on top so the parent can seed specific keys without
        #   leaking the rest of its state.
        parent_state = await context.state_manager.get_state_snapshot()
        if step.isolated_state:
            seed = dict(sub_workflow.state)
            seed.update(step.input)
            child_state = StateManager(initial_state=seed)
        else:
            child_state = StateManager(initial_state=parent_state)

        # Resume hand-off: when the parent was paused inside this subworkflow
        # (e.g. ``sub.gate``), the CLI ``resume --approve`` stored the
        # decision under ``_approval.<this_step_id>.<child_step_id>``. The
        # child engine's approval-gate executor looks it up under the
        # unqualified ``_approval.<child_step_id>``, so we strip the prefix
        # here. Without this rewrite the child would block again on resume
        # — there'd be no way to ever complete a nested gate. Applies to
        # both isolated and non-isolated modes; the parent's approval
        # bookkeeping is metadata, not user state.
        parent_approvals = parent_state.get("_approval", {})
        if isinstance(parent_approvals, dict):
            nested = parent_approvals.get(step.id)
            if isinstance(nested, dict):
                for child_step_id, decision in nested.items():
                    await child_state.set(f"_approval.{child_step_id}", decision)

        # Inherit the parent's security posture into the child workflow:
        # without this, a parent that redacts ``api_key`` writes the
        # secret in plaintext to the child checkpoint, and a parent that
        # locks the sandbox to ``allowed_domains=["api.openai.com"]`` is
        # bypassed by a subworkflow whose own config defaults to
        # ``sandbox.enabled=false``.
        from agentloom.core.models import StateKeyConfig

        if context.redaction_policy:
            for pattern in context.redaction_policy.patterns:
                if pattern not in sub_workflow.state_schema:
                    sub_workflow.state_schema[pattern] = StateKeyConfig(redact=True)

        # Sandbox inheritance: parent overrides child ONLY when the
        # parent itself has the sandbox enabled. Without this guard, a
        # parent running with the default ``enabled=False`` would wipe
        # out a child workflow that declared a stricter sandbox of its
        # own — a child loosening parent restrictions is the threat
        # model, child *tightening* its own surface should not regress.
        if context.sandbox_config.enabled:
            sub_workflow.config.sandbox = context.sandbox_config

        engine = WorkflowEngine(
            workflow=sub_workflow,
            state_manager=child_state,
            provider_gateway=context.provider_gateway,
            tool_registry=context.tool_registry,
            observer=context.observer,
            on_stream_chunk=context.on_stream_chunk,
            checkpointer=context.checkpointer,
            run_id=f"{context.run_id}.{step.id}" if context.run_id else step.id,
        )

        try:
            result = await engine.run()
        except PauseRequestedError as pause:
            # Defensive: child engines today return a paused
            # ``WorkflowResult`` rather than re-raising, but third-party
            # engines or future refactors might propagate the exception —
            # treat it symmetrically with the result-based path below.
            qualified = (
                f"{step.id}.{pause.step_id}" if pause.step_id else step.id
            )
            raise PauseRequestedError(qualified, str(pause)) from pause
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Subworkflow failed: {e}",
                duration_ms=duration,
            )

        # Child engines absorb their own ``PauseRequestedError`` and return
        # a paused ``WorkflowResult`` (so they can persist their own
        # checkpoint without cancelling sibling tasks). Surface the pause
        # to the parent here with a fully-qualified ``parent.child`` step
        # id so ``agentloom resume <parent_run_id> --approve`` lands on the
        # gate inside the child. Pre-0.5.0 the parent treated this as a
        # generic exception, marked the subworkflow FAILED, and left no
        # resume path at all.
        if result.status == WorkflowStatus.PAUSED:
            inner_paused_id: str | None = None
            for inner_id, inner_res in result.step_results.items():
                if inner_res.status == StepStatus.PAUSED:
                    inner_paused_id = inner_res.error or inner_id
                    break
            qualified = f"{step.id}.{inner_paused_id}" if inner_paused_id else step.id
            raise PauseRequestedError(qualified, str(result.error or ""))

        duration = (time.monotonic() - start) * 1000

        # Surface only the keys the parent asked for. ``return_keys: null``
        # (default) keeps today's behaviour of dumping the entire child
        # final state — workflows that want encapsulation set
        # ``return_keys: [result]`` (or similar) to drop the rest at the
        # boundary. Missing keys are simply omitted; we don't raise so the
        # child can choose not to emit some optional keys.
        child_final = result.final_state
        if step.return_keys is not None:
            child_final = {k: child_final[k] for k in step.return_keys if k in child_final}
        if step.output:
            await context.state_manager.set(step.output, child_final)

        status = (
            StepStatus.SUCCESS if result.status == WorkflowStatus.SUCCESS else StepStatus.FAILED
        )

        from agentloom.core.results import TokenUsage

        total_prompt = sum(r.token_usage.prompt_tokens for r in result.step_results.values())
        total_completion = sum(
            r.token_usage.completion_tokens for r in result.step_results.values()
        )
        token_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

        return StepResult(
            step_id=step.id,
            status=status,
            output=child_final,
            duration_ms=duration,
            cost_usd=result.total_cost_usd,
            token_usage=token_usage,
        )
