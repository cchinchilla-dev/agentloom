"""Subworkflow step executor — runs a nested workflow."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import anyio

from agentloom.checkpointing.base import CheckpointData
from agentloom.core.attempts import record_spend
from agentloom.core.results import (
    StepResult,
    StepStatus,
    TokenUsage,
    WorkflowStatus,
)
from agentloom.exceptions import BudgetExceededError, PauseRequestedError, StepError
from agentloom.resilience.budget import BudgetEnforcer
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
        decisions: dict[str, str] = {}

        # Resume hand-off: when the parent was paused inside this subworkflow
        # (e.g. ``sub.gate``), the CLI ``resume --approve`` stored the
        # decision under ``_approval.<this_step_id>.<child_step_id>``. The
        # child engine's approval-gate executor looks it up under the
        # unqualified ``_approval.<child_step_id>``, so we strip the prefix
        # here. Without this rewrite the child would block again on resume
        # — there'd be no way to ever complete a nested gate. Applies to
        # both isolated and non-isolated modes; the parent's approval
        # bookkeeping is metadata, not user state.
        #
        # Walking the path segment-by-segment (instead of ``dict.get(step.id)``
        # on the literal key) keeps the rewrite correct when ``step.id``
        # itself contains a ``.`` — the engine's ``from_checkpoint`` stores
        # the decision via ``state_manager.set("_approval.<step.id>...")``
        # which splits dots into nested dict keys, so a literal lookup on
        # ``"my.sub"`` would miss ``{"my": {"sub": {...}}}``.
        parent_approvals = parent_state.get("_approval", {})
        if isinstance(parent_approvals, dict):
            nested: Any = parent_approvals
            for segment in step.id.split("."):
                if not isinstance(nested, dict):
                    nested = None
                    break
                nested = nested.get(segment)
            if isinstance(nested, dict):
                decisions = dict(nested)
        for child_step_id, decision in decisions.items():
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

        # Budget propagation contract: when the child declares its own
        # ``budget_usd`` it owns the budget — give it a fresh enforcer
        # scoped to that limit. When the child has none, inherit the
        # parent's enforcer so child charges land in the parent's counter
        # and the parent gate trips at the correct step. The
        # ``budget_enforcer`` argument to ``WorkflowEngine`` exists for
        # exactly this hand-off — when ``None`` the engine builds a
        # fresh one, so the default top-level execution path is
        # unchanged.
        run_id = f"{context.run_id}.{step.id}" if context.run_id else step.id
        # Only the step paused at its child's gate resumes the child: a new run that reuses
        # the run_id starts it afresh, whatever checkpoint an earlier run left
        before = await context.state_manager.get_step_result(step.id)
        resumed = before is not None and before.status == StepStatus.PAUSED
        paused = await _paused_child(context.checkpointer, run_id) if resumed else None
        # What its pause charged already: the step reports what it spends from here
        cost_before = paused.spent_usd if paused is not None else 0.0
        parent_enforcer = getattr(context, "budget_enforcer", None)
        own_enforcer = (
            BudgetEnforcer(sub_workflow.config.budget_usd, spent_usd=cost_before)
            if sub_workflow.config.budget_usd is not None
            else None
        )
        child_enforcer = own_enforcer if own_enforcer is not None else parent_enforcer

        if paused is None:
            engine = WorkflowEngine(
                workflow=sub_workflow,
                state_manager=child_state,
                provider_gateway=context.provider_gateway,
                tool_registry=context.tool_registry,
                observer=context.observer,
                on_stream_chunk=context.on_stream_chunk,
                checkpointer=context.checkpointer,
                run_id=run_id,
                budget_enforcer=child_enforcer,
            )
        else:
            # Resumed at a gate of its child: the child goes on from its own checkpoint,
            # its finished steps kept, instead of running them and paying them again (#120)
            assert context.checkpointer is not None  # it holds the child's checkpoint
            engine = await WorkflowEngine.from_checkpoint(
                paused,
                checkpointer=context.checkpointer,
                provider_gateway=context.provider_gateway,
                tool_registry=context.tool_registry,
                observer=context.observer,
                on_stream_chunk=context.on_stream_chunk,
                approval_decisions=decisions,
                budget_enforcer=child_enforcer,
            )
        tokens_before = _child_tokens(await engine.state.all_step_results())

        def own_spend() -> float:
            return own_enforcer.spent - cost_before if own_enforcer is not None else 0.0

        def ran_tokens(results: Mapping[str, StepResult]) -> TokenUsage:
            # This attempt's: its child's steps cut short with no result included
            return _since(_child_tokens({**results, **engine._cut}), tokens_before)

        try:
            result = await engine.run()
        except PauseRequestedError as pause:
            # Defensive: child engines today return a paused
            # ``WorkflowResult`` rather than re-raising, but third-party
            # engines or future refactors might propagate the exception —
            # treat it symmetrically with the result-based path below.
            qualified = f"{step.id}.{pause.step_id}" if pause.step_id else step.id
            record_spend(own_spend(), "", "", TokenUsage())  # as below, without results
            raise PauseRequestedError(qualified, str(pause)) from pause
        except Exception as e:
            # Failed outside its steps: they were paid all the same (#120)
            record_spend(own_spend(), "", "", ran_tokens(await engine.state.all_step_results()))
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Subworkflow failed: {e}",
                duration_ms=duration,
            )
        except BaseException:
            # Cut short: its attempt still counts the tokens of the steps its child ran, and
            # pays what they spent when the child has a budget of its own: no result reports
            # either (#120). A shared budget has their charges already.
            with anyio.CancelScope(shield=True):
                ran = await engine.state.all_step_results()
            record_spend(own_spend(), "", "", ran_tokens(ran))
            raise

        # Budget-exhausted child: re-running the subworkflow would only
        # repeat the same over-budget provider calls, so the step must
        # fail without consuming the parent's retry budget. Two cases:
        #
        # * Shared enforcer (child declared no ``budget_usd``): the
        #   child's charges already landed on the parent counter, so
        #   re-raise ``BudgetExceededError`` and let the parent's
        #   terminal classifier surface the overrun as its own. The
        #   parent enforcer holds the canonical ``spent`` reading, so we
        #   use that rather than ``result.total_cost_usd``.
        # * Own enforcer (child declared its own ``budget_usd``): the
        #   breach is the child's, not the parent's — surface a
        #   permanent failure so the parent ends FAILED (not
        #   BUDGET_EXCEEDED) but still fails fast. Pre-0.5.0 this
        #   returned a plain FAILED result, which the engine's
        #   soft-failure path retried 4× — re-billing the child each
        #   time. It reports what the child spent, which the parent
        #   charges (#120).
        if result.status == WorkflowStatus.BUDGET_EXCEEDED:
            if parent_enforcer is not None and parent_enforcer is child_enforcer:
                budget = parent_enforcer.limit_usd or 0.0
                spent = parent_enforcer.spent
                record_spend(0.0, "", "", ran_tokens(result.step_results))  # charged already
                raise BudgetExceededError(budget, spent)
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Subworkflow exceeded its own budget: {result.error or 'budget exceeded'}",
                duration_ms=(time.monotonic() - start) * 1000,
                cost_usd=result.total_cost_usd - cost_before,
                token_usage=ran_tokens(result.step_results),
                error_classification="permanent",
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
            # The parent's attempt counts the tokens of its child's steps before the gate, and
            # pays what they spent when the child has a budget of its own: no result of its
            # own reports either (#120)
            record_spend(own_spend(), "", "", ran_tokens(result.step_results))
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
        # Only on success, like any step: a retry would see a failed attempt's output (#185)
        if step.output and result.status == WorkflowStatus.SUCCESS:
            await context.state_manager.set(step.output, child_final)

        status = (
            StepStatus.SUCCESS if result.status == WorkflowStatus.SUCCESS else StepStatus.FAILED
        )

        # When the child workflow ended FAILED, surface the underlying error
        # on the parent's step result. Without this, the parent's cascade-
        # skip block (engine.py) would point dependents at ``upstream
        # failure: sub`` while ``sub`` itself reported ``error=None`` — the
        # operator would have to crack open the child's step_results to
        # figure out what actually broke. Prefer ``result.error`` (the
        # child's surfaced error string) and fall back to the first failed
        # child step's error so something always lands here.
        child_error: str | None = None
        if status == StepStatus.FAILED:
            child_error = result.error
            if not child_error:
                for inner in result.step_results.values():
                    if inner.status == StepStatus.FAILED and inner.error:
                        child_error = f"child step '{inner.step_id}': {inner.error}"
                        break
        # A child step that failed for good (a stale replay, a sandbox block) fails the
        # subworkflow for good: running the child again would repeat it, tools included.
        permanent = status == StepStatus.FAILED and any(
            inner.error_classification == "permanent" for inner in result.step_results.values()
        )

        token_usage = ran_tokens(result.step_results)

        # Avoid double-charging the parent enforcer in the shared-budget
        # case. When the child ran against the parent's ``BudgetEnforcer``,
        # every child step already charged the parent counter via
        # ``self._budget.charge(...)`` inside the child engine's
        # ``_execute_step``. Returning ``result.total_cost_usd`` here
        # would make the parent's own ``_execute_step`` re-charge the
        # same amount on completion of the subworkflow step, so a parent
        # ``budget_usd: 0.25`` and a child that legitimately spent
        # ``$0.20`` would surface ``budget exceeded: spent $0.40`` even
        # though the real spend fit. Report zero on the step result in
        # the shared case so the charge is not duplicated — the parent
        # engine still surfaces the true total because it reports
        # ``WorkflowResult.total_cost_usd`` from the enforcer's recorded
        # spend, which already carries the child charges. The child's
        # own-budget case (fresh enforcer) reports the rolled-up cost so
        # the parent's enforcer learns about the subworkflow spend.
        reported_cost = (
            0.0
            if parent_enforcer is not None and parent_enforcer is child_enforcer
            else result.total_cost_usd - cost_before
        )

        return StepResult(
            step_id=step.id,
            status=status,
            output=child_final,
            error=child_error,
            duration_ms=duration,
            cost_usd=reported_cost,
            token_usage=token_usage,
            error_classification="permanent" if permanent else None,
        )


async def _paused_child(checkpointer: Any, run_id: str) -> CheckpointData | None:
    """The checkpoint of a child that paused at one of its gates, when the step resumes it."""
    if checkpointer is None:
        return None
    try:
        saved: CheckpointData = await checkpointer.load(run_id)
    except (KeyError, ValueError):  # none, or a run_id it refused to save under (a step id)
        return None
    return saved if saved.status == WorkflowStatus.PAUSED.value else None


def _since(now: TokenUsage, before: TokenUsage) -> TokenUsage:
    """The tokens a resumed child's steps used after its pause."""
    return TokenUsage(
        prompt_tokens=now.prompt_tokens - before.prompt_tokens,
        completion_tokens=now.completion_tokens - before.completion_tokens,
        total_tokens=now.total_tokens - before.total_tokens,
        reasoning_tokens=now.reasoning_tokens - before.reasoning_tokens,
    )


def _child_tokens(step_results: Mapping[str, StepResult]) -> TokenUsage:
    """The tokens of a child workflow's steps, their reasoning and totals as reported."""
    return sum((r.token_usage for r in step_results.values()), TokenUsage())
