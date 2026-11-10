"""Subworkflow step executor — runs a nested workflow."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio

from agentloom.checkpointing.base import CheckpointData
from agentloom.core.attempts import record_spend
from agentloom.core.models import SandboxConfig
from agentloom.core.redact import RedactionPolicy, redact_state
from agentloom.core.results import (
    StepResult,
    StepStatus,
    TokenUsage,
    WorkflowStatus,
)
from agentloom.exceptions import BudgetExceededError, PauseRequestedError, StepError
from agentloom.resilience.budget import BudgetEnforcer
from agentloom.resilience.retry import classify_failure
from agentloom.steps.base import BaseStep, StepContext
from agentloom.steps.tool_step import ToolStep
from agentloom.tools.builtins import with_sandbox
from agentloom.tools.registry import ToolRegistry
from agentloom.tools.sandbox import ToolSandbox


class SubworkflowStep(BaseStep):
    """Step that runs a nested workflow, passing state down and merging results up."""

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
        else:
            assert step.workflow_inline  # a subworkflow without either fails validation
            try:
                sub_workflow = WorkflowParser.from_dict(step.workflow_inline)
            except Exception as e:
                raise StepError(step.id, f"Invalid inline subworkflow: {e}") from e

        parent_state = await context.state_manager.get_state_snapshot()
        secret_inputs: list[str] = []
        if step.isolated_state:
            seed = dict(sub_workflow.state)
            try:
                rendered = ToolStep._resolve_args(
                    step.input, parent_state, step.id, "input references"
                )
                if context.redaction_policy:
                    # Patterns match key names, and ``input`` can give a secret another name
                    masked = ToolStep._resolve_args(
                        step.input,
                        redact_state(parent_state, context.redaction_policy),
                        step.id,
                        "input references",
                    )
                    secret_inputs = [key for key, value in rendered.items() if value != masked[key]]
                seed.update(rendered)
            except (KeyError, ValueError, IndexError) as e:
                raise StepError(step.id, f"Failed to resolve input: {e}", is_retryable=False) from e
            child_state = StateManager(initial_state=seed)
        else:
            # The parent's decisions answer its own gates, not a child gate that shares an id
            seed = {
                **sub_workflow.state,
                **{key: value for key, value in parent_state.items() if key != "_approval"},
            }
            child_state = StateManager(initial_state=seed)
        decisions: dict[str, str] = {}

        # A nested gate's decision is at ``_approval.<step.id>.<gate>``; the child reads it at
        # ``_approval.<gate>``, in isolated mode too.
        parent_approvals = parent_state.get("_approval", {})
        if isinstance(parent_approvals, dict):
            nested: Any = parent_approvals
            # By segment: ``set()`` splits a dotted ``step.id`` into nested keys.
            for segment in step.id.split("."):
                if not isinstance(nested, dict):
                    nested = None
                    break
                nested = nested.get(segment)
            if isinstance(nested, dict):
                decisions = dict(nested)
        for child_step_id, decision in decisions.items():
            await child_state.set(f"_approval.{child_step_id}", decision)

        # Inherit the parent's redaction, or the child's checkpoint holds its secrets in plain text.
        from agentloom.core.models import StateKeyConfig

        if context.redaction_policy:
            for pattern in context.redaction_policy.patterns:
                if pattern not in sub_workflow.state_schema:
                    sub_workflow.state_schema[pattern] = StateKeyConfig(redact=True)
            for key in secret_inputs:
                sub_workflow.state_schema[key] = StateKeyConfig(redact=True)

        sandbox = _effective_sandbox(context.sandbox_config, sub_workflow.config.sandbox)
        sub_workflow.config.sandbox = sandbox
        tool_registry = _child_registry(context.tool_registry, context.sandbox_config, sandbox)

        run_id = f"{context.run_id}.{step.id}" if context.run_id else step.id
        # Resume the child only from this step's pause, never from a checkpoint a past run left
        before = await context.state_manager.get_step_result(step.id)
        resumed = before is not None and before.status == StepStatus.PAUSED
        paused = await _paused_child(context.checkpointer, run_id) if resumed else None
        # What its pause charged already: the step reports what it spends from here
        cost_before = paused.spent_usd if paused is not None else 0.0
        # A child without its own ``budget_usd`` shares the parent's enforcer, and so its gate.
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
                tool_registry=tool_registry,
                observer=context.observer,
                on_stream_chunk=context.on_stream_chunk,
                checkpointer=context.checkpointer,
                run_id=run_id,
                budget_enforcer=child_enforcer,
            )
        else:
            # From the child's checkpoint: rerunning its finished steps would pay for them twice
            assert context.checkpointer is not None  # it holds the child's checkpoint
            engine = await WorkflowEngine.from_checkpoint(
                paused,
                checkpointer=context.checkpointer,
                provider_gateway=context.provider_gateway,
                tool_registry=tool_registry,
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
            # Defensive: run() returns a paused result (below); a raised pause is handled alike.
            qualified = f"{step.id}.{pause.step_id}" if pause.step_id else step.id
            record_spend(own_spend(), "", "", TokenUsage())  # as below, without results
            raise PauseRequestedError(qualified, str(pause), pause.expires_at) from pause
        except Exception as e:
            # Failed outside its steps: they were paid all the same
            record_spend(own_spend(), "", "", ran_tokens(await engine.state.all_step_results()))
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Subworkflow failed: {e}",
                duration_ms=duration,
                error_classification=classify_failure(e, step.retry.retryable_status_codes),
            )
        except BaseException:
            # Cut short: the attempt counts its child's tokens and, with an own budget, its spend,
            # which no result reports. A shared budget has their charges already.
            with anyio.CancelScope(shield=True):
                ran = await engine.state.all_step_results()
            record_spend(own_spend(), "", "", ran_tokens(ran))
            raise

        # An over-budget child fails without a retry: a rerun would repeat the over-budget calls.
        if result.status == WorkflowStatus.BUDGET_EXCEEDED:
            # Shared enforcer: the overrun is the parent's own, read from its counter.
            if parent_enforcer is child_enforcer and parent_enforcer is not None:
                budget = parent_enforcer.limit_usd or 0.0
                spent = parent_enforcer.spent
                record_spend(0.0, "", "", ran_tokens(result.step_results))  # charged already
                raise BudgetExceededError(budget, spent)
            # Own budget: the breach is the child's, so the parent ends FAILED and pays its spend.
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Subworkflow exceeded its own budget: {result.error or 'budget exceeded'}",
                duration_ms=(time.monotonic() - start) * 1000,
                cost_usd=result.total_cost_usd - cost_before,
                token_usage=ran_tokens(result.step_results),
                error_classification="permanent",
            )

        # Re-raise the child's pause as ``<step.id>.<gate>``, so resuming the parent reaches it.
        if result.status == WorkflowStatus.PAUSED:
            inner_paused_id: str | None = None
            for inner_id, inner_res in result.step_results.items():
                if inner_res.status == StepStatus.PAUSED:
                    inner_paused_id = inner_res.error or inner_id
                    break
            qualified = f"{step.id}.{inner_paused_id}" if inner_paused_id else step.id
            # The attempt counts the child's tokens before the gate and, with an own budget, its
            # spend, which no result reports
            record_spend(own_spend(), "", "", ran_tokens(result.step_results))
            expires_at = engine._gate_deadlines.get(inner_paused_id) if inner_paused_id else None
            raise PauseRequestedError(qualified, str(result.error or ""), expires_at)

        duration = (time.monotonic() - start) * 1000

        # A missing return key is omitted, not an error: a child may leave optional keys unset.
        child_final = result.final_state
        if step.return_keys is not None:
            child_final = {k: child_final[k] for k in step.return_keys if k in child_final}
        # The parent has the value under its own key; its patterns don't know this name
        if secret_inputs:
            child_final = redact_state(child_final, RedactionPolicy(secret_inputs))
        # Only on success, like any step: a retry would see a failed attempt's output
        if step.output and result.status == WorkflowStatus.SUCCESS:
            await context.state_manager.set(step.output, child_final)

        status = (
            StepStatus.SUCCESS if result.status == WorkflowStatus.SUCCESS else StepStatus.FAILED
        )

        # Carry the child's error, so dependents' ``upstream failure: sub`` has a visible cause.
        child_error: str | None = None
        if status == StepStatus.FAILED:
            child_error = result.error
            if not child_error:
                for inner in result.step_results.values():
                    if inner.status == StepStatus.FAILED and inner.error:
                        child_error = f"child step '{inner.step_id}': {inner.error}"
                        break
        # A permanent child failure is permanent here too: a rerun would repeat it, tools included.
        permanent = status == StepStatus.FAILED and any(
            inner.error_classification == "permanent" for inner in result.step_results.values()
        )

        token_usage = ran_tokens(result.step_results)

        # A shared enforcer has the child's charges already; the parent's ``_totals`` counts them.
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


def _common_limit(parent: list[str], child: list[str]) -> list[str] | None:
    """The entries of two lists where empty means no limit; ``None`` when they share none."""
    if not parent or not child:
        return [entry.lower() for entry in parent or child]
    theirs = {entry.lower() for entry in child}
    return [entry.lower() for entry in parent if entry.lower() in theirs] or None


def _common_dirs(parent: list[str], child: list[str]) -> list[str]:
    """The directories both lists reach: of each nested pair, the deeper one."""
    common: list[str] = []
    for mine in (Path(entry).resolve() for entry in parent):
        for theirs in (Path(entry).resolve() for entry in child):
            if theirs.is_relative_to(mine):
                common.append(str(theirs))
            elif mine.is_relative_to(theirs):
                common.append(str(mine))
    return list(dict.fromkeys(common))


def _effective_sandbox(parent: SandboxConfig, child: SandboxConfig) -> SandboxConfig:
    """The sandbox a child workflow runs under: what both allow, or the only enabled one."""
    if not parent.enabled:
        return child
    if not child.enabled:
        return parent
    domains = _common_limit(parent.allowed_domains, child.allowed_domains)
    schemes = _common_limit(parent.allowed_schemes, child.allowed_schemes)
    return SandboxConfig(
        enabled=True,
        allowed_commands=[c for c in parent.allowed_commands if c in child.allowed_commands],
        # By access, not list by list: one side's allowed_paths may be the other's readable_paths
        readable_paths=_common_dirs(
            parent.allowed_paths + parent.readable_paths,
            child.allowed_paths + child.readable_paths,
        ),
        writable_paths=_common_dirs(
            parent.allowed_paths + parent.writable_paths,
            child.allowed_paths + child.writable_paths,
        ),
        # An empty list of either means no limit, so sharing no entry has to refuse the network
        allow_network=(
            parent.allow_network
            and child.allow_network
            and domains is not None
            and schemes is not None
        ),
        allowed_domains=domains or [],
        allowed_schemes=schemes or [],
        max_write_bytes=min(
            (n for n in (parent.max_write_bytes, child.max_write_bytes) if n is not None),
            default=None,
        ),
        danger_opt_in=[d for d in parent.danger_opt_in if d in child.danger_opt_in],
        allow_internal_webhook_targets=(
            parent.allow_internal_webhook_targets and child.allow_internal_webhook_targets
        ),
        deny_internal_hosts=(
            parent.deny_internal_hosts
            or child.deny_internal_hosts
            # An empty list refuses internal hosts; the child's list must not reopen them
            or (not parent.allowed_domains and bool(child.allowed_domains))
        ),
    )


def _child_registry(registry: Any, parent: SandboxConfig, effective: SandboxConfig) -> Any:
    """The child's tools: *registry*, with its built-ins rebound when the sandbox differs."""
    if not effective.enabled or effective == parent or not isinstance(registry, ToolRegistry):
        return registry
    return with_sandbox(registry, ToolSandbox(**effective.model_dump()))


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
