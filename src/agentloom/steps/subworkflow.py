"""Subworkflow step executor — runs a nested workflow."""

from __future__ import annotations

import time

from agentloom.core.results import StepResult, StepStatus, WorkflowStatus
from agentloom.exceptions import StepError
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

        parent_state = await context.state_manager.get_state_snapshot()
        child_state = StateManager(initial_state=parent_state)

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
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Subworkflow failed: {e}",
                duration_ms=duration,
            )

        duration = (time.monotonic() - start) * 1000

        # TODO: selective merge — right now dumps entire child state back
        child_final = result.final_state
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
