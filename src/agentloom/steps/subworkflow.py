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

        # Import here to avoid circular imports
        from agentloom.core.engine import WorkflowEngine
        from agentloom.core.parser import WorkflowParser
        from agentloom.core.state import StateManager

        # Load the sub-workflow definition
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

        # Create child state with parent state snapshot
        parent_state = await context.state_manager.get_state_snapshot()
        child_state = StateManager(initial_state=parent_state)

        # Create and run nested engine
        engine = WorkflowEngine(
            workflow=sub_workflow,
            state_manager=child_state,
            provider_gateway=context.provider_gateway,
            tool_registry=context.tool_registry,
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

        # Merge child state back into parent
        # TODO: selective merge — right now dumps entire child state back
        child_final = result.final_state
        if step.output:
            await context.state_manager.set(step.output, child_final)

        status = (
            StepStatus.SUCCESS if result.status == WorkflowStatus.SUCCESS else StepStatus.FAILED
        )

        # Gather token usage from last successful step
        from agentloom.core.results import TokenUsage

        token_usage = TokenUsage()
        if result.step_results:
            last_key = list(result.step_results.keys())[-1]
            last_step = result.step_results.get(last_key)
            if last_step is not None:
                token_usage = last_step.token_usage

        return StepResult(
            step_id=step.id,
            status=status,
            output=child_final,
            duration_ms=duration,
            cost_usd=result.total_cost_usd,
            token_usage=token_usage,
        )
