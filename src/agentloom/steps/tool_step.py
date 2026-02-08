"""Tool execution step."""

from __future__ import annotations

import time
from typing import Any

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import StepError
from agentloom.steps.base import BaseStep, StepContext


class ToolStep(BaseStep):
    """Executes a registered tool with arguments resolved from state."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if context.tool_registry is None:
            raise StepError(step.id, "No tool registry configured")

        if not step.tool_name:
            raise StepError(step.id, "Tool step requires a 'tool_name' field")

        # Get the tool
        try:
            tool = context.tool_registry.get(step.tool_name)
        except KeyError as e:
            raise StepError(step.id, str(e)) from e

        # Resolve tool arguments from state
        resolved_args = self._resolve_args(step.tool_args, context.state_manager.state)

        # Execute the tool
        try:
            result = await tool.execute(**resolved_args)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Tool '{step.tool_name}' failed: {e}",
                duration_ms=duration,
            )

        duration = (time.monotonic() - start) * 1000

        # Store output
        if step.output:
            await context.state_manager.set(step.output, result)

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=result,
            duration_ms=duration,
        )

    @staticmethod
    def _resolve_args(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Resolve argument values that reference state variables.

        String values starting with 'state.' are resolved from the state dict.
        Other values are passed through as-is.
        """
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("state."):
                path = value[len("state.") :]
                parts = path.split(".")
                current: Any = state
                for part in parts:
                    if isinstance(current, dict):
                        current = current.get(part)
                    else:
                        current = None
                        break
                resolved[key] = current
            else:
                resolved[key] = value
        return resolved
