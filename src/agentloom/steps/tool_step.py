"""Tool execution step."""

from __future__ import annotations

import time
from typing import Any

from agentloom.core.results import StepResult, StepStatus
from agentloom.core.state import StateManager
from agentloom.core.templates import SafeFormatDict, build_template_vars
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

        try:
            tool = context.tool_registry.get(step.tool_name)
        except KeyError as e:
            raise StepError(step.id, str(e)) from e

        state_snapshot = await context.state_manager.get_state_snapshot()
        try:
            resolved_args = self._resolve_args(step.tool_args, state_snapshot)
        except (KeyError, ValueError, IndexError) as e:
            # Template rendering can raise on a typo in a placeholder, a
            # literal ``{`` in a JSON snippet, or a stray index. Surface
            # these as ``StepError`` tied to the step id instead of letting
            # the raw formatting exception bubble up.
            raise StepError(step.id, f"Failed to resolve tool args: {e}") from e

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

        * ``"state.<key>"`` — resolved by ``StateManager._resolve_key``
          (preserves object identity, not string conversion).
        * Strings with ``{...}`` placeholders — rendered with the same
          ``SafeFormatDict`` / ``build_template_vars`` pipeline as
          ``llm_call`` so ``tool_args: {path: "{state.user_file}"}`` works
          the way authors expect.
        * Everything else — passed through unchanged.
        """
        template_vars = build_template_vars(state)
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                if value.startswith("state."):
                    path = value[len("state.") :]
                    resolved[key] = StateManager._resolve_key(state, path)
                elif "{" in value:
                    resolved[key] = value.format_map(SafeFormatDict(template_vars))
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
