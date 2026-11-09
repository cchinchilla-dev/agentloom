"""Tool execution step."""

from __future__ import annotations

import re
import time
from typing import Any

from agentloom.core.inter_agent import log_tools
from agentloom.core.results import StepResult, StepStatus
from agentloom.core.state import StateManager
from agentloom.core.templates import SafeFormatDict, build_template_vars
from agentloom.exceptions import StepError
from agentloom.resilience.retry import classify_failure
from agentloom.steps._tools import report_tool_call
from agentloom.steps.base import BaseStep, StepContext

# Format only real placeholders, so raw JSON, HTML and ``{foo: true}`` pass (no spec after ``: ``).
# ``{color:red}`` still reads as ``{name:spec}``: it needs ``template: false`` or ``{{ }}``.
_PLACEHOLDER_RE = re.compile(r"\{(?:state(?:\.|\[)|[A-Za-z_][A-Za-z0-9_]*(?:\}|![rsa]|:(?!\s)))")


class ToolStep(BaseStep):
    """Step that runs a registered tool with arguments resolved from state."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if context.tool_registry is None:
            raise StepError(step.id, "No tool registry configured", is_retryable=False)

        assert step.tool_name is not None  # a tool step without one fails validation

        try:
            tool = context.tool_registry.get(step.tool_name)
        except KeyError as e:
            raise StepError(step.id, str(e)) from e

        state_snapshot = await context.state_manager.get_state_snapshot()
        try:
            resolved_args = self._resolve_args(step.tool_args, state_snapshot, step.id)
        except (KeyError, ValueError, IndexError) as e:
            # A placeholder typo or a stray index raises in format_map: tie it to the step.
            raise StepError(step.id, f"Failed to resolve tool args: {e}", is_retryable=False) from e

        try:
            result = await tool.execute(**resolved_args)
        except Exception as e:
            log_tools([(step.tool_name, resolved_args)], [(str(e), False)])
            duration = (time.monotonic() - start) * 1000
            report_tool_call(
                context.observer,
                step_id=step.id,
                call_id=step.id,
                tool_name=step.tool_name,
                arguments=resolved_args,
                result=str(e),
                duration_ms=duration,
                success=False,
            )
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Tool '{step.tool_name}' failed: {e}",
                duration_ms=duration,
                error_classification=classify_failure(e, step.retry.retryable_status_codes),
            )

        log_tools([(step.tool_name, resolved_args)], [(result, True)])
        duration = (time.monotonic() - start) * 1000
        report_tool_call(
            context.observer,
            step_id=step.id,
            call_id=step.id,
            tool_name=step.tool_name,
            arguments=resolved_args,
            result=result,
            duration_ms=duration,
            success=True,
        )

        if step.output:
            await context.state_manager.set(step.output, result)

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=result,
            duration_ms=duration,
        )

    @staticmethod
    def _resolve_args(
        args: dict[str, Any],
        state: dict[str, Any],
        step_id: str,
        reference: str = "tool_args reference",
    ) -> dict[str, Any]:
        """Resolve argument values that reference state variables.

        * ``"state.<key>"`` — resolved by ``StateManager._resolve_key``
          (preserves object identity, not string conversion). A path
          that does not exist in state raises a non-retryable
          ``StepError`` naming the missing key.
        * Strings matching :data:`_PLACEHOLDER_RE` (``{state.foo}``,
          ``{name}``, ``{name:.2f}``, ``{name!r}``) — rendered with the
          same ``SafeFormatDict`` / ``build_template_vars`` pipeline as
          ``llm_call`` so ``tool_args: {path: "{state.user_file}"}``
          works the way authors expect. Raw JSON / HTML / code that
          happens to contain ``{`` passes through unchanged.
        * Dicts of the shape ``{value: "...", template: false}`` — escape
          hatch for the rare string that looks like a placeholder but
          must be passed through verbatim. Returns ``value`` unchanged.
        * Other dicts and lists — resolved item by item, by these same rules.
        * Everything else — passed through unchanged.
        """
        template_vars = build_template_vars(state)

        def resolve(value: Any) -> Any:
            if isinstance(value, dict):
                if value.get("template") is False and set(value) == {"template", "value"}:
                    return value["value"]
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, str):
                if value.startswith("state."):
                    path = value[len("state.") :]
                    if not StateManager.key_exists(state, path):
                        # Non-retryable: a missing state key does not appear on a retry.
                        raise StepError(
                            step_id,
                            f"{reference} 'state.{path}' but no such key exists in workflow state",
                            is_retryable=False,
                        )
                    return StateManager._resolve_key(state, path)
                if _PLACEHOLDER_RE.search(value):
                    return value.format_map(SafeFormatDict(template_vars))
            return value

        return {key: resolve(value) for key, value in args.items()}
