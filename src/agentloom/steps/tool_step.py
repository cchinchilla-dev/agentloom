"""Tool execution step."""

from __future__ import annotations

import re
import time
from typing import Any

from agentloom.core.results import StepResult, StepStatus
from agentloom.core.state import StateManager
from agentloom.core.templates import SafeFormatDict, build_template_vars
from agentloom.exceptions import StepError
from agentloom.steps.base import BaseStep, StepContext

# Only trigger ``str.format_map`` on real placeholder patterns. The two
# placeholder grammars produced by ``build_template_vars`` are
# ``{state.<name>}`` / ``{state[<key>]}`` (the dotted/subscript surface
# served by ``DotAccessDict``) and ``{<bare_name>}`` / ``{<bare_name>:spec}``
# / ``{<bare_name>!r}`` (flat keys merged from ``state``). The previous
# heuristic — ``"{" in value`` — fired on every raw JSON / HTML / code
# snippet a user wrote into ``tool_args.content`` or ``tool_args.body``,
# blowing up with ``Max string recursion exceeded`` or
# ``Invalid format specifier`` for any non-trivial inline JSON. The regex
# requires either ``{state.`` / ``{state[`` or an identifier followed by
# ``}`` (bare), ``![rsa]`` (Python conversion flag), or ``:`` immediately
# followed by a non-whitespace character (the format-spec must not start
# with whitespace, so JS-object literals like ``{foo: true}`` and CSS
# rules with spaces are left untouched). A literal ``{"k": 1}`` is
# excluded because ``"`` is not a valid identifier start. CSS shapes
# without spaces (``{color:red}``) remain inherently ambiguous with
# ``{name:spec}`` placeholders — workflows shipping that style of
# content must use the ``template: false`` escape hatch or escape the
# braces as ``{{`` / ``}}``.
_PLACEHOLDER_RE = re.compile(r"\{(?:state(?:\.|\[)|[A-Za-z_][A-Za-z0-9_]*(?:\}|![rsa]|:(?!\s)))")


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
            resolved_args = self._resolve_args(step.tool_args, state_snapshot, step.id)
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
    def _resolve_args(args: dict[str, Any], state: dict[str, Any], step_id: str) -> dict[str, Any]:
        """Resolve argument values that reference state variables.

        * ``"state.<key>"`` — resolved by ``StateManager._resolve_key``
          (preserves object identity, not string conversion). A path
          that does not exist in state raises a non-retryable
          ``StepError`` naming the missing key — pre-0.5.0 it resolved
          to ``None`` and the tool surfaced a confusing downstream
          ``TypeError`` instead.
        * Strings matching :data:`_PLACEHOLDER_RE` (``{state.foo}``,
          ``{name}``, ``{name:.2f}``, ``{name!r}``) — rendered with the
          same ``SafeFormatDict`` / ``build_template_vars`` pipeline as
          ``llm_call`` so ``tool_args: {path: "{state.user_file}"}``
          works the way authors expect. Raw JSON / HTML / code that
          happens to contain ``{`` passes through unchanged.
        * Dicts of the shape ``{value: "...", template: false}`` — escape
          hatch for the rare string that looks like a placeholder but
          must be passed through verbatim. Returns ``value`` unchanged.
        * Everything else — passed through unchanged.
        """
        template_vars = build_template_vars(state)
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, dict) and value.get("template") is False and "value" in value:
                resolved[key] = value["value"]
                continue
            if isinstance(value, str):
                if value.startswith("state."):
                    path = value[len("state.") :]
                    if not StateManager.key_exists(state, path):
                        # Fail fast and non-retryably: a missing state
                        # key never appears on a retry, and a tool that
                        # receives ``None`` for a required arg surfaces
                        # an opaque ``TypeError`` four attempts later.
                        raise StepError(
                            step_id,
                            f"tool_args reference 'state.{path}' but no such key "
                            f"exists in workflow state",
                            is_retryable=False,
                        )
                    resolved[key] = StateManager._resolve_key(state, path)
                elif _PLACEHOLDER_RE.search(value):
                    resolved[key] = value.format_map(SafeFormatDict(template_vars))
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
