"""Router step executor — conditional branching based on state."""

from __future__ import annotations

import ast
import time
from typing import Any

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import StepError
from agentloom.steps.base import BaseStep, StepContext

# AST node types allowed in router expressions
_ALLOWED_NODES = (
    ast.Expression,
    ast.Compare,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Add,
    ast.Sub,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.IfExp,
)

# Functions allowed in expressions
_SAFE_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
    "isinstance": isinstance,
    "type": type,
}

# Allowed function names for AST validation
_ALLOWED_FUNCTIONS = set(_SAFE_BUILTINS.keys())


def _validate_expression(expr_str: str) -> ast.Expression:
    """Parse and validate that an expression only uses allowed constructs.

    Raises:
        StepError-compatible ValueError if the expression is unsafe.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Disallowed expression construct: {type(node).__name__}. "
                f"Only comparisons, boolean ops, and safe builtins are allowed."
            )
        # Check function calls are to allowed names only
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_FUNCTIONS:
                    raise ValueError(
                        f"Function '{node.func.id}' is not allowed in expressions. "
                        f"Allowed: {sorted(_ALLOWED_FUNCTIONS)}"
                    )
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("Only named function calls and attribute calls are allowed")

    return tree


def evaluate_expression(expr_str: str, namespace: dict[str, Any]) -> Any:
    """Safely evaluate a router expression against a namespace.

    Args:
        expr_str: The expression string (e.g., "state.classification == 'question'").
        namespace: Variables available in the expression.

    Returns:
        The result of evaluating the expression.
    """
    tree = _validate_expression(expr_str)
    code = compile(tree, "<router_expression>", "eval")
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(_SAFE_BUILTINS)
    safe_globals.update(namespace)
    return eval(code, safe_globals)  # noqa: S307


class RouterStep(BaseStep):
    """Evaluates conditions and returns the target step ID to activate."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if not step.conditions and not step.default:
            raise StepError(step.id, "Router step requires 'conditions' or 'default'")

        state_snapshot = await context.state_manager.get_state_snapshot()

        # Build namespace with state access
        namespace: dict[str, Any] = {}
        namespace.update(state_snapshot)

        class _StateProxy:
            def __getattr__(self, name: str) -> Any:
                return state_snapshot.get(name)

        namespace["state"] = _StateProxy()

        # Also expose step results
        steps_data = state_snapshot.get("steps", {})

        class _StepsProxy:
            def __getattr__(self, name: str) -> Any:
                step_data = steps_data.get(name, {})
                if isinstance(step_data, dict):

                    class _Inner:
                        def __getattr__(self2, k: str) -> Any:
                            return step_data.get(k)

                    return _Inner()
                return step_data

        namespace["steps"] = _StepsProxy()

        # Evaluate conditions in order
        # NOTE: first matching condition wins. no priority system yet
        target: str | None = None
        for condition in step.conditions:
            try:
                result = evaluate_expression(condition.expression, namespace)
                if result:
                    target = condition.target
                    break
            except Exception as e:
                raise StepError(
                    step.id,
                    f"Error evaluating condition '{condition.expression}': {e}",
                ) from e

        # Fallback to default
        if target is None:
            target = step.default

        if target is None:
            raise StepError(step.id, "No condition matched and no default target set")

        duration = (time.monotonic() - start) * 1000

        # Store the routing decision in state
        if step.output:
            await context.state_manager.set(step.output, target)

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=target,
            duration_ms=duration,
        )
