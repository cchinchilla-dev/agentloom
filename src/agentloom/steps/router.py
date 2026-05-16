"""Router step executor — conditional branching based on state."""

from __future__ import annotations

import ast
import time
from typing import Any

from agentloom.core.results import StepResult, StepStatus
from agentloom.exceptions import SecurityError, StepError
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
    ast.USub,
    ast.UAdd,
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
}

_ALLOWED_FUNCTIONS = set(_SAFE_BUILTINS.keys())

# Non-dunder attribute names that also reach into the Python object graph.
# Dunder attributes (anything starting with "_") are already blocked wholesale.
_BLOCKED_ATTR_NAMES = frozenset(
    {
        "mro",
        "format_map",
    }
)


def _reject_attribute(attr: str, expr_str: str) -> None:
    if attr.startswith("_"):
        raise SecurityError(
            f"Access to dunder/private attribute '{attr}' is not allowed in router expressions.",
            expression=expr_str,
        )
    if attr in _BLOCKED_ATTR_NAMES:
        raise SecurityError(
            f"Access to attribute '{attr}' is not allowed in router expressions.",
            expression=expr_str,
        )


def _reject_subscript(slice_node: ast.AST, expr_str: str) -> None:
    """Restrict subscript slices to literal int / str keys.

    Two checks combined into one gate:

    * **String constants** go through ``_reject_attribute`` so
      ``state['__class__']`` and ``state['_secret']`` are blocked exactly
      like their attribute-syntax twins.
    * **Non-constant slices** — variables, arithmetic (``'_' + 'secret'``),
      conditionals, calls — are refused outright because the validator
      cannot determine the resulting key at parse time, leaving subscript
      indirection as an end-run around the dunder gate.

    Integer constants pass through (list indexing, ``state['items'][0]``).
    Constant int/int/None slices (``state['items'][1:5]``) are permitted
    when every bound is also an integer constant.
    """
    if isinstance(slice_node, ast.Constant):
        if isinstance(slice_node.value, str):
            _reject_attribute(slice_node.value, expr_str)
            return
        if isinstance(slice_node.value, int) and not isinstance(slice_node.value, bool):
            return
        raise SecurityError(
            f"Subscript key must be a literal int or str "
            f"(got constant of type {type(slice_node.value).__name__}).",
            expression=expr_str,
        )
    if isinstance(slice_node, ast.Slice):
        for sub in (slice_node.lower, slice_node.upper, slice_node.step):
            if sub is None:
                continue
            if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, (ast.UAdd, ast.USub)):
                operand = sub.operand
                if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
                    continue
            if not (isinstance(sub, ast.Constant) and isinstance(sub.value, int)):
                raise SecurityError(
                    "Slice bounds must be integer constants (optionally with "
                    "a leading +/-) in router expressions.",
                    expression=expr_str,
                )
        return
    raise SecurityError(
        f"Subscript key must be a literal int or str "
        f"(got {type(slice_node).__name__}). Indirection through variables, "
        f"arithmetic, conditionals, or calls is not allowed.",
        expression=expr_str,
    )


def _validate_expression(expr_str: str) -> ast.Expression:
    """Parse and validate that an expression only uses allowed constructs.

    Raises:
        SecurityError if the expression contains a sandbox-bypass construct.
        ValueError for plain syntax errors.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SecurityError(
                f"Disallowed expression construct: {type(node).__name__}. "
                f"Only comparisons, boolean ops, and safe builtins are allowed.",
                expression=expr_str,
            )
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise SecurityError(
                f"Reference to dunder/private name '{node.id}' is not allowed "
                f"in router expressions.",
                expression=expr_str,
            )
        if isinstance(node, ast.Attribute):
            _reject_attribute(node.attr, expr_str)
        if isinstance(node, ast.Subscript):
            _reject_subscript(node.slice, expr_str)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_FUNCTIONS:
                    raise SecurityError(
                        f"Function '{node.func.id}' is not allowed in expressions. "
                        f"Allowed: {sorted(_ALLOWED_FUNCTIONS)}",
                        expression=expr_str,
                    )
            elif isinstance(node.func, ast.Attribute):
                # Attribute calls are allowed only if the attribute name passed
                # the dunder/blocklist check above (ast.walk visits children).
                # Still, reject any call whose receiver is not a plain
                # Name/Attribute/Subscript chain — e.g. calls on literals,
                # calls on calls, etc. — since those are not idiomatic router
                # predicates and widen the attack surface.
                receiver = node.func.value
                if not isinstance(receiver, ast.Name | ast.Attribute | ast.Subscript):
                    raise SecurityError(
                        "Attribute calls are only allowed on names, attributes, or subscripts.",
                        expression=expr_str,
                    )
            else:
                raise SecurityError(
                    "Only named function calls and attribute calls are allowed.",
                    expression=expr_str,
                )
            # Reject keyword arguments and starred unpacking — router
            # predicates never need them and they broaden the grammar.
            if node.keywords:
                raise SecurityError(
                    "Keyword arguments are not allowed in router expressions.",
                    expression=expr_str,
                )
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    raise SecurityError(
                        "Starred arguments are not allowed in router expressions.",
                        expression=expr_str,
                    )

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
    return eval(code, safe_globals)


class RouterStep(BaseStep):
    """Evaluates conditions and returns the target step ID to activate."""

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        start = time.monotonic()

        if not step.conditions and not step.default:
            raise StepError(step.id, "Router step requires 'conditions' or 'default'")

        state_snapshot = await context.state_manager.get_state_snapshot()

        # Router predicates address state through ``state.X`` only. Earlier
        # versions also flattened every top-level state key into the
        # namespace as a bare name, which let a workflow shadow safe
        # builtins (``len``, ``str``) by naming a state key the same, and
        # broadened the validator's attack surface by exposing arbitrary
        # variables that downstream subscript / indirection probes could
        # combine. The flat surface is dropped here — all reads now go
        # through ``state.X``.
        namespace: dict[str, Any] = {}

        def _wrap(value: Any) -> Any:
            """Wrap nested containers so dot/subscript chains keep working."""
            if isinstance(value, dict):
                return _DictProxy(value)
            if isinstance(value, list):
                return _ListProxy(value)
            return value

        class _DictProxy:
            """Proxy supporting both ``proxy.key`` and ``proxy['key']``."""

            __slots__ = ("_data",)
            _data: dict[str, Any]

            def __init__(self, data: dict[str, Any]) -> None:
                object.__setattr__(self, "_data", data)

            def __getattr__(self, name: str) -> Any:
                return _wrap(self._data.get(name))

            def __getitem__(self, key: Any) -> Any:
                if isinstance(key, str):
                    return _wrap(self._data.get(key))
                return _wrap(self._data[key])

            def __len__(self) -> int:
                return len(self._data)

            def __iter__(self) -> Any:
                # Yield keys (mapping protocol). Used by ``for k in
                # state.user`` and as the surface ``len()`` works against.
                return iter(self._data)

            def __contains__(self, key: Any) -> bool:
                return key in self._data

        class _ListProxy:
            """Proxy supporting ``proxy[i]`` and ``proxy[i:j]``."""

            __slots__ = ("_data",)
            _data: list[Any]

            def __init__(self, data: list[Any]) -> None:
                object.__setattr__(self, "_data", data)

            def __getitem__(self, key: Any) -> Any:
                value = self._data[key]
                if isinstance(value, list):
                    return _ListProxy(value)
                return _wrap(value)

            def __len__(self) -> int:
                return len(self._data)

            def __iter__(self) -> Any:
                return iter(_wrap(v) for v in self._data)

        class _StateProxy:
            def __getattr__(self, name: str) -> Any:
                return _wrap(state_snapshot.get(name))

            def __getitem__(self, key: Any) -> Any:
                if isinstance(key, str):
                    return _wrap(state_snapshot.get(key))
                return _wrap(state_snapshot[key])

        namespace["state"] = _StateProxy()

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

        # NOTE: first matching condition wins. no priority system yet
        target: str | None = None
        for condition in step.conditions:
            try:
                result = evaluate_expression(condition.expression, namespace)
                if result:
                    target = condition.target
                    break
            except SecurityError:
                # Sandbox bypass attempt — propagate unchanged so it surfaces
                # distinctly from ordinary step evaluation failures.
                raise
            except Exception as e:
                raise StepError(
                    step.id,
                    f"Error evaluating condition '{condition.expression}': {e}",
                ) from e

        if target is None:
            target = step.default

        if target is None:
            raise StepError(step.id, "No condition matched and no default target set")

        duration = (time.monotonic() - start) * 1000

        if step.output:
            await context.state_manager.set(step.output, target)

        return StepResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=target,
            duration_ms=duration,
        )
