"""Decorator for creating tools from plain functions."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

import anyio

from agentloom.tools.base import BaseTool


def tool(
    name: str | None = None,
    description: str = "",
) -> Callable[[Callable[..., Any]], BaseTool]:
    """Decorator to create a BaseTool from a plain function.

    Both ``async def`` and plain ``def`` functions are accepted. A sync
    function is automatically off-loaded to a worker thread via
    ``anyio.to_thread.run_sync`` so a blocking tool body never stalls the
    event loop — pre-0.5.0 a sync ``@tool`` raised ``TypeError: object
    ... can't be used in 'await' expression`` the first time it ran.

    Auto-generates a JSON Schema for parameters from type hints,
    including the generic shapes (``list[int]``, ``dict[str, V]``,
    ``Optional[T]``) that used to degrade silently to ``"string"``.

    Usage:
        @tool(name="fetch_url", description="Fetches content from a URL")
        async def fetch_url(url: str, timeout: int = 30) -> str:
            ...
    """

    def decorator(func: Callable[..., Any]) -> BaseTool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # Generate the schema from the ORIGINAL function — the sync
        # wrapper below carries ``**kwargs`` only, which would yield an
        # empty schema.
        schema = _generate_schema(func)

        if inspect.iscoroutinefunction(func):
            async_func = func
        else:
            # Off-load the blocking call to a worker thread so the event
            # loop keeps serving sibling steps. ``functools.partial``
            # binds the kwargs because ``run_sync`` passes positional
            # args only.
            import functools

            sync_func = func

            async def async_func(**kwargs: Any) -> Any:
                return await anyio.to_thread.run_sync(functools.partial(sync_func, **kwargs))

        class DecoratedTool(BaseTool):
            async def execute(self, **kwargs: Any) -> Any:
                return await async_func(**kwargs)

        instance = DecoratedTool()
        instance.name = tool_name
        instance.description = tool_desc.strip()
        instance.parameters_schema = schema

        return instance

    return decorator


# Scalar Python types → JSON Schema type names. Container / generic
# types are handled structurally by ``_python_to_json_schema``.
_SCALAR_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _python_to_json_schema(hint: Any) -> dict[str, Any]:
    """Translate a Python type hint to a JSON Schema fragment.

    Handles the generic shapes a tool author naturally writes —
    ``list[int]``, ``dict[str, V]``, ``Optional[T]`` / ``T | None``,
    nested combinations — that pre-0.5.0 silently degraded to
    ``{"type": "string"}``, leaving the model with a schema it could not
    satisfy and a tool that crashed on the wrong-shaped argument.
    """
    if hint is Any:
        # No constraint — an empty schema accepts any JSON value.
        return {}
    if hint is None or hint is type(None):
        return {"type": "null"}

    origin = get_origin(hint)
    if origin in (list, set, frozenset, tuple):
        args = get_args(hint)
        item = args[0] if args else Any
        return {"type": "array", "items": _python_to_json_schema(item)}
    if origin is dict:
        args = get_args(hint)
        value = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": _python_to_json_schema(value)}
    if origin in (typing.Union, types.UnionType):
        # ``Optional[T]`` / ``T | None`` → expose T's schema directly; the
        # ``required`` list already conveys whether the parameter may be
        # omitted. A genuine multi-type union becomes ``anyOf`` so the
        # model still sees every accepted shape.
        non_none = [a for a in get_args(hint) if a is not type(None)]
        if len(non_none) == 1:
            return _python_to_json_schema(non_none[0])
        if non_none:
            return {"anyOf": [_python_to_json_schema(a) for a in non_none]}
        return {"type": "null"}

    if isinstance(hint, type):
        json_type = _SCALAR_TYPE_MAP.get(hint)
        if json_type is not None:
            return {"type": json_type}
        # Bare ``list`` / ``dict`` without parameters.
        if hint is list:
            return {"type": "array"}
        if hint is dict:
            return {"type": "object"}

    # Unknown / unresolvable hint — fall back to string, the safest
    # cross-provider default.
    return {"type": "string"}


def _generate_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate a JSON Schema from a function's type hints and signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "kwargs", "args"):
            continue

        hint = hints.get(param_name)
        prop = _python_to_json_schema(hint) if hint is not None else {"type": "string"}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default

        properties[param_name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema
