"""Decorator for creating tools from plain functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from agentloom.tools.base import BaseTool


def tool(
    name: str | None = None,
    description: str = "",
) -> Callable[[Callable[..., Any]], BaseTool]:
    """Decorator to create a BaseTool from a plain async function.

    Auto-generates a JSON Schema for parameters from type hints.

    Usage:
        @tool(name="fetch_url", description="Fetches content from a URL")
        async def fetch_url(url: str, timeout: int = 30) -> str:
            ...
    """

    def decorator(func: Callable[..., Any]) -> BaseTool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # Generate schema from type hints
        schema = _generate_schema(func)

        class DecoratedTool(BaseTool):
            async def execute(self, **kwargs: Any) -> Any:
                return await func(**kwargs)

        instance = DecoratedTool()
        instance.name = tool_name
        instance.description = tool_desc.strip()
        instance.parameters_schema = schema

        return instance

    return decorator


# doesn't handle Union, Optional[T], or List[T] — just basic types for now
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


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

        prop: dict[str, Any] = {}
        hint = hints.get(param_name)

        if hint is not None:
            json_type = _TYPE_MAP.get(hint, "string")
            prop["type"] = json_type
        else:
            prop["type"] = "string"

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
