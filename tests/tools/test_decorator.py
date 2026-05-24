"""Tests for the @tool decorator."""

from __future__ import annotations

from agentloom.tools.decorator import tool


class TestToolDecorator:
    """Test the @tool decorator for creating tools from functions."""

    def test_creates_tool_with_name(self) -> None:
        @tool(name="my_tool", description="A test tool")
        async def my_func(x: str) -> str:
            return x

        assert my_func.name == "my_tool"

    def test_creates_tool_with_description(self) -> None:
        @tool(name="my_tool", description="Does something")
        async def my_func(x: str) -> str:
            return x

        assert my_func.description == "Does something"

    def test_default_name_from_function(self) -> None:
        @tool()
        async def fetch_data(url: str) -> str:
            return url

        assert fetch_data.name == "fetch_data"

    def test_description_from_docstring(self) -> None:
        @tool()
        async def fetch_data(url: str) -> str:
            """Fetches data from a URL."""
            return url

        assert fetch_data.description == "Fetches data from a URL."

    async def test_tool_is_callable(self) -> None:
        @tool(name="echo")
        async def echo(message: str) -> str:
            return message

        result = await echo.execute(message="hello")
        assert result == "hello"


class TestSchemaGeneration:
    """Test JSON Schema generation from type hints."""

    def test_string_parameter(self) -> None:
        @tool(name="test_tool")
        async def func(name: str) -> str:
            return name

        schema = func.parameters_schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"

    def test_integer_parameter(self) -> None:
        @tool(name="test_tool")
        async def func(count: int) -> int:
            return count

        schema = func.parameters_schema
        assert schema["properties"]["count"]["type"] == "integer"

    def test_float_parameter(self) -> None:
        @tool(name="test_tool")
        async def func(score: float) -> float:
            return score

        schema = func.parameters_schema
        assert schema["properties"]["score"]["type"] == "number"

    def test_boolean_parameter(self) -> None:
        @tool(name="test_tool")
        async def func(flag: bool) -> bool:
            return flag

        schema = func.parameters_schema
        assert schema["properties"]["flag"]["type"] == "boolean"

    def test_required_parameters(self) -> None:
        @tool(name="test_tool")
        async def func(required_param: str) -> str:
            return required_param

        schema = func.parameters_schema
        assert "required" in schema
        assert "required_param" in schema["required"]

    def test_optional_parameter_with_default(self) -> None:
        @tool(name="test_tool")
        async def func(name: str, timeout: int = 30) -> str:
            return name

        schema = func.parameters_schema
        # 'name' should be required, 'timeout' should not be
        assert "required" in schema
        assert "name" in schema["required"]
        assert "timeout" not in schema["required"]
        assert schema["properties"]["timeout"]["default"] == 30

    def test_multiple_parameters(self) -> None:
        @tool(name="search")
        async def func(query: str, limit: int = 10, verbose: bool = False) -> str:
            return query

        schema = func.parameters_schema
        assert len(schema["properties"]) == 3
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "verbose" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_no_parameters(self) -> None:
        @tool(name="noop")
        async def func() -> str:
            return "done"

        schema = func.parameters_schema
        assert schema["type"] == "object"
        assert schema["properties"] == {}

    def test_list_parameter(self) -> None:
        @tool(name="test_tool")
        async def func(items: list) -> list:
            return items

        schema = func.parameters_schema
        assert schema["properties"]["items"]["type"] == "array"

    def test_dict_parameter(self) -> None:
        @tool(name="test_tool")
        async def func(data: dict) -> dict:
            return data

        schema = func.parameters_schema
        assert schema["properties"]["data"]["type"] == "object"


class TestSyncFunctionWrapping:
    """F45: a sync ``@tool`` function is auto-wrapped onto a worker
    thread. Pre-0.5.0 it raised ``TypeError: object ... can't be used in
    'await' expression`` the first time the tool ran."""

    async def test_sync_function_wrapped(self) -> None:
        @tool(name="sync_t")
        def sync_fn(x: str) -> str:
            return f"got {x}"

        result = await sync_fn.execute(x="hi")
        assert result == "got hi"

    async def test_async_function_still_works(self) -> None:
        @tool(name="async_t")
        async def async_fn(x: str) -> str:
            return f"async {x}"

        assert await async_fn.execute(x="hi") == "async hi"

    async def test_sync_function_schema_generated_from_original(self) -> None:
        # The schema must come from the original signature, not the
        # ``**kwargs``-only async wrapper.
        @tool(name="sync_schema")
        def sync_fn(name: str, count: int = 3) -> str:
            return name

        schema = sync_fn.parameters_schema
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["required"] == ["name"]


class TestGenericTypeSchema:
    """F46: generic type hints (``list[int]``, ``Optional[T]``,
    ``dict[str, V]``) must produce structurally correct JSON Schema.
    Pre-0.5.0 they all degraded to ``{"type": "string"}``."""

    def test_list_int_hint(self) -> None:
        @tool(name="t")
        async def t(items: list[int]) -> str:
            return ""

        prop = t.parameters_schema["properties"]["items"]
        assert prop == {"type": "array", "items": {"type": "integer"}}

    def test_list_str_hint(self) -> None:
        @tool(name="t")
        async def t(tags: list[str]) -> str:
            return ""

        prop = t.parameters_schema["properties"]["tags"]
        assert prop == {"type": "array", "items": {"type": "string"}}

    def test_optional_hint(self) -> None:
        @tool(name="t")
        async def t(name: str | None = None) -> str:
            return ""

        # Optional[str] exposes str's schema; "required" conveys None-ness.
        prop = t.parameters_schema["properties"]["name"]
        assert prop["type"] == "string"
        assert "name" not in t.parameters_schema.get("required", [])

    def test_dict_hint(self) -> None:
        @tool(name="t")
        async def t(meta: dict[str, int]) -> str:
            return ""

        prop = t.parameters_schema["properties"]["meta"]
        assert prop == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_nested_generic_hint(self) -> None:
        @tool(name="t")
        async def t(rows: list[dict[str, int]]) -> str:
            return ""

        prop = t.parameters_schema["properties"]["rows"]
        assert prop == {
            "type": "array",
            "items": {"type": "object", "additionalProperties": {"type": "integer"}},
        }

    def test_union_multi_type_hint(self) -> None:
        @tool(name="t")
        async def t(val: int | str) -> str:
            return ""

        prop = t.parameters_schema["properties"]["val"]
        assert "anyOf" in prop
        assert {"type": "integer"} in prop["anyOf"]
        assert {"type": "string"} in prop["anyOf"]


class TestPythonToJsonSchemaBranches:
    """Direct coverage of ``_python_to_json_schema`` edge branches that
    a normal ``@tool`` signature rarely exercises."""

    def test_any_hint_is_unconstrained(self) -> None:
        from typing import Any

        from agentloom.tools.decorator import _python_to_json_schema

        assert _python_to_json_schema(Any) == {}

    def test_none_hint_is_null_type(self) -> None:
        from agentloom.tools.decorator import _python_to_json_schema

        assert _python_to_json_schema(None) == {"type": "null"}
        assert _python_to_json_schema(type(None)) == {"type": "null"}

    def test_bare_list_and_dict(self) -> None:
        from agentloom.tools.decorator import _python_to_json_schema

        assert _python_to_json_schema(list) == {"type": "array"}
        assert _python_to_json_schema(dict) == {"type": "object"}

    def test_list_without_args_defaults_to_any_items(self) -> None:
        from agentloom.tools.decorator import _python_to_json_schema

        assert _python_to_json_schema(list[object]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_unknown_type_falls_back_to_string(self) -> None:
        from agentloom.tools.decorator import _python_to_json_schema

        class Custom:
            pass

        assert _python_to_json_schema(Custom) == {"type": "string"}

    async def test_untyped_parameter_defaults_to_string(self) -> None:
        @tool(name="untyped")
        async def t(x) -> str:  # type: ignore[no-untyped-def]
            return str(x)

        assert t.parameters_schema["properties"]["x"] == {"type": "string"}

    def test_var_keyword_parameter_is_skipped(self) -> None:
        # ``**kwargs`` is part of the signature but not a real input the
        # model can fill — the schema must omit it.
        @tool(name="vkw")
        async def t(name: str, **kwargs: object) -> str:
            return name

        schema = t.parameters_schema
        assert "kwargs" not in schema["properties"]
        assert "name" in schema["properties"]

    def test_var_positional_parameter_is_skipped(self) -> None:
        # Same contract for ``*args``: it never lands in the JSON Schema.
        @tool(name="vpos")
        async def t(name: str, *args: object) -> str:
            return name

        schema = t.parameters_schema
        assert "args" not in schema["properties"]
        assert "name" in schema["properties"]

    def test_unresolvable_forward_reference_falls_back_to_string(self) -> None:
        # ``get_type_hints`` raises NameError when an annotation cannot be
        # resolved (forward reference to a name that never gets defined).
        # The decorator must catch that and fall back to per-param schema
        # — pre-0.5.0 this surfaced raw at import time.
        from agentloom.tools.decorator import _generate_schema

        def fn(x: "DefinitelyNotARealName") -> str:  # type: ignore[name-defined]  # noqa: F821, UP037
            return ""

        schema = _generate_schema(fn)
        # Without resolvable hints the parameter still appears, defaulted
        # to the safe string shape.
        assert schema["properties"]["x"] == {"type": "string"}
