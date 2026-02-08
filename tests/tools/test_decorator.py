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
