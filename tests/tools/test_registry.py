"""Tests for the ToolRegistry module."""

from __future__ import annotations

import pytest

from agentloom.tools.registry import ToolRegistry
from tests.conftest import MockTool


class TestToolRegistration:
    """Test registering tools in the registry."""

    def test_register_tool(self) -> None:
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        # Should not raise
        retrieved = registry.get("mock_tool")
        assert retrieved is tool

    def test_register_tool_without_name_raises(self) -> None:
        registry = ToolRegistry()
        tool = MockTool()
        tool.name = ""
        with pytest.raises(ValueError, match="name"):
            registry.register(tool)

    def test_register_multiple_tools(self) -> None:
        registry = ToolRegistry()
        tool1 = MockTool(result="r1")
        tool1.name = "tool_one"
        tool2 = MockTool(result="r2")
        tool2.name = "tool_two"

        registry.register(tool1)
        registry.register(tool2)

        assert registry.get("tool_one") is tool1
        assert registry.get("tool_two") is tool2

    def test_overwrite_tool(self) -> None:
        registry = ToolRegistry()
        tool_v1 = MockTool(result="v1")
        tool_v2 = MockTool(result="v2")

        registry.register(tool_v1)
        registry.register(tool_v2)

        assert registry.get("mock_tool") is tool_v2


class TestToolGet:
    """Test getting tools from the registry."""

    def test_get_existing_tool(self, tool_registry: ToolRegistry) -> None:
        tool = tool_registry.get("mock_tool")
        assert tool is not None
        assert tool.name == "mock_tool"

    def test_get_missing_tool_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_tool")

    def test_get_missing_tool_shows_available(self) -> None:
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        with pytest.raises(KeyError, match="mock_tool"):
            registry.get("other_tool")


class TestToolList:
    """Test listing tools in the registry."""

    def test_list_empty_registry(self) -> None:
        registry = ToolRegistry()
        tools = registry.list()
        assert tools == []

    def test_list_single_tool(self, tool_registry: ToolRegistry) -> None:
        tools = tool_registry.list()
        assert len(tools) == 1
        assert tools[0].name == "mock_tool"
        assert tools[0].description == "A mock tool for testing"

    def test_list_multiple_tools(self) -> None:
        registry = ToolRegistry()
        t1 = MockTool()
        t1.name = "alpha"
        t1.description = "Alpha tool"
        t2 = MockTool()
        t2.name = "beta"
        t2.description = "Beta tool"

        registry.register(t1)
        registry.register(t2)

        tools = registry.list()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"alpha", "beta"}

    def test_list_returns_tool_info_with_schema(self, tool_registry: ToolRegistry) -> None:
        tools = tool_registry.list()
        info = tools[0]
        assert info.schema is not None
        assert "properties" in info.schema
