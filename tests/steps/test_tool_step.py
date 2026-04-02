"""Tests for tool step executor."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.core.models import StepDefinition, StepType, WorkflowConfig
from agentloom.core.results import StepStatus
from agentloom.core.state import StateManager
from agentloom.steps.base import StepContext
from agentloom.steps.tool_step import ToolStep
from agentloom.tools.registry import ToolRegistry
from tests.conftest import MockTool


class TestResolveArgs:
    def test_literal_values_pass_through(self) -> None:
        result = ToolStep._resolve_args({"url": "https://example.com"}, {})
        assert result == {"url": "https://example.com"}

    def test_state_reference_resolved(self) -> None:
        state = {"user_url": "https://test.com"}
        result = ToolStep._resolve_args({"url": "state.user_url"}, state)
        assert result == {"url": "https://test.com"}

    def test_nested_state_reference(self) -> None:
        state = {"config": {"api_url": "https://api.test.com"}}
        result = ToolStep._resolve_args({"url": "state.config.api_url"}, state)
        assert result == {"url": "https://api.test.com"}

    def test_missing_state_reference_returns_none(self) -> None:
        result = ToolStep._resolve_args({"url": "state.missing"}, {})
        assert result == {"url": None}

    def test_non_string_values_pass_through(self) -> None:
        result = ToolStep._resolve_args({"count": 5, "flag": True}, {})
        assert result == {"count": 5, "flag": True}

    def test_mixed_literal_and_state(self) -> None:
        state = {"name": "Alice"}
        result = ToolStep._resolve_args(
            {"greeting": "hello", "name": "state.name"},
            state,
        )
        assert result == {"greeting": "hello", "name": "Alice"}

    def test_state_reference_with_index(self) -> None:
        state = {"items": ["first", "second"]}
        result = ToolStep._resolve_args({"val": "state.items[0]"}, state)
        assert result == {"val": "first"}

    def test_state_reference_with_nested_index(self) -> None:
        state = {"items": [{"name": "Alice"}, {"name": "Bob"}]}
        result = ToolStep._resolve_args({"val": "state.items[1].name"}, state)
        assert result == {"val": "Bob"}


class TestToolStep:
    @pytest.fixture
    def step(self) -> ToolStep:
        return ToolStep()

    def _make_context(
        self,
        step_def: StepDefinition,
        state: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> StepContext:
        return StepContext(
            step_definition=step_def,
            state_manager=StateManager(initial_state=state or {}),
            tool_registry=registry,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )

    async def test_no_registry_raises(self, step: ToolStep) -> None:
        ctx = self._make_context(
            StepDefinition(id="s", type=StepType.TOOL, tool_name="mock_tool"),
            registry=None,
        )
        with pytest.raises(Exception, match="No tool registry"):
            await step.execute(ctx)

    async def test_no_tool_name_raises(self, step: ToolStep) -> None:
        registry = ToolRegistry()
        ctx = self._make_context(
            StepDefinition(id="s", type=StepType.TOOL),
            registry=registry,
        )
        with pytest.raises(Exception, match="requires a 'tool_name'"):
            await step.execute(ctx)

    async def test_missing_tool_raises(self, step: ToolStep) -> None:
        registry = ToolRegistry()
        ctx = self._make_context(
            StepDefinition(id="s", type=StepType.TOOL, tool_name="nonexistent"),
            registry=registry,
        )
        with pytest.raises(Exception, match="nonexistent"):
            await step.execute(ctx)

    async def test_successful_execution(self, step: ToolStep) -> None:
        mock = MockTool(result={"data": "test"})
        registry = ToolRegistry()
        registry.register(mock)

        ctx = self._make_context(
            StepDefinition(
                id="t",
                type=StepType.TOOL,
                tool_name="mock_tool",
                tool_args={"input": "hello"},
                output="result",
            ),
            registry=registry,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.output == {"data": "test"}
        assert mock.calls == [{"input": "hello"}]

    async def test_tool_failure_returns_failed_result(self, step: ToolStep) -> None:
        class FailingTool(MockTool):
            async def execute(self, **kwargs: Any) -> Any:
                raise RuntimeError("tool broke")

        registry = ToolRegistry()
        registry.register(FailingTool())

        ctx = self._make_context(
            StepDefinition(
                id="t",
                type=StepType.TOOL,
                tool_name="mock_tool",
                tool_args={"input": "x"},
            ),
            registry=registry,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.FAILED
        assert "tool broke" in (result.error or "")

    async def test_output_stored_in_state(self, step: ToolStep) -> None:
        mock = MockTool(result="stored_value")
        registry = ToolRegistry()
        registry.register(mock)

        state_mgr = StateManager()
        ctx = StepContext(
            step_definition=StepDefinition(
                id="t",
                type=StepType.TOOL,
                tool_name="mock_tool",
                tool_args={"input": "x"},
                output="my_output",
            ),
            state_manager=state_mgr,
            tool_registry=registry,
            workflow_config=WorkflowConfig(),
            workflow_model="mock-model",
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        stored = await state_mgr.get("my_output")
        assert stored == "stored_value"
