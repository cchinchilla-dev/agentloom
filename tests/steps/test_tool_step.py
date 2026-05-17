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


class TestToolArgsTemplating:
    """Regression: tool_args with `{state.x}` placeholders must be rendered."""

    async def test_tool_args_templated_like_llm_call(self) -> None:
        from agentloom.core.models import (
            StepDefinition,
            StepType,
            WorkflowConfig,
            WorkflowDefinition,
        )
        from agentloom.core.state import StateManager
        from agentloom.steps.base import StepContext
        from agentloom.steps.tool_step import ToolStep
        from agentloom.tools.base import BaseTool
        from agentloom.tools.registry import ToolRegistry

        captured: dict[str, object] = {}

        class CaptureTool(BaseTool):
            name = "capture"
            description = "x"
            parameters_schema = {"type": "object", "properties": {"path": {"type": "string"}}}

            async def execute(self, **kwargs: object) -> object:
                captured.update(kwargs)
                return "ok"

        reg = ToolRegistry()
        reg.register(CaptureTool())

        state = StateManager(initial_state={"user_file": "/data/input.txt"})
        step_def = StepDefinition(
            id="call_tool",
            type=StepType.TOOL,
            tool_name="capture",
            tool_args={
                "path": "{state.user_file}",
                "direct": "state.user_file",
                "passthrough": "literal",
            },
        )
        workflow = WorkflowDefinition(
            name="t",
            config=WorkflowConfig(provider="mock", model="m"),
            state={},
            steps=[step_def],
        )
        ctx = StepContext(
            step_definition=step_def,
            state_manager=state,
            provider_gateway=None,
            tool_registry=reg,
            run_id="r",
            workflow_name=workflow.name,
            workflow_model="m",
            workflow_provider="mock",
            sandbox_config=workflow.config.sandbox,
            observer=None,
            stream=False,
            on_stream_chunk=None,
        )
        await ToolStep().execute(ctx)
        # Template rendered, state.-prefix resolved, literal passed through.
        assert captured == {
            "path": "/data/input.txt",
            "direct": "/data/input.txt",
            "passthrough": "literal",
        }


class TestPlaceholderTriggerNarrowed:
    """Regression: ``_resolve_args`` only expands real placeholders.

    The previous heuristic — any ``{`` in the value — fired ``format_map``
    on raw JSON / HTML / code snippets that workflow authors routinely
    pass through ``tool_args.content`` and ``tool_args.body``. Failures
    surfaced as ``Max string recursion exceeded`` (nested braces) or
    ``Invalid format specifier`` (colon in JSON). The regex now requires
    a real ``{state.…}`` / ``{state[…]}`` / ``{<name>[}:!]`` shape so
    raw payloads pass through unchanged.
    """

    def test_literal_json_content_passes_through_unchanged(self) -> None:
        # F9 reproducer: ``content: |`` block carrying inline JSON.
        json_content = '{\n  "k": [1, 2, 3],\n  "nested": {"v": true}\n}'
        result = ToolStep._resolve_args({"content": json_content}, {})
        assert result == {"content": json_content}

    def test_literal_json_body_passes_through_unchanged(self) -> None:
        # F57 reproducer: ``body:`` carrying inline JSON with a colon.
        body = '{"k": [1,2,3], "nested": {"v": true}}'
        result = ToolStep._resolve_args({"body": body}, {})
        assert result == {"body": body}

    def test_html_with_braces_passes_through(self) -> None:
        html = "<style>.x { color: red; }</style>"
        result = ToolStep._resolve_args({"content": html}, {})
        assert result == {"content": html}

    def test_lone_braces_pass_through(self) -> None:
        # Mismatched / standalone braces never match the placeholder shape.
        result = ToolStep._resolve_args({"raw": "a } b { c"}, {})
        assert result == {"raw": "a } b { c"}

    def test_state_dot_placeholder_still_renders(self) -> None:
        result = ToolStep._resolve_args(
            {"greet": "hello {state.name}"},
            {"name": "Alice"},
        )
        assert result == {"greet": "hello Alice"}

    def test_state_subscript_placeholder_still_renders(self) -> None:
        # ``{state[items][0]}`` — the regex matches the ``{state[`` shape
        # so subscript-form placeholders still trigger expansion.
        result = ToolStep._resolve_args(
            {"first": "got {state[items][0]}"},
            {"items": ["A", "B"]},
        )
        assert result == {"first": "got A"}

    def test_bare_placeholder_still_renders(self) -> None:
        result = ToolStep._resolve_args({"greet": "hello {name}"}, {"name": "Bob"})
        assert result == {"greet": "hello Bob"}

    def test_format_spec_placeholder_still_renders(self) -> None:
        result = ToolStep._resolve_args(
            {"price": "cost: {total:.2f}"},
            {"total": 1234.5678},
        )
        assert result == {"price": "cost: 1234.57"}

    def test_conversion_flag_placeholder_still_renders(self) -> None:
        result = ToolStep._resolve_args({"r": "value={n!r}"}, {"n": "x"})
        assert result == {"r": "value='x'"}

    def test_template_false_escape_hatch_disables_expansion(self) -> None:
        # Even a string that looks like a placeholder is returned
        # untouched when the caller flags it ``template: false``.
        result = ToolStep._resolve_args(
            {"content": {"value": "{state.foo}", "template": False}},
            {"foo": "expanded"},
        )
        assert result == {"content": "{state.foo}"}

    def test_template_false_with_non_string_value_passes_through(self) -> None:
        # The escape hatch carries any payload, not just strings.
        result = ToolStep._resolve_args(
            {"data": {"value": [1, 2, 3], "template": False}},
            {},
        )
        assert result == {"data": [1, 2, 3]}

    def test_dict_without_template_false_passes_through_unchanged(self) -> None:
        # A plain dict argument (no ``template: false`` marker) is not
        # the escape hatch — it falls through to the existing
        # "non-string value" path and reaches the tool verbatim.
        payload = {"value": "x", "extra": 1}
        result = ToolStep._resolve_args({"body": payload}, {})
        assert result == {"body": payload}

    def test_js_object_literal_with_space_after_colon_passes_through(self) -> None:
        # ``{foo: true}`` shape (JS object literal): identifier-then-``:``
        # is normally a placeholder shape, but the negative lookahead in
        # the regex refuses ``:`` followed by whitespace so this stays
        # literal instead of triggering format_map (which would raise).
        result = ToolStep._resolve_args({"body": "{foo: true, bar: false}"}, {})
        assert result == {"body": "{foo: true, bar: false}"}

    def test_css_rule_with_spaces_passes_through(self) -> None:
        # ``.x { color: red; }`` shape: the inner ``:`` has whitespace
        # after it, so the regex does not match.
        css = ".x { color: red; padding: 0; }"
        result = ToolStep._resolve_args({"style": css}, {})
        assert result == {"style": css}
