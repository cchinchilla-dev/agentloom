"""Tests for LLM call step executor."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.core.models import StepDefinition, StepType, WorkflowConfig
from agentloom.core.results import StepStatus
from agentloom.core.state import StateManager
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.base import StepContext
from agentloom.steps.llm_call import DotAccessDict, LLMCallStep, SafeFormatDict

# -- DotAccessDict --


class TestDotAccessDict:
    def test_simple_access(self) -> None:
        d = DotAccessDict({"name": "Alice"})
        assert d.name == "Alice"

    def test_nested_access(self) -> None:
        d = DotAccessDict({"user": {"name": "Bob"}})
        assert d.user.name == "Bob"

    def test_missing_key_returns_empty_string(self) -> None:
        d = DotAccessDict({"name": "Alice"})
        assert d.missing == ""

    def test_str_representation(self) -> None:
        d = DotAccessDict({"a": 1})
        assert "a" in str(d)

    def test_format_renders_dict(self) -> None:
        d = DotAccessDict({"x": 1})
        assert f"{d}" == str({"x": 1})


# -- SafeFormatDict --


class TestSafeFormatDict:
    def test_existing_key(self) -> None:
        d = SafeFormatDict(name="Alice")
        assert "Hello {name}".format_map(d) == "Hello Alice"

    def test_missing_key_preserved(self) -> None:
        d = SafeFormatDict(name="Alice")
        assert "Hello {missing}".format_map(d) == "Hello {missing}"

    def test_mixed_keys(self) -> None:
        d = SafeFormatDict(a="1")
        result = "{a} and {b}".format_map(d)
        assert result == "1 and {b}"


# -- LLMCallStep --


class TestLLMCallStep:
    @pytest.fixture
    def step(self) -> LLMCallStep:
        return LLMCallStep()

    @pytest.fixture
    def gateway(self) -> ProviderGateway:
        gw = ProviderGateway()
        return gw

    def _make_context(
        self,
        step_def: StepDefinition,
        state: dict[str, Any] | None = None,
        gateway: ProviderGateway | None = None,
    ) -> StepContext:
        return StepContext(
            step_definition=step_def,
            state_manager=StateManager(initial_state=state or {}),
            provider_gateway=gateway,
            workflow_config=WorkflowConfig(provider="mock", model="mock-model"),
            workflow_model="mock-model",
        )

    async def test_no_gateway_raises(self, step: LLMCallStep) -> None:
        ctx = self._make_context(
            StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi"),
            gateway=None,
        )
        with pytest.raises(Exception, match="No provider gateway"):
            await step.execute(ctx)

    async def test_no_prompt_raises(self, step: LLMCallStep) -> None:
        gw = ProviderGateway()
        ctx = self._make_context(
            StepDefinition(id="s", type=StepType.LLM_CALL),
            gateway=gw,
        )
        with pytest.raises(Exception, match="requires a 'prompt'"):
            await step.execute(ctx)

    async def test_template_rendering(self, step: LLMCallStep) -> None:
        vars_ = step._build_template_vars({"question": "What?"})
        prompt = "Answer: {question}".format_map(SafeFormatDict(vars_))
        assert prompt == "Answer: What?"

    async def test_state_dot_access_in_template(self, step: LLMCallStep) -> None:
        vars_ = step._build_template_vars({"question": "What?"})
        prompt = "Answer: {state.question}".format_map(SafeFormatDict(vars_))
        assert prompt == "Answer: What?"

    async def test_successful_execution(self, step: LLMCallStep) -> None:
        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = self._make_context(
            StepDefinition(
                id="answer",
                type=StepType.LLM_CALL,
                prompt="Answer: {state.question}",
                output="answer",
            ),
            state={"question": "test"},
            gateway=gw,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.output is not None
        assert result.cost_usd > 0
        assert result.token_usage.total_tokens == 30

    async def test_system_prompt_rendered(self, step: LLMCallStep) -> None:
        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = self._make_context(
            StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="hi",
                system_prompt="You are {state.role}",
                output="out",
            ),
            state={"role": "helpful"},
            gateway=gw,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        # Verify system message was sent
        assert provider.calls[0]["messages"][0]["role"] == "system"
        assert "helpful" in provider.calls[0]["messages"][0]["content"]
