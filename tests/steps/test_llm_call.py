"""Tests for LLM call step executor."""

from __future__ import annotations

from typing import Any

import pytest

from agentloom.core.models import Attachment, StepDefinition, StepType, WorkflowConfig
from agentloom.core.results import StepStatus
from agentloom.core.state import StateManager
from agentloom.core.templates import DotAccessDict, DotAccessList, SafeFormatDict
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.base import StepContext
from agentloom.steps.llm_call import LLMCallStep


# DotAccessDict
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

    def test_list_access_returns_dot_access_list(self) -> None:
        d = DotAccessDict({"items": ["a", "b"]})
        assert isinstance(d.items, DotAccessList)

    def test_list_nested_dict_access(self) -> None:
        d = DotAccessDict({"items": [{"name": "Alice"}]})
        assert d.items[0].name == "Alice"

    def test_list_out_of_bounds_returns_empty(self) -> None:
        d = DotAccessDict({"items": [1]})
        assert d.items[5] == ""

    def test_format_map_with_index(self) -> None:
        state = {"items": ["alpha", "beta"]}
        flat: dict[str, object] = dict(state)
        flat["state"] = DotAccessDict(state)
        result = "{state.items[0]}".format_map(flat)
        assert result == "alpha"


# DotAccessList
class TestDotAccessList:
    def test_int_index(self) -> None:
        dl = DotAccessList(["a", "b", "c"])
        assert dl[0] == "a"
        assert dl[2] == "c"

    def test_string_index(self) -> None:
        dl = DotAccessList(["a", "b"])
        assert dl["1"] == "b"

    def test_negative_index(self) -> None:
        dl = DotAccessList([1, 2, 3])
        assert dl[-1] == 3

    def test_out_of_bounds(self) -> None:
        dl = DotAccessList([1])
        assert dl[5] == ""

    def test_nested_dict_wrapping(self) -> None:
        dl = DotAccessList([{"key": "val"}])
        assert isinstance(dl[0], DotAccessDict)
        assert dl[0].key == "val"

    def test_nested_list_wrapping(self) -> None:
        dl = DotAccessList([[1, 2]])
        assert isinstance(dl[0], DotAccessList)
        assert dl[0][1] == 2

    def test_str_representation(self) -> None:
        dl = DotAccessList([1, 2])
        assert str(dl) == "[1, 2]"

    def test_invalid_string_index(self) -> None:
        dl = DotAccessList(["a"])
        assert dl["abc"] == ""


# SafeFormatDict
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


# LLMCallStep
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

    async def test_multimodal_attachment_resolved(self, step: LLMCallStep) -> None:
        """Attachments are resolved and passed as content blocks."""
        import base64
        import tempfile

        from agentloom.providers.multimodal import ImageBlock, TextBlock

        # Create a tiny image file
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(tiny_png)
            f.flush()
            img_path = f.name

        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = self._make_context(
            StepDefinition(
                id="analyze",
                type=StepType.LLM_CALL,
                prompt="Describe this image",
                attachments=[Attachment(type="image", source=img_path)],
                output="desc",
            ),
            gateway=gw,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS

        # Verify content blocks were sent
        user_msg = provider.calls[0]["messages"][0]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        assert isinstance(content, list)
        assert isinstance(content[0], TextBlock)
        assert isinstance(content[1], ImageBlock)
        assert content[1].media_type == "image/png"
        # Verify attachment_count is set on StepResult
        assert result.attachment_count == 1

    async def test_attachment_source_template_rendered(self, step: LLMCallStep) -> None:
        """Template variables in attachment source are resolved."""
        import base64
        import tempfile

        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(tiny_png)
            f.flush()
            img_path = f.name

        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = self._make_context(
            StepDefinition(
                id="analyze",
                type=StepType.LLM_CALL,
                prompt="Describe",
                attachments=[Attachment(type="image", source="{state.img}")],
                output="desc",
            ),
            state={"img": img_path},
            gateway=gw,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS

    async def test_text_only_backward_compat(self, step: LLMCallStep) -> None:
        """Steps without attachments produce plain string messages."""
        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = self._make_context(
            StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="Hello",
                output="out",
            ),
            gateway=gw,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        user_msg = provider.calls[0]["messages"][0]
        assert isinstance(user_msg["content"], str)

    async def test_streaming_execution(self, step: LLMCallStep) -> None:
        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = StepContext(
            step_definition=StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="Hello",
                output="out",
            ),
            state_manager=StateManager(initial_state={}),
            provider_gateway=gw,
            workflow_model="mock-model",
            stream=True,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.output == "Mock response"
        assert result.token_usage.total_tokens == 30
        assert result.time_to_first_token_ms is not None
        assert result.time_to_first_token_ms >= 0

    async def test_streaming_callback_invoked(self, step: LLMCallStep) -> None:
        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        received_chunks: list[tuple[str, str]] = []

        def _on_chunk(step_id: str, text: str) -> None:
            received_chunks.append((step_id, text))

        ctx = StepContext(
            step_definition=StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="Hello",
                output="out",
            ),
            state_manager=StateManager(initial_state={}),
            provider_gateway=gw,
            workflow_model="mock-model",
            stream=True,
            on_stream_chunk=_on_chunk,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert len(received_chunks) > 0
        assert all(sid == "s" for sid, _ in received_chunks)
        assert "".join(text for _, text in received_chunks) == "Mock response"

    async def test_nonstreaming_backward_compat(self, step: LLMCallStep) -> None:
        """stream=False preserves existing complete() behavior."""
        from tests.conftest import MockProvider

        provider = MockProvider()
        gw = ProviderGateway()
        gw.register(provider, models=["mock-model"])

        ctx = StepContext(
            step_definition=StepDefinition(
                id="s",
                type=StepType.LLM_CALL,
                prompt="Hello",
                output="out",
            ),
            state_manager=StateManager(initial_state={}),
            provider_gateway=gw,
            workflow_model="mock-model",
            stream=False,
        )
        result = await step.execute(ctx)
        assert result.status == StepStatus.SUCCESS
        assert result.time_to_first_token_ms is None


class TestThinkingConfig:
    """``StepDefinition.thinking`` activates provider-side reasoning from
    YAML/dict config. The step layer forwards the ``ThinkingConfig`` object
    to the gateway under a single ``thinking_config`` kwarg; each provider
    adapter translates it to its own request shape (Anthropic ``thinking``,
    Gemini ``thinkingConfig``, Ollama ``think``)."""

    @staticmethod
    def test_disabled_thinking_emits_no_kwargs() -> None:
        from agentloom.core.models import StepDefinition, StepType, ThinkingConfig
        from agentloom.steps.llm_call import LLMCallStep

        step = StepDefinition(
            id="s",
            type=StepType.LLM_CALL,
            prompt="hi",
            thinking=ThinkingConfig(enabled=False, budget_tokens=2048),
        )
        assert LLMCallStep._build_thinking_kwargs(step) == {}

    @staticmethod
    def test_enabled_thinking_forwards_config_object() -> None:
        from agentloom.core.models import StepDefinition, StepType, ThinkingConfig
        from agentloom.steps.llm_call import LLMCallStep

        cfg = ThinkingConfig(enabled=True, budget_tokens=2048)
        step = StepDefinition(
            id="s",
            type=StepType.LLM_CALL,
            prompt="hi",
            thinking=cfg,
        )
        kwargs = LLMCallStep._build_thinking_kwargs(step)
        assert kwargs == {"thinking_config": cfg}

    @staticmethod
    def test_enabled_without_budget_still_forwards() -> None:
        from agentloom.core.models import StepDefinition, StepType, ThinkingConfig
        from agentloom.steps.llm_call import LLMCallStep

        cfg = ThinkingConfig(enabled=True)
        step = StepDefinition(
            id="s",
            type=StepType.LLM_CALL,
            prompt="hi",
            thinking=cfg,
        )
        kwargs = LLMCallStep._build_thinking_kwargs(step)
        # The translation to ``budget_tokens`` happens per-provider, so the
        # step layer just passes the config object through.
        assert kwargs == {"thinking_config": cfg}

    @staticmethod
    def test_no_thinking_attribute_emits_empty() -> None:
        from agentloom.core.models import StepDefinition, StepType
        from agentloom.steps.llm_call import LLMCallStep

        step = StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi")
        assert LLMCallStep._build_thinking_kwargs(step) == {}

    @staticmethod
    def test_yaml_thinking_config_parses() -> None:
        # Workflow YAML must accept ``thinking`` as a nested mapping under
        # the step definition without any extra wiring. We feed a parsed
        # mapping into ``from_dict`` rather than the YAML string form so
        # the test stays clear of the path-existence quirk in
        # ``WorkflowParser.from_yaml`` (which calls ``Path.exists()`` on
        # the input and trips ENAMETOOLONG on multi-line strings).
        import yaml as yaml_loader

        from agentloom.core.parser import WorkflowParser

        yaml_text = """
name: thinking-test
config:
  provider: anthropic
  model: claude-opus-4
steps:
  - id: complex_reasoning
    type: llm_call
    prompt: "Solve {state.problem}"
    thinking:
      enabled: true
      budget_tokens: 4096
      level: high
      capture_reasoning: true
"""
        wf = WorkflowParser.from_dict(yaml_loader.safe_load(yaml_text))
        s = wf.steps[0]
        assert s.thinking is not None
        assert s.thinking.enabled is True
        assert s.thinking.budget_tokens == 4096
        assert s.thinking.level == "high"
        assert s.thinking.capture_reasoning is True
