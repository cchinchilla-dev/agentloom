"""A turn with both ``tools`` and a schema replies through the output tool (#120)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from agentloom.core.models import (
    Conversation,
    Message,
    ResponseSchema,
    StepDefinition,
    StepType,
    ToolDefinition,
)
from agentloom.core.results import StepResult, StepStatus
from agentloom.core.state import StateManager
from agentloom.exceptions import StepError
from agentloom.providers.base import ProviderResponse, ToolCall
from agentloom.steps._structured import OUTPUT_TOOL_REMINDER
from agentloom.steps.base import StepContext
from agentloom.steps.llm_call import LLMCallStep
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry

_ADD = ToolDefinition(
    name="add",
    description="Add two integers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)
_SUM = ResponseSchema(
    type="json_schema",
    schema={
        "type": "object",
        "properties": {"total": {"type": "integer"}},
        "required": ["total"],
    },
)


class Total(BaseModel):
    """The reply of the pydantic-mode tests, loaded by dotted path."""

    total: int


class _Add(BaseTool):
    name = "add"
    description = "Add two integers."
    parameters_schema = _ADD.parameters

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return kwargs["a"] + kwargs["b"]


class _Gateway:
    """Answers ``complete()`` from a script of turns and keeps what each call sent."""

    def __init__(self, *turns: ProviderResponse) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._turns.pop(0)


def _turn(content: str = "", *calls: ToolCall, cost: float = 0.0) -> ProviderResponse:
    return ProviderResponse(
        content=content, model="m", provider="openai", tool_calls=list(calls), cost_usd=cost
    )


def _final(n: int, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"f{n}", name="final_result", arguments=arguments)


def _add(n: int, a: int, b: int) -> ToolCall:
    return ToolCall(id=f"a{n}", name="add", arguments={"a": a, "b": b})


class _Typed(LLMCallStep):
    """Replies through the output tool, as an agent step with tools and an output_schema."""

    def __init__(self, schema: ResponseSchema = _SUM) -> None:
        self.schema = schema

    def _output_schema(self, step: StepDefinition) -> ResponseSchema | None:
        return self.schema


def _step(*, iterations: int = 4, **fields: Any) -> StepDefinition:
    return StepDefinition(
        id="s",
        type=StepType.LLM_CALL,
        prompt="Add 2 and 3.",
        tools=[_ADD],
        max_tool_iterations=iterations,
        output="reply",
        **fields,
    )


async def _run(
    step: StepDefinition,
    gateway: _Gateway,
    *,
    executor: LLMCallStep | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[StepResult, StateManager, _Add]:
    tool = _Add()
    registry = ToolRegistry()
    registry.register(tool)
    manager = StateManager(state)
    context = StepContext(
        step_definition=step,
        state_manager=manager,
        provider_gateway=gateway,
        tool_registry=registry,
    )
    result = await (executor or _Typed()).execute(context)
    return result, manager, tool


def _tool_results(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in call["messages"] if m["role"] == "tool"]


class TestOutputTool:
    async def test_reply_comes_through_the_output_tool(self) -> None:
        gateway = _Gateway(_turn("", _final(1, total=5), cost=0.25))
        result, state, tool = await _run(_step(), gateway)

        assert result.status == StepStatus.SUCCESS
        assert result.output == '{"total": 5}'
        assert result.cost_usd == 0.25
        assert await state.get("reply") == {"total": 5}
        assert tool.calls == []
        (call,) = gateway.calls
        names = [t.name for t in call["agentloom_tools"]]
        assert names == ["add", "final_result"]
        assert call["agentloom_tools"][1].parameters["required"] == ["total"]
        assert call["agentloom_tool_choice"] == "required"
        assert "agentloom_response_schema" not in call

    async def test_tools_run_before_the_reply(self) -> None:
        gateway = _Gateway(
            _turn("", _add(1, 2, 3), cost=0.1),
            _turn("", _final(2, total=5), cost=0.2),
        )
        result, state, tool = await _run(_step(), gateway)

        assert result.status == StepStatus.SUCCESS
        assert tool.calls == [{"a": 2, "b": 3}]
        assert result.cost_usd == pytest.approx(0.3)
        assert await state.get("reply") == {"total": 5}
        assert _tool_results(gateway.calls[1]) == [
            {"role": "tool", "tool_call_id": "a1", "content": "5"}
        ]

    async def test_invalid_arguments_go_back_as_the_tool_error(self) -> None:
        gateway = _Gateway(
            _turn("", _final(1, total="five")),
            _turn("", _final(2, total=5)),
        )
        result, state, _ = await _run(_step(), gateway)

        assert result.status == StepStatus.SUCCESS
        assert await state.get("reply") == {"total": 5}
        (sent,) = _tool_results(gateway.calls[1])
        assert sent["tool_call_id"] == "f1"
        assert sent["content"].startswith("The arguments of final_result failed validation.")
        assert "'five' is not of type 'integer'" in sent["content"]

    async def test_other_calls_run_next_to_an_invalid_reply(self) -> None:
        gateway = _Gateway(
            _turn("", _final(1), _add(1, 1, 1)),
            _turn("", _final(2, total=2)),
        )
        result, _, tool = await _run(_step(), gateway)

        assert result.status == StepStatus.SUCCESS
        assert tool.calls == [{"a": 1, "b": 1}]
        sent = _tool_results(gateway.calls[1])
        assert [m["tool_call_id"] for m in sent] == ["f1", "a1"]
        assert "'total' is a required property" in sent[0]["content"]
        assert sent[1]["content"] == "2"

    async def test_a_valid_reply_ends_the_turn_without_running_other_calls(self) -> None:
        gateway = _Gateway(_turn("", _add(1, 2, 3), _final(1, total=5)))
        result, _, tool = await _run(_step(), gateway)

        assert result.status == StepStatus.SUCCESS
        assert tool.calls == []
        assert len(gateway.calls) == 1

    def test_the_reminder_names_the_tool(self) -> None:
        assert OUTPUT_TOOL_REMINDER == (
            "Reply by calling the final_result tool, with your reply as its arguments."
        )

    async def test_a_text_reply_is_sent_back_for_the_tool(self) -> None:
        gateway = _Gateway(_turn("It is 5."), _turn("", _final(2, total=5)))
        result, state, _ = await _run(_step(), gateway)

        assert result.status == StepStatus.SUCCESS
        assert await state.get("reply") == {"total": 5}
        assert gateway.calls[1]["messages"][-2:] == [
            {"role": "assistant", "content": "It is 5."},
            {"role": "user", "content": OUTPUT_TOOL_REMINDER},
        ]

    async def test_no_valid_reply_within_the_cap_fails_for_good(self) -> None:
        gateway = _Gateway(_turn("It is 5."), _turn("", _final(2, total="x")))
        result, state, _ = await _run(_step(iterations=2), gateway)

        assert result.status == StepStatus.FAILED
        assert result.error == "Step 's': No valid 'final_result' call within 2 model calls"
        assert result.error_classification == "permanent"
        assert await state.get("reply") is None

    async def test_pydantic_reply_is_the_model_instance(self) -> None:
        schema = ResponseSchema(type="pydantic", model=f"{__name__}.Total")
        gateway = _Gateway(_turn("", _final(1, total=5)))
        result, state, _ = await _run(_step(), gateway, executor=_Typed(schema))

        assert result.status == StepStatus.SUCCESS
        assert await state.get("reply") == Total(total=5)
        parameters = gateway.calls[0]["agentloom_tools"][1].parameters
        assert parameters["properties"]["total"]["type"] == "integer"
        assert parameters["additionalProperties"] is False

    async def test_json_object_takes_any_object(self) -> None:
        gateway = _Gateway(_turn("", _final(1, anything=[1, 2])))
        result, state, _ = await _run(
            _step(), gateway, executor=_Typed(ResponseSchema(type="json_object"))
        )

        assert result.status == StepStatus.SUCCESS
        assert await state.get("reply") == {"anything": [1, 2]}
        assert gateway.calls[0]["agentloom_tools"][1].parameters == {"type": "object"}

    @pytest.mark.parametrize(
        "schema",
        [
            ResponseSchema(type="json_schema", schema={"type": "array"}),
            ResponseSchema(type="pydantic", model="tests.steps.nowhere.Model"),
        ],
    )
    async def test_a_schema_the_tool_cannot_carry_fails_for_good(
        self, schema: ResponseSchema
    ) -> None:
        gateway = _Gateway()
        result, _, _ = await _run(_step(), gateway, executor=_Typed(schema))

        assert result.status == StepStatus.FAILED
        assert result.error_classification == "permanent"
        assert gateway.calls == []

    async def test_a_pydantic_model_that_fails_to_load_at_validation_fails_for_good(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentloom.core.models import ResponseSchemaConfigError

        def _broken(*_a: Any) -> Any:
            raise ResponseSchemaConfigError("model vanished")

        monkeypatch.setattr("agentloom.steps._structured.validate_parsed", _broken)
        result, _, _ = await _run(_step(), _Gateway(_turn("", _final(1, total=5))))

        assert result.status == StepStatus.FAILED
        assert result.error == "Step 's': model vanished"
        assert result.error_classification == "permanent"

    @pytest.mark.parametrize("choice", ["required", "none"])
    async def test_a_tool_choice_other_than_auto_is_kept(self, choice: str) -> None:
        gateway = _Gateway(_turn("", _final(1, total=5)))
        await _run(_step(tool_choice=choice), gateway)

        assert gateway.calls[0]["agentloom_tool_choice"] == choice

    async def test_conversation_keeps_the_reply_as_json(self) -> None:
        gateway = _Gateway(_turn("", _final(1, total=5)))
        result, state, _ = await _run(_step(conversation="chat", speaker="calc"), gateway)

        assert result.status == StepStatus.SUCCESS
        chat = Conversation.model_validate(await state.get("chat"))
        assert chat.messages[-1] == Message(role="assistant", content='{"total": 5}', name="calc")

    async def test_a_registry_is_needed_only_for_calls_it_runs(self) -> None:
        gateway = _Gateway(_turn("", _final(1)), _turn("", _final(2, total=5)))
        context = StepContext(
            step_definition=_step(), state_manager=StateManager(), provider_gateway=gateway
        )
        result = await _Typed().execute(context)

        assert result.status == StepStatus.SUCCESS

    async def test_without_an_output_schema_final_result_is_a_tool_like_any(self) -> None:
        class _FinalResult(BaseTool):
            name = "final_result"

            async def execute(self, **kwargs: Any) -> Any:
                return "ran"

        registry = ToolRegistry()
        registry.register(_FinalResult())
        gateway = _Gateway(_turn("", _final(1, total=5)), _turn("done"))
        step = StepDefinition(
            id="s",
            type=StepType.LLM_CALL,
            prompt="hi",
            tools=[ToolDefinition(name="final_result")],
            output="reply",
        )
        context = StepContext(
            step_definition=step,
            state_manager=StateManager(),
            provider_gateway=gateway,
            tool_registry=registry,
        )
        result = await LLMCallStep().execute(context)

        assert result.output == "done"
        assert _tool_results(gateway.calls[1])[0]["content"] == "ran"
        assert [t.name for t in gateway.calls[0]["agentloom_tools"]] == ["final_result"]


class _Hooked(LLMCallStep):
    """Sets a turn budget of one message and refuses replies that mention ``nope``."""

    def _apply_memory(self, conversation: Conversation) -> None:
        conversation.token_budget, conversation.trim_policy = 20, "drop_oldest"

    def _accept(self, step: StepDefinition, response: ProviderResponse) -> None:
        if "nope" in response.content:
            raise StepError(step.id, "refused", is_retryable=False)


class TestHooks:
    _CHAT = {
        "messages": [
            {"role": "user", "content": "old question " * 40},
            {"role": "assistant", "content": "old answer " * 40},
        ],
        "token_budget": 5000,
        "trim_policy": "drop_pairs",
    }

    async def test_a_turn_budget_trims_without_being_stored(self) -> None:
        step = StepDefinition(id="s", type=StepType.LLM_CALL, prompt="hi", conversation="chat")
        gateway = _Gateway(_turn("hello"))
        result, state, _ = await _run(step, gateway, executor=_Hooked(), state={"chat": self._CHAT})

        assert result.status == StepStatus.SUCCESS
        sent = gateway.calls[0]["messages"]
        assert [m["content"] for m in sent] == ["hi"]
        chat = Conversation.model_validate(await state.get("chat"))
        assert (chat.token_budget, chat.trim_policy) == (5000, "drop_pairs")

    async def test_a_refused_reply_writes_nothing(self) -> None:
        step = StepDefinition(
            id="s", type=StepType.LLM_CALL, prompt="hi", conversation="chat", output="reply"
        )
        result, state, _ = await _run(
            step, _Gateway(_turn("nope")), executor=_Hooked(), state={"chat": self._CHAT}
        )

        assert result.status == StepStatus.FAILED
        assert result.error == "Step 's': refused"
        assert result.error_classification == "permanent"
        assert await state.get("reply") is None
        assert await state.get("chat") == self._CHAT

    async def test_plain_llm_call_keeps_its_behaviour(self) -> None:
        step = StepDefinition(
            id="s", type=StepType.LLM_CALL, prompt="hi", conversation="chat", output="reply"
        )
        result, state, _ = await _run(
            step, _Gateway(_turn("nope")), executor=LLMCallStep(), state={"chat": self._CHAT}
        )

        assert result.status == StepStatus.SUCCESS
        chat = Conversation.model_validate(await state.get("chat"))
        assert [m.content for m in chat.messages][-2:] == ["hi", "nope"]
        assert json.dumps(await state.get("reply")) == '"nope"'
