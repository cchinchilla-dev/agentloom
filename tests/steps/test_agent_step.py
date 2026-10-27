"""The ``agent`` step and ``Agent.invoke`` (#120)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from agentloom import Agent, AgentResult
from agentloom.checkpointing.base import CheckpointData
from agentloom.checkpointing.file import FileCheckpointer
from agentloom.core.engine import WorkflowEngine
from agentloom.core.inter_agent import EventLog, recording_events
from agentloom.core.models import AgentDefinition, Conversation, Message, WorkflowDefinition
from agentloom.core.results import StepStatus, WorkflowResult, WorkflowStatus
from agentloom.exceptions import AgentError, ProviderError, RecordingMismatchError
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse, ToolCall
from agentloom.providers.gateway import ProviderGateway
from agentloom.steps.agent import _OwnTape
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry


class Ticket(BaseModel):
    """A typed reply, loaded by dotted path."""

    ticket: str


class _Scripted(BaseProvider):
    """Answers each call with the next turn of a script, or raises it; keeps what each sent."""

    name = "scripted"
    accepts_step_id = True

    def __init__(self, *turns: ProviderResponse | Exception, delay: float = 0.0) -> None:
        super().__init__()
        self._turns = list(turns)
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> ProviderResponse:
        self.calls.append({**kw, "messages": [dict(m) for m in kw["messages"]]})
        if self._delay:
            await anyio.sleep(self._delay)
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    async def stream(self, **kw: Any) -> StreamResponse:
        raise AssertionError("an agent turn does not stream")

    def supports_model(self, _m: str) -> bool:
        return True


class _Add(BaseTool):
    name = "add"
    description = "Add two integers."
    parameters_schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs["a"] + kwargs["b"]


class _DeleteAccount(BaseTool):
    """Registered in the workflow, not given to the agent."""

    name = "delete_account"
    description = "Delete the account."

    def __init__(self) -> None:
        self.ran = False

    async def execute(self, **kwargs: Any) -> Any:
        self.ran = True
        return "deleted"


class _Tape:
    """A tool tape that serves the given results and notes what it is asked."""

    def __init__(self, served: list[tuple[str, bool]] | None = None) -> None:
        self._results = list(served or [])
        self.served: list[tuple[str, list[str]]] = []
        self.captured: list[tuple[str, list[str]]] = []
        self.attempts: list[tuple[str, Any]] = []

    async def begin_attempt(self, step_id: str, failed: Any) -> None:
        self.attempts.append((step_id, failed))

    def serve(self, step_id: str, calls: list[ToolCall]) -> list[tuple[str, bool] | None]:
        self.served.append((step_id, [c.name for c in calls]))
        return [self._results.pop(0) if self._results else None for _ in calls]

    async def capture(self, step_id: str, results: list[tuple[ToolCall, str, bool]]) -> None:
        self.captured.append((step_id, [c.name for c, _, _ in results]))


def _turn(content: str = "", *calls: ToolCall, cost: float = 0.0) -> ProviderResponse:
    return ProviderResponse(
        content=content, model="m", provider="openai", tool_calls=list(calls), cost_usd=cost
    )


def _call(name: str, n: int = 1, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"{name}{n}", name=name, arguments=arguments)


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _workflow(*steps: dict[str, Any], state: dict[str, Any] | None = None) -> WorkflowDefinition:
    return WorkflowDefinition.model_validate(
        {
            "name": "agents",
            "config": {"provider": "openai", "model": "m"},
            "state": state or {},
            "steps": list(steps),
        }
    )


def _agent_step(agent: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "id": "ask",
        "type": "agent",
        "agent": agent,
        "prompt": "{state.question}",
        "output": "reply",
        "retry": {"max_retries": 0},
        **fields,
    }


async def _run(
    workflow: WorkflowDefinition,
    provider: _Scripted,
    registry: ToolRegistry | None = None,
    **engine: Any,
) -> WorkflowResult:
    gateway = ProviderGateway()
    gateway.register(provider)
    return await WorkflowEngine(
        workflow=workflow, provider_gateway=gateway, tool_registry=registry, **engine
    ).run()


def _tool_results(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in call["messages"] if m["role"] == "tool"]


SUPPORT = {"name": "support", "system_prompt": "You help.", "tools": ["add"]}


class TestAgentStep:
    async def test_agent_step_in_workflow_persists_conversation(self) -> None:
        provider = _Scripted(_turn("", _call("add", a=2, b=3), cost=0.1), _turn("5", cost=0.2))
        workflow = _workflow(
            _agent_step(SUPPORT, conversation="state.chat"), state={"question": "2+3?"}
        )
        result = await _run(workflow, provider, _registry(_Add()))

        assert result.step_results["ask"].status == StepStatus.SUCCESS
        assert result.step_results["ask"].cost_usd == pytest.approx(0.3)
        assert result.final_state["reply"] == "5"
        chat = Conversation.model_validate(result.final_state["chat"])
        assert chat.messages == [
            Message(role="user", content="2+3?", name="support"),
            Message(role="assistant", content="5", name="support"),
        ]
        first = provider.calls[0]
        assert first["messages"][0] == {"role": "system", "content": "You help."}
        assert [t.name for t in first["agentloom_tools"]] == ["add"]
        assert first["step_id"] == "ask"

    async def test_agent_invoke_runs_tool_call_loop(self) -> None:
        provider = _Scripted(_turn("", _call("add", a=2, b=3), cost=0.1), _turn("5", cost=0.2))
        gateway = ProviderGateway()
        gateway.register(provider)
        agent = Agent(
            AgentDefinition(**SUPPORT), gateway=gateway, tool_registry=_registry(_Add()), model="m"
        )
        previous = Conversation(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ]
        )

        result = await agent.invoke("What is 2+3?", conversation=previous)

        assert isinstance(result, AgentResult)
        assert (result.reply, result.iterations) == ("5", 2)
        assert result.cost_usd == pytest.approx(0.3)
        assert [m.content for m in result.conversation.messages] == [
            "hi",
            "hello",
            "What is 2+3?",
            "5",
        ]
        assert [m.content for m in previous.messages] == ["hi", "hello"]
        assert provider.calls[0]["model"] == "m"

    async def test_a_tool_the_registry_lacks_fails_before_any_model_call(self) -> None:
        provider = _Scripted()
        result = await _run(
            _workflow(_agent_step(SUPPORT), state={"question": "q"}), provider, _registry()
        )

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert ask.error == "Agent 'support': Tool 'add' not found. Available: (none)"
        assert ask.error_classification == "permanent"
        assert provider.calls == []

    async def test_registry_tools_need_a_registry(self) -> None:
        result = await _run(_workflow(_agent_step(SUPPORT), state={"question": "q"}), _Scripted())

        assert result.step_results["ask"].error == (
            "Agent 'support': tools ['add'] need a tool registry"
        )

    async def test_agent_tool_registry_scoping_blocks_other_tools(self) -> None:
        delete = _DeleteAccount()
        provider = _Scripted(_turn("", _call("delete_account")), _turn("done"))
        result = await _run(
            _workflow(_agent_step(SUPPORT), state={"question": "q"}),
            provider,
            _registry(_Add(), delete),
        )

        assert result.final_state["reply"] == "done"
        assert not delete.ran
        assert [t.name for t in provider.calls[0]["agentloom_tools"]] == ["add"]
        (sent,) = _tool_results(provider.calls[1])
        assert sent["content"] == (
            "tool 'delete_account' not registered: Tool 'delete_account' not found. Available: add"
        )

    async def test_agent_max_iterations_enforced(self) -> None:
        agent = {**SUPPORT, "loop_policy": {"max_iterations": 2}}
        provider = _Scripted(
            _turn("", _call("add", 1, a=1, b=1)), _turn("", _call("add", 2, a=1, b=1))
        )
        result = await _run(
            _workflow(_agent_step(agent), state={"question": "q"}), provider, _registry(_Add())
        )

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert ask.error == (
            "Step 'ask': Agent 'support' reached max_iterations (2) without a final reply"
        )
        assert ask.error_classification == "permanent"
        assert "reply" not in result.final_state

    async def test_warn_and_return_keeps_the_last_reply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        policy = {"max_iterations": 1, "on_max_iterations": "warn_and_return"}
        provider = _Scripted(_turn("almost", _call("add", a=1, b=1)))
        with caplog.at_level(logging.WARNING, logger="agentloom.steps"):
            result = await _run(
                _workflow(_agent_step({**SUPPORT, "loop_policy": policy}), state={"question": "q"}),
                provider,
                _registry(_Add()),
            )

        assert result.step_results["ask"].status == StepStatus.SUCCESS
        assert result.final_state["reply"] == "almost"
        assert (
            "Agent 'support' reached max_iterations (1); returning its last reply." in caplog.text
        )

    async def test_agent_output_schema_validates_reply(self) -> None:
        schema = {"type": "pydantic", "model": f"{__name__}.Ticket"}
        provider = _Scripted(
            _turn("", _call("final_result", ticket="T-1")),
            _turn('{"ticket": "T-2"}'),
        )
        result = await _run(
            _workflow(
                _agent_step({**SUPPORT, "output_schema": schema}),
                _agent_step(
                    {"name": "plain", "output_schema": schema},
                    id="alone",
                    output="alone",
                    depends_on=["ask"],
                ),
                state={"question": "q"},
            ),
            provider,
            _registry(_Add()),
        )

        assert result.final_state["reply"] == Ticket(ticket="T-1")
        assert result.final_state["alone"] == Ticket(ticket="T-2")
        with_tools, alone = provider.calls
        assert [t.name for t in with_tools["agentloom_tools"]] == ["add", "final_result"]
        assert "agentloom_response_schema" not in with_tools
        assert alone["agentloom_response_schema"].model == f"{__name__}.Ticket"
        assert "agentloom_tools" not in alone

    async def test_agent_constraint_max_cost_aborts_when_exceeded(self) -> None:
        agent = {**SUPPORT, "constraints": {"max_cost_usd": 0.5}}
        provider = _Scripted(
            _turn("", _call("add", 1, a=1, b=1), cost=0.3),
            _turn("", _call("add", 2, a=1, b=1), cost=0.3),
            _turn("never sent"),
        )
        result = await _run(
            _workflow(_agent_step(agent), state={"question": "q"}), provider, _registry(_Add())
        )

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert ask.error == (
            "Step 'ask': Agent 'support' spent $0.6000, over its max_cost_usd of $0.5"
        )
        assert ask.error_classification == "permanent"
        assert len(provider.calls) == 2

    async def test_the_last_call_over_the_cap_fails_the_turn_too(self) -> None:
        agent = {"name": "pricey", "constraints": {"max_cost_usd": 0.5}}
        result = await _run(
            _workflow(_agent_step(agent), state={"question": "q"}), _Scripted(_turn("ok", cost=0.6))
        )

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert ask.error == (
            "Step 'ask': Agent 'pricey' spent $0.6000, over its max_cost_usd of $0.5"
        )
        assert "reply" not in result.final_state

    async def test_agent_forbidden_patterns_marks_step_failed(self) -> None:
        agent = {"name": "polite", "constraints": {"forbidden_patterns": ["as a language model"]}}
        provider = _Scripted(_turn("As A Language Model, I cannot."))
        step = _agent_step(agent, conversation="chat", retry={"max_retries": 2, "backoff_base": 0})
        result = await _run(_workflow(step, state={"question": "q"}), provider)

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert ask.error == (
            "Step 'ask': Agent 'polite' replied with the forbidden pattern 'as a language model'"
        )
        assert len(provider.calls) == 1
        assert "reply" not in result.final_state and "chat" not in result.final_state

    async def test_max_latency_fails_the_turn_like_a_timeout(self) -> None:
        agent = {"name": "slow", "constraints": {"max_latency_ms": 20}}
        provider = _Scripted(_turn("late"), _turn("late again"), delay=5)
        step = _agent_step(agent, retry={"max_retries": 1, "backoff_base": 0, "jitter": False})
        result = await _run(_workflow(step, state={"question": "q"}), provider)

        ask = result.step_results["ask"]
        assert result.status == WorkflowStatus.FAILED
        assert ask.status == StepStatus.FAILED
        assert ask.error == "Agent 'slow' exceeded max_latency_ms (20)"
        assert ask.error_classification == "transient"
        assert len(provider.calls) == 2

    async def test_memory_trims_the_turn_and_leaves_the_conversation_settings(self) -> None:
        chat = {
            "messages": [
                {"role": "user", "content": "old question " * 40},
                {"role": "assistant", "content": "old answer " * 40},
            ],
            "token_budget": 5000,
        }
        agent = {"name": "brief", "memory": {"token_budget": 20, "trim_policy": "drop_pairs"}}
        provider = _Scripted(_turn("ok"))
        result = await _run(
            _workflow(
                _agent_step(agent, conversation="chat"), state={"question": "q", "chat": chat}
            ),
            provider,
        )

        assert [m["content"] for m in provider.calls[0]["messages"]] == ["q"]
        stored = Conversation.model_validate(result.final_state["chat"])
        assert (stored.token_budget, stored.trim_policy) == (5000, "drop_oldest")

    async def test_two_agents_share_a_conversation(self) -> None:
        alice = {"name": "alice", "system_prompt": "You are Alice."}
        bob = {"name": "bob", "system_prompt": "You are Bob."}
        provider = _Scripted(_turn("plan"), _turn("critique"))
        result = await _run(
            _workflow(
                _agent_step(alice, conversation="chat", output="a"),
                _agent_step(
                    bob,
                    id="bob",
                    conversation="chat",
                    output="b",
                    prompt="Critique.",
                    depends_on=["ask"],
                ),
                state={"question": "Plan."},
            ),
            provider,
        )

        chat = Conversation.model_validate(result.final_state["chat"])
        assert [(m.name, m.content) for m in chat.messages] == [
            ("alice", "Plan."),
            ("alice", "plan"),
            ("bob", "Critique."),
            ("bob", "critique"),
        ]
        assert provider.calls[1]["messages"][0] == {"role": "system", "content": "You are Bob."}
        assert not any(m["content"] == "You are Alice." for m in provider.calls[1]["messages"])

    async def test_the_reply_reaches_the_stream_callback_once_checked(self) -> None:
        chunks: list[tuple[str, str]] = []
        result = await _run(
            _workflow(_agent_step({"name": "talker"}, stream=True), state={"question": "q"}),
            _Scripted(_turn("whole reply")),
            on_stream_chunk=lambda step, chunk: chunks.append((step, chunk)),
        )

        assert result.final_state["reply"] == "whole reply"
        assert chunks == [("ask", "whole reply")]

    async def test_a_broken_stream_callback_does_not_fail_the_turn(self) -> None:
        def _broken(step: str, chunk: str) -> None:
            raise RuntimeError("callback broke")

        result = await _run(
            _workflow(_agent_step({"name": "talker"}, stream=True), state={"question": "q"}),
            _Scripted(_turn("whole reply")),
            on_stream_chunk=_broken,
        )

        assert result.step_results["ask"].status == StepStatus.SUCCESS

    async def test_a_checkpointed_agent_step_resumes(self, tmp_path: Path) -> None:
        workflow = _workflow(
            _agent_step({"name": "first"}, id="first", output="one"),
            _agent_step({"name": "second"}, id="second", output="two", depends_on=["first"]),
            state={"question": "q"},
        )
        checkpoint = CheckpointData(
            workflow_name="agents",
            run_id="r",
            workflow_definition=workflow.model_dump(),
            state={"question": "q", "one": "1"},
            step_results={"first": {"step_id": "first", "status": "success", "output": "1"}},
            completed_steps=["first"],
            status="failed",
            created_at="2026-10-27T10:00:00+00:00",
            updated_at="2026-10-27T10:00:01+00:00",
        )
        provider = _Scripted(_turn("2"))
        gateway = ProviderGateway()
        gateway.register(provider)

        engine = await WorkflowEngine.from_checkpoint(
            checkpoint_data=checkpoint,
            checkpointer=FileCheckpointer(checkpoint_dir=tmp_path),
            provider_gateway=gateway,
        )
        result = await engine.run()

        assert result.status == WorkflowStatus.SUCCESS
        assert result.final_state["two"] == "2"
        assert [call["step_id"] for call in provider.calls] == ["second"]

    async def test_the_system_prompt_is_a_template(self) -> None:
        escaped = {"name": "a", "system_prompt": 'Reply as {{"ok": true}} for {state.question}.'}
        provider = _Scripted(_turn("done"))
        await _run(_workflow(_agent_step(escaped), state={"question": "q"}), provider)
        assert provider.calls[0]["messages"][0]["content"] == 'Reply as {"ok": true} for q.'

    async def test_a_broken_template_fails_without_a_retry(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        broken = {"name": "a", "system_prompt": 'Reply as {"ok": true}.'}
        provider = _Scripted()
        step = _agent_step(broken, retry={"max_retries": 2, "backoff_base": 0})
        with caplog.at_level(logging.WARNING):
            result = await _run(_workflow(step, state={"question": "q"}), provider)

        ask = result.step_results["ask"]
        assert "Prompt template error" in (ask.error or "")
        assert ask.error_classification == "permanent"
        assert "failed after 1 attempt(s)" in caplog.text
        assert provider.calls == []

    async def test_agent_events_carry_the_agent_name(self) -> None:
        agent = {
            "name": "support",
            "tools": ["add", {"name": "escalate", "agent": {"name": "specialist"}}],
        }
        provider = _Scripted(
            _turn("", _call("escalate", message="help")),
            _turn("specialist answer"),
            _turn("done"),
        )
        log = EventLog()
        with recording_events(log):
            await _run(
                _workflow(_agent_step(agent), state={"question": "q"}), provider, _registry(_Add())
            )

        events = log.events()
        assert {e.agent_id for e in events} == {"support"}
        assert [e.tool_name for e in events if e.type == "tool_call"] == ["escalate"]
        written = [e.key for e in events if e.type == "state_transition"]
        assert written == ["reply"]


class TestNestedAgents:
    SPECIALIST = {
        "name": "specialist",
        "description": "Knows billing.",
        "system_prompt": "You know billing.",
    }

    def _support(self, **nested: Any) -> dict[str, Any]:
        return {
            "name": "support",
            "tools": [{"name": "escalate", "agent": {**self.SPECIALIST, **nested}}],
        }

    async def test_nested_agent_as_tool(self) -> None:
        provider = _Scripted(
            _turn("", _call("escalate", message="Why is my bill high?"), cost=0.1),
            _turn("Roaming fees.", cost=0.05),
            _turn("It is roaming.", cost=0.1),
        )
        result = await _run(
            _workflow(_agent_step(self._support()), state={"question": "q"}), provider
        )

        assert result.final_state["reply"] == "It is roaming."
        assert result.step_results["ask"].cost_usd == pytest.approx(0.25)
        parent, nested, last = provider.calls
        (tool,) = parent["agentloom_tools"]
        assert (tool.name, tool.description) == ("escalate", "Knows billing.")
        assert tool.parameters["required"] == ["message"]
        assert nested["step_id"] == "ask::escalate#1"
        assert nested["messages"] == [
            {"role": "system", "content": "You know billing."},
            {"role": "user", "content": "Why is my bill high?", "name": "specialist"},
        ]
        (sent,) = _tool_results(last)
        assert sent["content"] == "Roaming fees."

    async def test_each_call_of_a_nested_agent_has_its_own_recording_key(self) -> None:
        provider = _Scripted(
            _turn("", _call("escalate", 1, message="one")),
            _turn("first"),
            _turn("", _call("escalate", 2, message="two")),
            _turn("second"),
            _turn("done"),
        )
        await _run(_workflow(_agent_step(self._support()), state={"question": "q"}), provider)

        assert [c["step_id"] for c in provider.calls] == [
            "ask",
            "ask::escalate#1",
            "ask",
            "ask::escalate#2",
            "ask",
        ]

    async def test_a_typed_nested_reply_comes_back_as_json(self) -> None:
        typed = self._support(output_schema={"type": "pydantic", "model": f"{__name__}.Ticket"})
        provider = _Scripted(
            _turn("", _call("escalate", message="open one")),
            _turn('{"ticket": "T-9"}'),
            _turn("Opened T-9."),
        )
        await _run(_workflow(_agent_step(typed), state={"question": "q"}), provider)

        (sent,) = _tool_results(provider.calls[2])
        assert sent["content"] == '{"ticket": "T-9"}'

    @pytest.mark.parametrize("arguments", [{}, {"message": ""}, {"message": 3}])
    async def test_a_call_without_a_message_is_an_error_for_the_model(
        self, arguments: dict[str, Any]
    ) -> None:
        provider = _Scripted(_turn("", _call("escalate", **arguments)), _turn("sorry"))
        result = await _run(
            _workflow(_agent_step(self._support()), state={"question": "q"}), provider
        )

        assert result.final_state["reply"] == "sorry"
        (sent,) = _tool_results(provider.calls[1])
        assert "'message' must be a non-empty string" in sent["content"]

    async def test_a_failed_nested_turn_is_an_error_for_the_model(self) -> None:
        failing = self._support(constraints={"forbidden_patterns": ["no idea"]})
        provider = _Scripted(
            _turn("", _call("escalate", message="?")),
            _turn("No idea."),
            _turn("I could not find out."),
        )
        result = await _run(_workflow(_agent_step(failing), state={"question": "q"}), provider)

        assert result.final_state["reply"] == "I could not find out."
        (sent,) = _tool_results(provider.calls[2])
        assert "forbidden pattern 'no idea'" in sent["content"]

    async def test_a_transient_nested_failure_fails_the_calling_step(self) -> None:
        provider = _Scripted(
            _turn("", _call("escalate", message="?")),
            ProviderError("openai", "unavailable", status_code=503),
        )
        result = await _run(
            _workflow(_agent_step(self._support()), state={"question": "q"}), provider
        )

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert "Agent tool 'escalate' failed: " in (ask.error or "")
        assert ask.error_classification == "transient"
        assert [call["step_id"] for call in provider.calls] == ["ask", "ask::escalate#1"]

    async def test_the_steps_retry_runs_the_turn_again_with_the_same_keys(self) -> None:
        provider = _Scripted(
            _turn("", _call("escalate", message="?")),
            ProviderError("openai", "unavailable", status_code=503),
            _turn("", _call("escalate", message="?")),
            _turn("Roaming."),
            _turn("It is roaming."),
        )
        step = _agent_step(
            self._support(), retry={"max_retries": 1, "backoff_base": 0, "jitter": False}
        )
        result = await _run(_workflow(step, state={"question": "q"}), provider)

        assert result.final_state["reply"] == "It is roaming."
        assert [call["step_id"] for call in provider.calls] == [
            "ask",
            "ask::escalate#1",
            "ask",
            "ask::escalate#1",
            "ask",
        ]

    async def test_a_replay_mismatch_in_a_nested_turn_fails_at_once(self) -> None:
        provider = _Scripted(
            _turn("", _call("escalate", message="?")), RecordingMismatchError("no turn recorded")
        )
        step = _agent_step(self._support(), retry={"max_retries": 2, "backoff_base": 0})
        result = await _run(_workflow(step, state={"question": "q"}), provider)

        assert result.status == WorkflowStatus.FAILED
        assert "no turn recorded" in (result.step_results["ask"].error or "")
        assert len(provider.calls) == 2

    async def test_an_agent_tool_is_off_the_tape_and_its_own_tools_are_on_it(self) -> None:
        tape = _Tape(served=[("4", True)])
        registry = _registry(_Add())
        registry.tape = tape
        support = {"name": "support", "tools": ["add", {"name": "escalate", "agent": SUPPORT}]}
        provider = _Scripted(
            _turn("", _call("add", a=2, b=2), _call("escalate", message="2+3?")),
            _turn("", _call("add", 2, a=2, b=3)),
            _turn("5"),
            _turn("", _call("escalate", 2, message="again")),
            _turn("6"),
            _turn("done"),
        )
        result = await _run(
            _workflow(_agent_step(support), state={"question": "q"}), provider, registry
        )

        assert result.final_state["reply"] == "done"
        assert tape.served == [("ask", ["add"]), ("ask::escalate#1", ["add"])]
        assert tape.captured == [("ask::escalate#1", ["add"]), ("ask", ["add"])]
        sent_add, sent_escalate = _tool_results(provider.calls[3])
        assert sent_add["content"] == "4"  # served from the tape
        assert sent_escalate["content"] == "5"  # the nested turn ran

    async def test_the_filtered_tape_passes_attempts_through(self) -> None:
        tape = _Tape()
        await _OwnTape(tape, {"escalate"}).begin_attempt("ask", (1,))
        assert tape.attempts == [("ask", (1,))]

    async def test_nested_spend_counts_toward_the_callers_cap(self) -> None:
        support = {**self._support(), "constraints": {"max_cost_usd": 0.5}}
        provider = _Scripted(
            _turn("", _call("escalate", message="?"), cost=0.1),
            _turn("pricey", cost=0.6),
            _turn("never sent"),
        )
        result = await _run(_workflow(_agent_step(support), state={"question": "q"}), provider)

        ask = result.step_results["ask"]
        assert ask.status == StepStatus.FAILED
        assert ask.error == (
            "Step 'ask': Agent 'support' spent $0.7000, over its max_cost_usd of $0.5"
        )
        assert len(provider.calls) == 2

    async def test_a_nested_agent_uses_the_workflow_registry_for_its_own_tools(self) -> None:
        support = self._support(tools=["add"])
        provider = _Scripted(
            _turn("", _call("escalate", message="2+2?")),
            _turn("", _call("add", a=2, b=2)),
            _turn("4"),
            _turn("It is 4."),
        )
        result = await _run(
            _workflow(_agent_step(support), state={"question": "q"}), provider, _registry(_Add())
        )

        assert result.final_state["reply"] == "It is 4."
        assert [t.name for t in provider.calls[0]["agentloom_tools"]] == ["escalate"]
        assert [t.name for t in provider.calls[1]["agentloom_tools"]] == ["add"]


class TestAgentApi:
    async def test_iterations_count_the_agents_own_calls(self) -> None:
        provider = _Scripted(
            _turn("", _call("escalate", message="?"), cost=0.1),
            _turn("nested answer", cost=0.2),
            _turn("done", cost=0.1),
        )
        gateway = ProviderGateway()
        gateway.register(provider)
        definition = AgentDefinition.model_validate({**TestNestedAgents()._support(), "model": "m"})

        result = await Agent(definition, gateway=gateway).invoke("hi")

        assert result.iterations == 2
        assert result.cost_usd == pytest.approx(0.4)

    def test_from_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.yaml").write_text("name: a\nmodel: m\n")
        agent = Agent.from_yaml("a.yaml", gateway=ProviderGateway())
        assert agent.definition == AgentDefinition(name="a", model="m")

    def test_an_agent_needs_a_model(self) -> None:
        with pytest.raises(ValueError, match="Agent 'a' names no model; pass model="):
            Agent(AgentDefinition(name="a"), gateway=ProviderGateway())

    async def test_a_failed_turn_raises(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted(_turn("As a language model...")))
        definition = AgentDefinition.model_validate(
            {"name": "a", "model": "m", "constraints": {"forbidden_patterns": ["language model"]}}
        )

        with pytest.raises(AgentError, match="forbidden pattern 'language model'"):
            await Agent(definition, gateway=gateway).invoke("hi")

    async def test_a_message_with_braces_reaches_the_model_as_written(self) -> None:
        provider = _Scripted(_turn("ok"))
        gateway = ProviderGateway()
        gateway.register(provider)
        await Agent(AgentDefinition(name="a", model="m"), gateway=gateway).invoke(
            '{"x": 1} {state.y}'
        )

        assert provider.calls[0]["messages"][-1]["content"] == '{"x": 1} {state.y}'

    async def test_invoke_logs_no_events(self) -> None:
        gateway = ProviderGateway()
        gateway.register(_Scripted(_turn("ok")))
        log = EventLog()
        with recording_events(log):
            await Agent(AgentDefinition(name="a", model="m"), gateway=gateway).invoke("hi")

        assert log.events() == []
