"""Agent step executor: one turn of an agent, run as an ``llm_call`` (#120)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import anyio
from pydantic import BaseModel

from agentloom.core.inter_agent import paused_events
from agentloom.core.models import (
    AgentDefinition,
    AgentTool,
    Conversation,
    ResponseSchema,
    RetryConfig,
    StepDefinition,
    StepType,
    ToolDefinition,
)
from agentloom.core.results import StepResult, StepStatus
from agentloom.core.state import StateManager
from agentloom.exceptions import StepError, ToolNotFoundError
from agentloom.providers.base import ProviderResponse, ToolCall
from agentloom.steps.base import StepContext
from agentloom.steps.llm_call import LLMCallStep
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agentloom.core.attempts import Attempt
    from agentloom.providers.tool_tape import ToolTape

logger = logging.getLogger("agentloom.steps")

# The finish reasons of a tool loop that reached its cap.
_CAP_REASONS = ("max_tool_iterations", "max_tool_iterations_no_answer")


class AgentStep(LLMCallStep):
    """Runs a turn of the step's agent: an ``llm_call`` with the agent's settings.

    The turn sees only the agent's tools, counts its model calls (``iterations``)
    and fails when it breaks a constraint or, under ``on_max_iterations: error``,
    reaches ``max_iterations``. It never streams: the reply reaches the stream
    callback in one chunk, once it has passed the checks.
    """

    def __init__(self) -> None:
        self._agent = AgentDefinition(name="")
        # Model calls of the last turn, the agent's own (not those of agents it called).
        self.iterations = 0

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        agent = step.agent
        assert agent is not None  # an agent step without one fails validation
        self._agent = agent
        self.iterations = 0
        start = time.monotonic()

        gateway = None
        if context.provider_gateway is not None:
            gateway = _MeteredGateway(context.provider_gateway, step.id, agent)
        try:
            registry = _scoped_registry(agent, context, gateway)
        except LookupError as e:
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Agent {agent.name!r}: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
                error_classification="permanent",
            )
        tools = [
            ToolDefinition(name=t.name, description=t.description, parameters=t.schema)
            for t in registry.list()
        ]
        turn = StepDefinition(
            id=step.id,
            type=StepType.LLM_CALL,
            model=agent.model,
            system_prompt=agent.system_prompt,
            prompt=step.prompt,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            tools=tools,
            max_tool_iterations=agent.loop_policy.max_iterations,
            # With tools, the reply comes through the output tool (``_output_schema``).
            response_schema=None if tools else agent.output_schema,
            conversation=step.conversation,
            # Re-asserts the agent's system prompt on its turns of a shared conversation.
            speaker=agent.name,
            output=step.output,
            attachments=step.attachments,
            retry=step.retry,
        )
        inner = context.model_copy(
            update={
                "step_definition": turn,
                "tool_registry": registry,
                "provider_gateway": gateway,
                "stream": False,
            }
        )
        latency = agent.constraints.max_latency_ms
        try:
            with anyio.fail_after(latency / 1000 if latency is not None else None):
                result = await super().execute(inner)
        except TimeoutError:
            # A failure the step's retry repeats, as a step timeout: a slow provider is transient.
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Agent {agent.name!r} exceeded max_latency_ms ({latency})",
                duration_ms=(time.monotonic() - start) * 1000,
                error_classification="transient",
            )
        finally:
            self.iterations = gateway.calls if gateway is not None else 0

        if result.status == StepStatus.SUCCESS:
            if gateway is not None:
                # What the turn cost, the agents it called as tools and its summaries included.
                result.cost_usd = gateway.spent
            if context.stream and context.on_stream_chunk and result.output:
                try:
                    context.on_stream_chunk(step.id, result.output)
                except Exception:
                    logger.warning("Stream chunk callback failed")
        return result

    def _output_schema(self, step: StepDefinition) -> ResponseSchema | None:
        return self._agent.output_schema

    def _apply_memory(self, conversation: Conversation) -> None:
        if (memory := self._agent.memory) is not None:
            conversation.token_budget = memory.token_budget
            conversation.trim_policy = memory.trim_policy

    def _accept(self, step: StepDefinition, response: ProviderResponse) -> None:
        agent = self._agent
        if response.finish_reason in _CAP_REASONS:
            cap = agent.loop_policy.max_iterations
            if agent.loop_policy.on_max_iterations == "error":
                raise StepError(
                    step.id,
                    f"Agent {agent.name!r} reached max_iterations ({cap}) without a final reply",
                    is_retryable=False,
                )
            logger.warning(
                "Agent %r reached max_iterations (%d); returning its last reply.", agent.name, cap
            )
        reply = (response.content or "").casefold()
        for pattern in agent.constraints.forbidden_patterns:
            if pattern.casefold() in reply:
                raise StepError(
                    step.id,
                    f"Agent {agent.name!r} replied with the forbidden pattern {pattern!r}",
                    is_retryable=False,
                )


class _MeteredGateway:
    """The gateway of an agent's turn: counts its model calls and enforces ``max_cost_usd``.

    The agents it calls as tools use it too, so what they spend counts toward the cap,
    checked after each call and before the next one.
    """

    def __init__(self, inner: Any, step_id: str, agent: AgentDefinition) -> None:
        self._inner = inner
        self._step_id = step_id
        self._agent = agent
        self.spent = 0.0
        self.calls = 0

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self._check_spend()
        response: ProviderResponse = await self._inner.complete(**kwargs)
        self.spent += response.cost_usd
        if kwargs.get("step_id") == self._step_id:
            self.calls += 1
        self._check_spend()
        return response

    def _check_spend(self) -> None:
        cap = self._agent.constraints.max_cost_usd
        if cap is not None and self.spent > cap:
            raise StepError(
                self._step_id,
                f"Agent {self._agent.name!r} spent ${self.spent:.4f}, "
                f"over its max_cost_usd of ${cap}",
                is_retryable=False,
            )


def _scoped_registry(
    agent: AgentDefinition, context: StepContext, gateway: _MeteredGateway | None
) -> ToolRegistry:
    """A registry with only the agent's tools, in its order, on the workflow's tape.

    Raises:
        LookupError: A tool the workflow's registry lacks, or no registry for them.
    """
    base = context.tool_registry
    names = [t for t in agent.tools if isinstance(t, str)]
    if names and base is None:
        raise LookupError(f"tools {names} need a tool registry")
    view = ToolRegistry()
    tape = getattr(base, "tape", None)
    agents = {entry.name for entry in agent.tools if not isinstance(entry, str)}
    view.tape = _OwnTape(tape, agents) if tape is not None and agents else tape
    for entry in agent.tools:
        if isinstance(entry, str):
            assert base is not None
            try:
                view.register(base.get(entry))
            except ToolNotFoundError as e:
                # Its message, without the quotes ``KeyError.__str__`` adds.
                raise LookupError(e.args[0]) from e
        else:
            view.register(_AgentAsTool(entry, context, gateway))
    return view


class _OwnTape:
    """The workflow's tape for the registry tools of a turn, without the turn's agent tools.

    An agent called as a tool is neither served from the tape nor taped: its turn replays from
    the recording in every tools mode, its model calls, cost and constraints included, and its
    own tools are on the tape under its own key.
    """

    def __init__(self, tape: ToolTape, agents: set[str]) -> None:
        self._tape = tape
        self._agents = agents

    async def begin_attempt(self, step_id: str, failed: Attempt | None) -> None:
        await self._tape.begin_attempt(step_id, failed)

    def serve(self, step_id: str, calls: list[ToolCall]) -> list[tuple[str, bool] | None]:
        own = [call for call in calls if call.name not in self._agents]
        served = iter(self._tape.serve(step_id, own) if own else [])
        return [None if call.name in self._agents else next(served) for call in calls]

    async def capture(self, step_id: str, results: list[tuple[ToolCall, str, bool]]) -> None:
        if own := [result for result in results if result[0].name not in self._agents]:
            await self._tape.capture(step_id, own)


class _AgentAsTool(BaseTool):
    """Another agent offered as a tool: each call runs a turn of it on a fresh conversation."""

    parameters_schema = {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "What to ask the agent."}},
        "required": ["message"],
    }

    def __init__(self, entry: AgentTool, context: StepContext, gateway: Any) -> None:
        self.name = entry.name
        self.description = entry.description or entry.agent.description
        self._agent = entry.agent
        self._context = context
        self._gateway = gateway
        self._calls = 0

    async def execute(self, **kwargs: Any) -> Any:
        message = kwargs.get("message")
        if not isinstance(message, str) or not message:
            raise ValueError("'message' must be a non-empty string")
        # One recording key per call, in the order the calls start, so parallel calls replay.
        self._calls += 1
        step_id = f"{self._context.step_definition.id}::{self.name}#{self._calls}"
        result, state, _ = await run_turn(
            self._agent,
            message,
            step_id=step_id,
            gateway=self._gateway,
            tool_registry=self._context.tool_registry,
            model=self._context.workflow_model,
        )
        if result.status != StepStatus.SUCCESS:
            if result.error_classification == "transient":
                # It fails the calling step, whose retry repeats the turn: a recording keeps
                # only the attempt that finished, so the run replays
                raise StepError(
                    self._context.step_definition.id,
                    f"Agent tool {self.name!r} failed: {result.error}",
                )
            raise RuntimeError(result.error)
        reply = await state.get("reply")
        return reply.model_dump(mode="json") if isinstance(reply, BaseModel) else reply


async def run_turn(
    agent: AgentDefinition,
    message: str,
    *,
    step_id: str,
    gateway: Any,
    tool_registry: Any = None,
    model: str = "gpt-4o-mini",
    conversation: Conversation | None = None,
) -> tuple[StepResult, StateManager, AgentStep]:
    """Run a turn of *agent* on its own state: for ``Agent.invoke`` and agents as tools.

    The state holds the message, the conversation (``conversation``) and the reply
    (``reply``); the turn logs no inter-agent events, as it is not a step of a workflow.
    ``model`` applies when the agent names none.
    """
    state = StateManager(
        {"message": message, "conversation": (conversation or Conversation()).model_dump()}
    )
    step = StepDefinition(
        id=step_id,
        type=StepType.AGENT,
        agent=agent,
        # The message reaches the model as written: a value inserted in a template is not parsed.
        prompt="{state.message}",
        conversation="conversation",
        output="reply",
        retry=RetryConfig(max_retries=0),
    )
    context = StepContext(
        step_definition=step,
        state_manager=state,
        provider_gateway=gateway,
        tool_registry=tool_registry,
        workflow_model=model,
    )
    executor = AgentStep()
    with paused_events():
        result = await executor.execute(context)
    return result, state, executor
