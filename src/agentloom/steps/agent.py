"""Agent step executor: one turn of an agent, run as an ``llm_call`` (#120)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import anyio
from pydantic import BaseModel

from agentloom.core.inter_agent import called_agent, log_reply
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
from agentloom.core.results import StepResult, StepStatus, TokenUsage
from agentloom.core.state import StateManager
from agentloom.exceptions import RecordingMismatchError, StepError, ToolNotFoundError
from agentloom.providers.base import ProviderResponse, ToolCall
from agentloom.steps._tools import FailsStep
from agentloom.steps.base import StepContext
from agentloom.steps.llm_call import LLMCallStep
from agentloom.tools.base import BaseTool
from agentloom.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agentloom.core.attempts import Attempt
    from agentloom.core.protocols import GatewayProtocol, ToolRegistryProtocol
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
        self._agent = AgentDefinition(name="agent")  # the step's, once it runs
        self._gateway: _MeteredGateway | None = None
        # Model calls of the last turn, the agent's own (not those of agents it called).
        self.iterations = 0

    async def execute(self, context: StepContext) -> StepResult:
        step = context.step_definition
        agent = step.agent
        assert agent is not None  # an agent step without one fails validation
        self._agent = agent
        self.iterations = 0
        start = time.monotonic()

        if context.provider_gateway is None:
            raise StepError(step.id, "No provider gateway configured")
        gateway = _MeteredGateway(context.provider_gateway, step.id, agent)
        self._gateway = gateway
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
            # The step's retry repeats the turn; inside it, max_iterations bounds the re-asks too.
            retry=step.retry.model_copy(
                update={"max_retries": agent.loop_policy.max_iterations - 1}
            ),
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
            result = StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Agent {agent.name!r} exceeded max_latency_ms ({latency})",
                duration_ms=(time.monotonic() - start) * 1000,
                error_classification="transient",
            )
        finally:
            self.iterations = gateway.calls

        # What the turn cost, failed or not: the agents it called as tools and its summaries
        # included, as the provider charged them.
        result.cost_usd = gateway.spent
        result.token_usage = gateway.usage
        succeeded = result.status == StepStatus.SUCCESS
        if succeeded and context.stream and context.on_stream_chunk and result.output:
            try:
                context.on_stream_chunk(step.id, result.output)
            except Exception:
                logger.warning("Stream chunk callback failed")
        return result

    def _output_schema(self, step: StepDefinition) -> ResponseSchema | None:
        return self._agent.output_schema

    def _apply_memory(self, conversation: Conversation) -> None:
        if (memory := self._agent.memory) is not None:
            # A setting the agent leaves out keeps the conversation's own
            if memory.token_budget is not None:
                conversation.token_budget = memory.token_budget
            if memory.trim_policy is not None:
                conversation.trim_policy = memory.trim_policy

    def _accept(self, step: StepDefinition, response: ProviderResponse) -> None:
        agent = self._agent
        assert self._gateway is not None  # set by execute, which calls this
        # An agent called as a tool in the last call may have broken the cap, its error going
        # back to the model as the tool's
        self._gateway.check_spend()
        if response.finish_reason in _CAP_REASONS:
            cap = agent.loop_policy.max_iterations
            # A last call with only tool calls leaves no reply for warn_and_return to return
            replied = bool((response.content or "").strip())
            if agent.loop_policy.on_max_iterations == "error" or not replied:
                raise StepError(
                    step.id,
                    f"Agent {agent.name!r} reached max_iterations ({cap}) without a final reply",
                    is_retryable=False,
                )
            logger.warning(
                "Agent %r reached max_iterations (%d); returning its last reply.", agent.name, cap
            )
        # A typed reply is read by its values, not its field names nor its JSON escapes.
        texts = _texts(response.parsed) if response.parsed is not None else [response.content]
        replies = [(text or "").casefold() for text in texts]
        for pattern in agent.constraints.forbidden_patterns:
            if any(pattern.casefold() in reply for reply in replies):
                raise StepError(
                    step.id,
                    f"Agent {agent.name!r} replied with the forbidden pattern {pattern!r}",
                    is_retryable=False,
                )


def _texts(value: Any) -> list[str | None]:
    """The strings of a typed reply: its values, at any depth."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return [text for item in value.values() for text in _texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    return [value] if isinstance(value, str) else []


class _MeteredGateway:
    """The gateway of an agent's turn: counts its model calls and enforces ``max_cost_usd``.

    The agents it calls as tools use it too, so what they spend counts toward the cap,
    checked after each call and before the next one, and so do their tokens.
    """

    def __init__(self, inner: Any, step_id: str, agent: AgentDefinition) -> None:
        self._inner = inner
        self._step_id = step_id
        self._agent = agent
        self.spent = 0.0
        self.usage = TokenUsage()
        self.calls = 0

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self.check_spend()
        response: ProviderResponse = await self._inner.complete(**kwargs)
        self.spent += response.cost_usd
        used = response.usage
        self.usage = TokenUsage(
            prompt_tokens=self.usage.prompt_tokens + used.prompt_tokens,
            completion_tokens=self.usage.completion_tokens + used.completion_tokens,
            total_tokens=self.usage.total_tokens + used.total_tokens,
            reasoning_tokens=self.usage.reasoning_tokens + used.reasoning_tokens,
        )
        if kwargs.get("step_id") == self._step_id:
            self.calls += 1
        self.check_spend()
        return response

    def check_spend(self) -> None:
        cap = self._agent.constraints.max_cost_usd
        if cap is not None and self.spent > cap:
            raise StepError(
                self._step_id,
                f"Agent {self._agent.name!r} spent ${self.spent:.4f}, "
                f"over its max_cost_usd of ${cap}",
                is_retryable=False,
            )


def _scoped_registry(
    agent: AgentDefinition, context: StepContext, gateway: _MeteredGateway
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
        number = self._calls
        step_id = f"{self._context.step_definition.id}::{self.name}#{number}"
        with called_agent(self.name, number, self._agent.name, message):
            reply = await self._turn(message, step_id)
            log_reply(reply)
        return reply

    async def _turn(self, message: str, step_id: str) -> Any:
        try:
            result, state, _ = await run_turn(
                self._agent,
                message,
                step_id=step_id,
                gateway=self._gateway,
                tool_registry=self._context.tool_registry,
                model=self._context.workflow_model,
                state=await self._context.state_manager.get_state_snapshot(),
                # A provider error the calling step would not retry is no transient failure
                retryable_status_codes=self._context.step_definition.retry.retryable_status_codes,
            )
        except RecordingMismatchError as e:
            raise FailsStep(e) from e  # a stale replay fails at once, as anywhere else
        if result.status != StepStatus.SUCCESS:
            if result.error_classification == "transient":
                # It fails the calling step, whose retry repeats the turn: a recording keeps
                # only the attempt that finished, so the run replays
                raise FailsStep(
                    StepError(
                        self._context.step_definition.id,
                        f"Agent tool {self.name!r} failed: {result.error}",
                    )
                )
            raise RuntimeError(result.error)
        reply = await state.get("reply")
        return reply.model_dump(mode="json") if isinstance(reply, BaseModel) else reply


async def run_turn(
    agent: AgentDefinition,
    message: str,
    *,
    step_id: str,
    gateway: GatewayProtocol,
    tool_registry: ToolRegistryProtocol | None = None,
    model: str = "gpt-4o-mini",
    conversation: Conversation | None = None,
    retryable_status_codes: list[int] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[StepResult, StateManager, AgentStep]:
    """Run a turn of *agent* on its own state: for ``Agent.invoke`` and agents as tools.

    The state holds the message, the conversation (``conversation``) and the reply
    (``reply``), over *state* when given (the calling workflow's, for an agent used as a
    tool, so its prompts render as the caller's do); the caller decides what the turn logs in the
    inter-agent event log. ``model`` applies when the agent names none;
    ``retryable_status_codes`` classify a provider error of the turn as transient (the
    defaults of a step otherwise).
    """
    retry = RetryConfig(max_retries=0)
    if retryable_status_codes is not None:
        retry = RetryConfig(max_retries=0, retryable_status_codes=retryable_status_codes)
    turn_state = StateManager(
        {
            **(state or {}),
            "message": message,
            "conversation": (conversation or Conversation()).model_dump(),
        }
    )
    step = StepDefinition(
        id=step_id,
        type=StepType.AGENT,
        agent=agent,
        # The message reaches the model as written: a value inserted in a template is not parsed.
        prompt="{state.message}",
        conversation="conversation",
        output="reply",
        retry=retry,
    )
    context = StepContext(
        step_definition=step,
        state_manager=turn_state,
        provider_gateway=gateway,
        tool_registry=tool_registry,
        workflow_model=model,
    )
    executor = AgentStep()
    result = await executor.execute(context)
    return result, turn_state, executor
