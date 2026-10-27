"""Agents: loading their definitions from YAML, and running one from Python (#120)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from agentloom.core.models import AgentDefinition, Conversation
from agentloom.core.results import StepStatus
from agentloom.exceptions import AgentError

if TYPE_CHECKING:
    from agentloom.core.protocols import GatewayProtocol, ToolRegistryProtocol


class AgentResult(BaseModel):
    """What a turn of an agent produced."""

    # Typed per the agent's output_schema (a Pydantic instance or a dict); the text without one.
    reply: object
    conversation: Conversation
    cost_usd: float
    iterations: int


class Agent:
    """Runs an agent from Python, outside a workflow::

        agent = Agent.from_yaml("agents/support.yaml", gateway=gateway, tool_registry=registry)
        result = await agent.invoke("My bill seems wrong")

    ``model`` applies when the definition names none.
    """

    def __init__(
        self,
        definition: AgentDefinition,
        *,
        gateway: GatewayProtocol,
        tool_registry: ToolRegistryProtocol | None = None,
        model: str | None = None,
    ) -> None:
        if definition.model is None and model is None:
            raise ValueError(f"Agent {definition.name!r} names no model; pass model=")
        self.definition = definition
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._model = model or definition.model or ""

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        gateway: GatewayProtocol,
        tool_registry: ToolRegistryProtocol | None = None,
        model: str | None = None,
    ) -> Agent:
        """The agent defined in the YAML file at *path* (see :func:`load_agent`)."""
        return cls(load_agent(path), gateway=gateway, tool_registry=tool_registry, model=model)

    async def invoke(self, prompt: str, *, conversation: Conversation | None = None) -> AgentResult:
        """Run a turn: *prompt* is the user turn, appended to *conversation* (a new one if omitted).

        Raises:
            AgentError: The turn failed; the message says why.
        """
        from agentloom.steps.agent import run_turn

        result, state, step = await run_turn(
            self.definition,
            prompt,
            step_id=self.definition.name,
            gateway=self._gateway,
            tool_registry=self._tool_registry,
            model=self._model,
            conversation=conversation,
        )
        if result.status != StepStatus.SUCCESS:
            raise AgentError(result.error or f"Agent {self.definition.name!r} failed")
        return AgentResult(
            reply=await state.get("reply"),
            conversation=Conversation.model_validate(await state.get("conversation")),
            cost_usd=result.cost_usd,
            iterations=step.iterations,
        )


def load_agent(path: str | Path) -> AgentDefinition:
    """Load the agent defined in the YAML file at *path*, and the agents it uses as tools.

    Paths are relative to the working directory, as a subworkflow's ``workflow_path``.

    Raises:
        ValueError: The file is missing or malformed, or the definitions form a cycle.
    """
    return _load(Path(path), ())


def resolve_agent(value: Any) -> Any:
    """Load *value* when it is a path; anything else is left to the model's validation."""
    return load_agent(value) if isinstance(value, str | Path) else value


def _load(path: Path, chain: tuple[Path, ...]) -> AgentDefinition:
    if not path.is_file():
        raise ValueError(f"agent file '{path}' not found")
    key = path.resolve()
    if key in chain:
        cycle = " -> ".join(os.path.relpath(p) for p in (*chain[chain.index(key) :], key))
        raise ValueError(f"agent definitions form a cycle: {cycle}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"agent file '{path}': invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"agent file '{path}' must hold a mapping")
    # Nested agents load here, with the chain of files that led to them, so a cycle stops.
    for tool in raw.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("agent"), str):
            tool["agent"] = _load(Path(tool["agent"]), (*chain, key))
    try:
        return AgentDefinition.model_validate(raw)
    except ValueError as e:
        raise ValueError(f"agent file '{path}': {e}") from e
