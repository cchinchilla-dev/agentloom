"""Base step executor and step context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentloom.checkpointing.base import BaseCheckpointer
from agentloom.core.models import SandboxConfig, StepDefinition
from agentloom.core.redact import RedactionPolicy
from agentloom.core.results import StepResult


class StepContext(BaseModel):
    """Context passed to each step during execution.

    Most collaborator fields are typed ``Any`` so tests can pass mocks or minimal stand-ins;
    their contracts are the Protocols in ``agentloom.core.protocols``
    (``StateManagerProtocol``, ``GatewayProtocol``, ``ObserverProtocol``,
    ``ToolRegistryProtocol``, ``CheckpointerProtocol``, ``StreamCallbackProtocol``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_definition: StepDefinition
    state_manager: Any  # StateManagerProtocol
    provider_gateway: Any | None = None  # GatewayProtocol
    tool_registry: Any | None = None  # ToolRegistryProtocol
    workflow_model: str = "gpt-4o-mini"
    workflow_provider: str = "openai"
    run_id: str = ""
    workflow_name: str = ""
    capture_prompts: bool = (
        False  # WorkflowConfig.capture_prompts — gate for span-event prompt capture
    )
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    redaction_policy: RedactionPolicy = Field(default_factory=RedactionPolicy)
    observer: Any | None = None  # ObserverProtocol
    stream: bool = False
    on_stream_chunk: Callable[[str, str], None] | None = None
    # Imported at runtime, not under TYPE_CHECKING: Pydantic resolves this annotation.
    checkpointer: BaseCheckpointer | None = None
    # The running engine's enforcer; a subworkflow without its own ``budget_usd`` shares it.
    budget_enforcer: Any | None = None  # BudgetEnforcer


class BaseStep(ABC):
    """Abstract base class for all step executors."""

    @abstractmethod
    async def execute(self, context: StepContext) -> StepResult:
        """Execute this step and return its result."""
        ...
