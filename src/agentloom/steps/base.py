"""Base step executor and step context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentloom.checkpointing.base import BaseCheckpointer
from agentloom.core.models import SandboxConfig, StepDefinition
from agentloom.core.results import StepResult

# Protocols exported from core.protocols describe the contract each of
# these collaborators satisfies. Field types stay permissive so tests can
# pass in MagicMocks or minimal stand-ins without subclassing the full
# implementations; static-analysis users should import the Protocols and
# cast at the boundary.


class StepContext(BaseModel):
    """Context passed to each step during execution.

    Field shapes are documented informally; the formal contracts live in
    ``agentloom.core.protocols`` (``StateManagerProtocol``,
    ``GatewayProtocol``, ``ObserverProtocol``, ``ToolRegistryProtocol``,
    ``CheckpointerProtocol``, ``StreamCallbackProtocol``).
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
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    observer: Any | None = None  # ObserverProtocol
    stream: bool = False
    on_stream_chunk: Callable[[str, str], None] | None = None
    # Concrete (not ``Any``) because the eager ABC import is required for
    # Pydantic v2 model-field resolution — ``TYPE_CHECKING``-only imports
    # raised ``PydanticUserError: BaseCheckpointer is not fully defined``
    # at startup (see #111). The rest of the collaborator types use
    # ``Any`` + a comment naming the intended protocol.
    checkpointer: BaseCheckpointer | None = None


class BaseStep(ABC):
    """Abstract base class for all step executors."""

    @abstractmethod
    async def execute(self, context: StepContext) -> StepResult:
        """Execute this step and return its result.

        Args:
            context: The step execution context with state, providers, and tools.

        Returns:
            StepResult with status, output, and metadata.
        """
        ...
