"""Base step executor and step context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from agentloom.core.models import SandboxConfig, StepDefinition
from agentloom.core.results import StepResult


class StepContext(BaseModel):
    """Context passed to each step during execution."""

    model_config = {"arbitrary_types_allowed": True}

    step_definition: StepDefinition
    state_manager: Any  # StateManager (Any to avoid circular import at runtime)
    provider_gateway: Any | None = None  # ProviderGateway
    tool_registry: Any | None = None  # ToolRegistry
    workflow_model: str = "gpt-4o-mini"
    workflow_provider: str = "openai"
    sandbox_config: SandboxConfig = SandboxConfig()


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
