"""Step registry — maps step types to executor classes."""

from __future__ import annotations

from agentloom.core.models import StepType
from agentloom.steps.base import BaseStep


class StepRegistry:
    """Registry that maps StepType values to executor classes."""

    def __init__(self) -> None:
        self._registry: dict[StepType, type[BaseStep]] = {}

    def register(self, step_type: StepType, executor_cls: type[BaseStep]) -> None:
        self._registry[step_type] = executor_cls

    def get(self, step_type: StepType) -> type[BaseStep]:
        if step_type not in self._registry:
            available = ", ".join(t.value for t in self._registry)
            raise KeyError(f"No executor for step type \'{step_type.value}\'. Available: {available}")
        return self._registry[step_type]


def create_default_registry() -> StepRegistry:
    """Create a registry with all built-in step types."""
    from agentloom.steps.llm_call import LLMCallStep

    registry = StepRegistry()
    registry.register(StepType.LLM_CALL, LLMCallStep)
    return registry
