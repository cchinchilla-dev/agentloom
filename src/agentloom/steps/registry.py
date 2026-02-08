"""Step registry — maps step types to executor classes."""

from __future__ import annotations

from agentloom.core.models import StepType
from agentloom.steps.base import BaseStep


class StepRegistry:
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
    from agentloom.steps.llm_call import LLMCallStep
    from agentloom.steps.router import RouterStep
    from agentloom.steps.tool_step import ToolStep

    registry = StepRegistry()
    registry.register(StepType.LLM_CALL, LLMCallStep)
    registry.register(StepType.ROUTER, RouterStep)
    registry.register(StepType.TOOL, ToolStep)
    return registry
