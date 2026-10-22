"""Run an AgentLoom workflow through the ``WorkflowHandle`` protocol."""

from __future__ import annotations

from typing import Any

from agentloom.core.engine import WorkflowEngine
from agentloom.core.models import WorkflowDefinition
from agentloom.core.protocols import GatewayProtocol, ObserverProtocol, ToolRegistryProtocol
from agentloom.core.results import WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.exceptions import WorkflowError


class WorkflowAdapter:
    """A :class:`WorkflowDefinition` as a ``contracts.stable.WorkflowHandle``.

    Each :meth:`run` builds a fresh :class:`WorkflowEngine` seeded with
    ``{**workflow.state, **inputs}``, as a ``subworkflow`` step with
    ``isolated_state`` does, and returns the final state. Any other status
    than success (failed, timed out, over budget, paused at an approval
    gate) raises :class:`WorkflowError`, so a caller never takes partial
    state for a result.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        provider_gateway: GatewayProtocol,
        tool_registry: ToolRegistryProtocol | None = None,
        observer: ObserverProtocol | None = None,
    ) -> None:
        self.workflow = workflow
        self.provider_gateway = provider_gateway
        self.tool_registry = tool_registry
        self.observer = observer

    @property
    def input_schema(self) -> dict[str, Any]:
        # Workflows declare no typed inputs yet: any object, naming the
        # ``state:`` keys without their defaults, which may be secrets.
        return {
            "type": "object",
            "properties": {key: {} for key in self.workflow.state},
            "additionalProperties": True,
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        engine = WorkflowEngine(
            workflow=self.workflow,
            state_manager=StateManager(initial_state={**self.workflow.state, **inputs}),
            provider_gateway=self.provider_gateway,
            tool_registry=self.tool_registry,
            observer=self.observer,
        )
        result = await engine.run()
        if result.status != WorkflowStatus.SUCCESS:
            detail = f": {result.error}" if result.error else ""
            raise WorkflowError(
                f"Workflow {self.workflow.name!r} ended with status {result.status.value}{detail}"
            )
        return dict(result.final_state)
