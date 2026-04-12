"""Abstract checkpointer protocol and checkpoint data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class CheckpointData(BaseModel):
    """Serializable snapshot of a workflow execution.

    Contains everything needed to resume a paused or failed workflow:
    the original definition, current state, completed step results,
    and metadata about the execution status.
    """

    workflow_name: str
    run_id: str
    workflow_definition: dict[str, Any] = Field(
        description="Serialized WorkflowDefinition for reconstruction on resume.",
    )
    state: dict[str, Any] = Field(default_factory=dict)
    step_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Serialized StepResult dicts, keyed by step ID.",
    )
    completed_steps: list[str] = Field(default_factory=list)
    status: str = "running"
    paused_step_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


class BaseCheckpointer(ABC):
    """Abstract interface for checkpoint persistence backends.

    Implementations must use non-blocking I/O (e.g. ``anyio.to_thread``)
    so the engine stays responsive while checkpoints are saved.
    """

    @abstractmethod
    async def save(self, data: CheckpointData) -> None:
        """Persist a checkpoint snapshot."""
        ...

    @abstractmethod
    async def load(self, run_id: str) -> CheckpointData:
        """Load a checkpoint by run ID.

        Raises:
            KeyError: If no checkpoint exists for *run_id*.
        """
        ...

    @abstractmethod
    async def list_runs(self) -> list[CheckpointData]:
        """Return metadata for every stored checkpoint."""
        ...

    @abstractmethod
    async def delete(self, run_id: str) -> None:
        """Remove a checkpoint by run ID.

        Raises:
            KeyError: If no checkpoint exists for *run_id*.
        """
        ...
