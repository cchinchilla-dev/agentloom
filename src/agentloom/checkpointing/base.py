"""Abstract checkpointer base class and checkpoint data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# The format this runtime writes; ``WorkflowEngine.from_checkpoint`` refuses a newer one.
CURRENT_CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointData(BaseModel):
    """Serializable snapshot of a workflow execution.

    Contains everything needed to resume a paused or failed workflow:
    the original definition, current state, completed step results,
    and metadata about the execution status.
    """

    workflow_name: str
    run_id: str
    # A checkpoint without the key loads as the current version; a newer one is refused on resume.
    schema_version: int = CURRENT_CHECKPOINT_SCHEMA_VERSION
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
    # When each pending approval gate expires (ISO 8601, UTC), by id: ``sub.gate`` for a child's
    gate_deadlines: dict[str, str] = Field(default_factory=dict)
    # The spend a resumed run starts from; without it, the step results' cost stands in.
    spent_usd: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    @property
    def expires_at(self) -> str | None:
        """When the gate this run is paused at expires; ``None`` without a ``timeout_seconds``."""
        if self.status != "paused" or self.paused_step_id is None:
            return None
        return self.gate_deadlines.get(self.paused_step_id)

    def gate_expired(self, now: datetime) -> bool:
        """Whether the gate this run is paused at has run out of time by *now*."""
        expires_at = self.expires_at
        return expires_at is not None and now >= datetime.fromisoformat(expires_at)


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
        """Return every stored checkpoint."""
        ...

    @abstractmethod
    async def delete(self, run_id: str) -> None:
        """Remove a checkpoint by run ID.

        Raises:
            KeyError: If no checkpoint exists for *run_id*.
        """
        ...
