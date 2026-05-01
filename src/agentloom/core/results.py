"""Result models for step and workflow execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(StrEnum):
    """Execution status of a step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    PAUSED = "paused"


class TokenUsage(BaseModel):
    """Token usage for an LLM call.

    ``reasoning_tokens`` covers the ``completion_tokens_details.reasoning_tokens``
    field on OpenAI o-series responses and Anthropic's extended-thinking
    ``thinking_tokens``. Providers bill these at the output rate, so cost
    calculation must include them even though the user never sees the raw
    chain-of-thought.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def billable_completion_tokens(self) -> int:
        """Tokens billed at the provider's output rate.

        Always equal to ``completion_tokens + reasoning_tokens`` — exposed
        as a property so downstream cost code does not have to remember
        which field to add.
        """
        return self.completion_tokens + self.reasoning_tokens


class StepResult(BaseModel):
    """Result from executing a single step."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    model: str | None = None
    provider: str | None = None
    attachment_count: int = 0
    time_to_first_token_ms: float | None = None


class WorkflowStatus(StrEnum):
    """Execution status of a workflow."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    PAUSED = "paused"


class WorkflowResult(BaseModel):
    """Result from executing a complete workflow."""

    workflow_name: str
    status: WorkflowStatus
    step_results: dict[str, StepResult] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None
