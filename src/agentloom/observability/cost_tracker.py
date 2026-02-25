"""Cost tracking and aggregation — pure Python, no external dependencies."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CostEntry(BaseModel):
    """A single cost entry."""

    step_id: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class CostSummary(BaseModel):
    """Aggregated cost summary for a workflow run."""

    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    cost_by_provider: dict[str, float] = Field(default_factory=dict)
    cost_by_step: dict[str, float] = Field(default_factory=dict)
    entries: list[CostEntry] = Field(default_factory=list)


class CostTracker:
    """Tracks and aggregates costs across a workflow execution.

    Pure Python — always available regardless of optional dependencies.
    """

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []

    def record(
        self,
        step_id: str,
        model: str,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a cost entry."""
        self._entries.append(
            CostEntry(
                step_id=step_id,
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            )
        )

    def summary(self) -> CostSummary:
        """Generate an aggregated cost summary."""
        cost_by_model: dict[str, float] = {}
        cost_by_provider: dict[str, float] = {}
        cost_by_step: dict[str, float] = {}
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0

        for entry in self._entries:
            total_cost += entry.cost_usd
            total_prompt += entry.prompt_tokens
            total_completion += entry.completion_tokens

            cost_by_model[entry.model] = cost_by_model.get(entry.model, 0.0) + entry.cost_usd
            cost_by_provider[entry.provider] = (
                cost_by_provider.get(entry.provider, 0.0) + entry.cost_usd
            )
            cost_by_step[entry.step_id] = cost_by_step.get(entry.step_id, 0.0) + entry.cost_usd

        return CostSummary(
            total_cost_usd=total_cost,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            cost_by_model=cost_by_model,
            cost_by_provider=cost_by_provider,
            cost_by_step=cost_by_step,
            entries=list(self._entries),
        )

    def reset(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()
