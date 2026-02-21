"""Budget enforcement for workflow cost control."""

from __future__ import annotations

from agentloom.exceptions import BudgetExceededError


class BudgetEnforcer:
    """Tracks and enforces spending limits for workflow execution."""

    def __init__(self, limit_usd: float | None = None) -> None:
        self.limit_usd = limit_usd
        self._spent: float = 0.0

    @property
    def spent(self) -> float:
        """Total amount spent so far."""
        return self._spent

    @property
    def remaining(self) -> float | None:
        """Remaining budget, or None if no limit set."""
        if self.limit_usd is None:
            return None
        return max(0.0, self.limit_usd - self._spent)

    def record(self, cost: float) -> None:
        """Record a cost and check against budget.

        Args:
            cost: Cost in USD to record.

        Raises:
            BudgetExceededError: If the budget limit is exceeded.
        """
        self._spent += cost
        if self.limit_usd is not None and self._spent > self.limit_usd:
            raise BudgetExceededError(self.limit_usd, self._spent)

    def check(self, estimated_cost: float = 0.0) -> bool:
        """Check if a given cost would exceed the budget.

        Args:
            estimated_cost: Estimated cost of the next operation.

        Returns:
            True if within budget, False if would exceed.
        """
        if self.limit_usd is None:
            return True
        return (self._spent + estimated_cost) <= self.limit_usd

    def reset(self) -> None:
        """Reset the spent counter."""
        self._spent = 0.0
