"""Tests for budget enforcer."""

from __future__ import annotations

import pytest

from agentloom.exceptions import BudgetExceededError
from agentloom.resilience.budget import BudgetEnforcer


class TestBudgetEnforcer:
    def test_no_limit(self) -> None:
        be = BudgetEnforcer(limit_usd=None)
        be.record(100.0)
        assert be.remaining is None
        assert be.spent == 100.0

    def test_within_budget(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        be.record(0.5)
        assert be.spent == 0.5
        assert be.remaining == 0.5

    def test_exceeds_budget_raises(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        be.record(0.5)
        with pytest.raises(BudgetExceededError):
            be.record(0.6)

    def test_check_within(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        be.record(0.3)
        assert be.check(estimated_cost=0.5) is True

    def test_check_exceeds(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        be.record(0.8)
        assert be.check(estimated_cost=0.5) is False

    def test_check_no_limit(self) -> None:
        be = BudgetEnforcer(limit_usd=None)
        assert be.check(estimated_cost=999.0) is True

    def test_reset(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        be.record(0.9)
        be.reset()
        assert be.spent == 0.0
        assert be.remaining == 1.0

    def test_remaining_never_negative(self) -> None:
        import contextlib

        be = BudgetEnforcer(limit_usd=1.0)
        with contextlib.suppress(BudgetExceededError):
            be.record(2.0)
        assert be.remaining == 0.0
