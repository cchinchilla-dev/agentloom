"""Tests for budget enforcer."""

from __future__ import annotations

import contextlib

import anyio
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
        be = BudgetEnforcer(limit_usd=1.0)
        with contextlib.suppress(BudgetExceededError):
            be.record(2.0)
        assert be.remaining == 0.0

    def test_has_limit_true(self) -> None:
        assert BudgetEnforcer(limit_usd=1.0).has_limit() is True

    def test_has_limit_false_when_none(self) -> None:
        assert BudgetEnforcer(limit_usd=None).has_limit() is False


class TestAsyncBudgetPrimitives:
    """The engine-wired ``estimate`` and ``charge``, whose lock closes the parallel-layer race."""

    async def test_estimate_within_budget(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        await be.charge(0.3)
        assert await be.estimate(0.5) is True

    async def test_estimate_exceeds_budget(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        await be.charge(0.8)
        assert await be.estimate(0.5) is False

    async def test_estimate_with_no_limit_always_true(self) -> None:
        be = BudgetEnforcer(limit_usd=None)
        assert await be.estimate(99999.0) is True

    async def test_charge_raises_on_overrun(self) -> None:
        be = BudgetEnforcer(limit_usd=1.0)
        await be.charge(0.6)
        with pytest.raises(BudgetExceededError):
            await be.charge(0.5)
        # The overrun is still recorded, so observability sees how far it overshot.
        assert be.spent == pytest.approx(1.1)

    async def test_charge_atomic_under_50_parallel_writers(self) -> None:
        """50 coroutines charging $0.01 against $0.10: exactly 10 succeed, the rest raise."""
        be = BudgetEnforcer(limit_usd=0.10)
        successes = 0
        failures = 0

        async def attempt() -> None:
            nonlocal successes, failures
            try:
                await be.charge(0.01)
                successes += 1
            except BudgetExceededError:
                failures += 1

        async with anyio.create_task_group() as tg:
            for _ in range(50):
                tg.start_soon(attempt)

        # Strict ``>`` lets the 10th charge land on $0.10; the other 40 still count before raising.
        assert successes == 10
        assert failures == 40
        assert be.spent == pytest.approx(0.50)
