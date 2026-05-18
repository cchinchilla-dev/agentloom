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
    """``estimate`` and ``charge`` are the engine-wired primitives. The
    ``anyio.Lock`` is what closes the parallel-layer race that the
    pre-0.5.0 bare-float counter could not detect."""

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
        # Even on overrun, the charge is recorded so observability sees
        # how far the overshoot went.
        assert be.spent == pytest.approx(1.1)

    async def test_charge_atomic_under_50_parallel_writers(self) -> None:
        """50 coroutines each charging $0.01 against a $0.10 budget must
        produce exactly 10 successful charges, with all others raising
        ``BudgetExceededError``. Pre-lock, two coroutines that both read
        ``_spent = 0.09`` and added ``0.01`` would both succeed, leaving
        the counter at ``0.11`` and the limit silently unenforced."""
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

        # Exactly 10 charges fit within $0.10 (charge raises on strict
        # ``>``, so the 10th brings spent to exactly $0.10 and does not
        # raise). The 11th onwards each still increment spent inside
        # the lock before raising — so no drops, no double-counts: the
        # final spent is precisely ``50 * 0.01 == 0.50``.
        assert successes == 10
        assert failures == 40
        assert be.spent == pytest.approx(0.50)
