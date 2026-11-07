"""Budget enforcement for workflow cost control.

The engine uses :class:`BudgetEnforcer` as follows:

* :meth:`BudgetEnforcer.has_limit` and :attr:`BudgetEnforcer.remaining` —
  the pre-dispatch gate: the engine refuses to start a step once the
  budget is spent.
* :meth:`BudgetEnforcer.charge` — records actual cost and raises
  ``BudgetExceededError`` on overrun. Wrapped in an ``anyio.Lock`` so
  concurrent steps in a parallel layer cannot interleave
  read-modify-write and race past the limit.
* A child subworkflow without its own ``budget_usd`` shares the parent's
  instance, so its charges land in the parent counter.
"""

from __future__ import annotations

import anyio

from agentloom.exceptions import BudgetExceededError


class BudgetEnforcer:
    """Atomic, cross-task budget tracker.

    Used as the single source of truth for workflow spend. The
    ``anyio.Lock`` is what makes :meth:`charge` safe under the parallel
    layer pattern — without it, two coroutines that both read
    ``_spent = 0.099`` and add ``0.002`` against a ``$0.1`` budget would
    both succeed, leaving the counter at ``0.103`` and the limit
    unenforced.
    """

    def __init__(self, limit_usd: float | None = None, spent_usd: float = 0.0) -> None:
        self.limit_usd = limit_usd
        self._spent: float = spent_usd  # a resumed run's, from its checkpoint
        self._lock = anyio.Lock()

    @property
    def spent(self) -> float:
        """Total amount spent so far."""
        return self._spent

    @property
    def remaining(self) -> float | None:
        """Remaining budget, or ``None`` if no limit set."""
        if self.limit_usd is None:
            return None
        return max(0.0, self.limit_usd - self._spent)

    def has_limit(self) -> bool:
        """True when a non-``None`` budget cap is configured."""
        return self.limit_usd is not None

    async def estimate(self, cost_usd: float = 0.0) -> bool:
        """Return ``True`` if charging *cost_usd* would still fit.

        Non-mutating pre-flight check. Acquires the lock to read ``_spent``
        consistently with any concurrent :meth:`charge` — the answer is
        a point-in-time snapshot, so a caller that races a charge after
        the check is expected to discover the overrun on its own
        ``charge``.
        """
        if self.limit_usd is None:
            return True
        async with self._lock:
            return (self._spent + cost_usd) <= self.limit_usd

    async def charge(self, cost_usd: float) -> None:
        """Record an actual cost atomically.

        Raises:
            BudgetExceededError: If ``_spent`` exceeds ``limit_usd`` after
                this charge. The exception carries the limit and the
                post-charge total so the engine's terminal-state
                classifier can surface the overrun without re-reading
                state.
        """
        async with self._lock:
            self._spent += cost_usd
            if self.limit_usd is not None and self._spent > self.limit_usd:
                raise BudgetExceededError(self.limit_usd, self._spent)

    def record(self, cost: float) -> None:
        """Synchronous, lock-free charge for callers outside an async context.

        Prefer :meth:`charge` so the lock applies. Raises
        ``BudgetExceededError`` on overrun.
        """
        self._spent += cost
        if self.limit_usd is not None and self._spent > self.limit_usd:
            raise BudgetExceededError(self.limit_usd, self._spent)

    def check(self, estimated_cost: float = 0.0) -> bool:
        """Synchronous pre-flight check.

        Mirrors :meth:`estimate` for non-async callers. Same caveat: the
        answer is a point-in-time snapshot, so a concurrent charge can
        invalidate it.
        """
        if self.limit_usd is None:
            return True
        return (self._spent + estimated_cost) <= self.limit_usd

    def reset(self) -> None:
        """Reset the spent counter (lock-free; for tests / explicit reuse)."""
        self._spent = 0.0
