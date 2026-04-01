"""Circuit breaker pattern for provider failure isolation."""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import Any, TypeVar

from agentloom.exceptions import CircuitOpenError

T = TypeVar("T")


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold, rejecting calls
    HALF_OPEN = "half_open"  # Testing if provider has recovered


class CircuitBreaker:
    """Circuit breaker for isolating failing providers.

    States:
        CLOSED  -> OPEN: after `fail_threshold` consecutive failures
        OPEN    -> HALF_OPEN: after `reset_timeout` seconds
        HALF_OPEN -> CLOSED: on success
        HALF_OPEN -> OPEN: on failure
    """

    # Callback type: (name, old_state, new_state) -> None
    OnStateChange = Callable[[str, "CircuitState", "CircuitState"], None]

    def __init__(
        self,
        name: str = "",
        fail_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        on_state_change: OnStateChange | None = None,
    ) -> None:
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.on_state_change = on_state_change
        # NOTE: hardcoded to 1 test call in half-open, might be too conservative

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic OPEN -> HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._set_state(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
        return self._state

    def _set_state(self, new_state: CircuitState) -> None:
        """Transition to a new state and fire the callback if registered."""
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        if self.on_state_change:
            self.on_state_change(self.name, old_state, new_state)

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def call(self, coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Execute an async callable through the circuit breaker.

        Args:
            coro_factory: A zero-argument callable that returns a coroutine.

        Returns:
            The result of the coroutine.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(self.name)

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(self.name)
            self._half_open_calls += 1

        try:
            result = await coro_factory()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._set_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
        self._failure_count = 0
        self._success_count += 1

    def _on_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.fail_threshold:
            self._set_state(CircuitState.OPEN)

    def allow_request(self) -> None:
        """Pre-check for streaming: raises CircuitOpenError if a request should not proceed.

        Unlike ``call()``, this does not wrap a coroutine — it only validates
        that the circuit allows a new request through.  Pair with
        ``record_success()`` / ``record_failure()`` after the work completes.

        NOTE: Under concurrent streaming (parallel DAG steps), multiple
        coroutines may pass ``allow_request()`` before any calls
        ``record_success()``.  In the HALF_OPEN state this can exceed
        ``half_open_max_calls``.  This is acceptable under cooperative
        (single-threaded) concurrency but would require a lock if
        moved to a threaded executor.
        """
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(self.name)
        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(self.name)
            self._half_open_calls += 1

    def record_success(self) -> None:
        """Record a successful call (deferred feedback for streaming)."""
        self._on_success()

    def record_failure(self) -> None:
        """Record a failed call (deferred feedback for streaming)."""
        self._on_failure()

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._set_state(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
