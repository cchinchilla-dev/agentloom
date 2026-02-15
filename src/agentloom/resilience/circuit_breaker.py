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

    def __init__(
        self,
        name: str = "",
        fail_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
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
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

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
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
        self._failure_count = 0
        self._success_count += 1

    def _on_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.fail_threshold:
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
