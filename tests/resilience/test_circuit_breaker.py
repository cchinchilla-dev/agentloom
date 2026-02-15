"""Tests for the CircuitBreaker module."""

from __future__ import annotations

import time

import pytest

from agentloom.exceptions import CircuitOpenError
from agentloom.resilience.circuit_breaker import CircuitBreaker, CircuitState


class TestClosedState:
    """Test circuit breaker in the closed (normal) state."""

    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

    async def test_successful_call_stays_closed(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=3)

        async def success() -> str:
            return "ok"

        result = await cb.call(success)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    async def test_failure_below_threshold_stays_closed(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=3)

        async def failing() -> str:
            raise RuntimeError("boom")

        # Fail twice (threshold is 3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2


class TestOpenState:
    """Test circuit breaker transitioning to and staying in open state."""

    async def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=3)

        async def failing() -> str:
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    async def test_open_circuit_rejects_calls(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2)

        async def failing() -> str:
            raise RuntimeError("boom")

        # Trigger open state
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        # Next call should be rejected with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await cb.call(failing)

    async def test_open_circuit_rejects_even_successful_calls(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2)

        async def failing() -> str:
            raise RuntimeError("boom")

        async def success() -> str:
            return "ok"

        # Trigger open state
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        # Even a "success" callable should be rejected
        with pytest.raises(CircuitOpenError):
            await cb.call(success)


class TestHalfOpenState:
    """Test circuit breaker half-open state and recovery."""

    async def test_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2, reset_timeout=0.1)

        async def failing() -> str:
            raise RuntimeError("boom")

        # Trigger open state
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        # Simulate time advancing past the reset timeout by manipulating
        # the internal _last_failure_time so the elapsed time exceeds reset_timeout
        cb._last_failure_time = time.monotonic() - 1.0
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_success_closes_circuit(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2, reset_timeout=0.1)

        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("boom")
            return "recovered"

        # Trigger open state
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(flaky)

        assert cb.state == CircuitState.OPEN

        # Simulate time passing by shifting last failure time into the past
        cb._last_failure_time = time.monotonic() - 1.0
        assert cb.state == CircuitState.HALF_OPEN

        # Successful call in half-open should close the circuit
        result = await cb.call(flaky)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens_circuit(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2, reset_timeout=0.1)

        async def always_failing() -> str:
            raise RuntimeError("still broken")

        # Trigger open state
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(always_failing)

        assert cb.state == CircuitState.OPEN

        # Simulate time passing by shifting last failure time into the past
        cb._last_failure_time = time.monotonic() - 1.0
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen the circuit
        with pytest.raises(RuntimeError):
            await cb.call(always_failing)

        assert cb.state == CircuitState.OPEN


class TestReset:
    """Test manual circuit breaker reset."""

    async def test_manual_reset(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2)

        async def failing() -> str:
            raise RuntimeError("boom")

        # Trigger open state
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerConfig:
    """Test circuit breaker with different configurations."""

    async def test_threshold_of_one(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=1)

        async def failing() -> str:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(failing)

        assert cb.state == CircuitState.OPEN

    async def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=3)

        async def failing() -> str:
            raise RuntimeError("boom")

        async def success() -> str:
            return "ok"

        # Two failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)
        assert cb.failure_count == 2

        # One success resets the count
        await cb.call(success)
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED
