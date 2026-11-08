"""Tests for the CircuitBreaker module."""

from __future__ import annotations

import time

import anyio
import pytest

from agentloom.exceptions import CircuitOpenError
from agentloom.resilience.circuit_breaker import CircuitBreaker, CircuitState


class TestClosedState:
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

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2


class TestOpenState:
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

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await cb.call(failing)

    async def test_open_circuit_rejects_even_successful_calls(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2)

        async def failing() -> str:
            raise RuntimeError("boom")

        async def success() -> str:
            return "ok"

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await cb.call(success)


class TestHalfOpenState:
    async def test_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2, reset_timeout=0.1)

        async def failing() -> str:
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        # Backdate the last failure past reset_timeout instead of sleeping.
        cb._last_failure_time = time.monotonic() - 1.0
        assert cb._maybe_transition_to_half_open() == CircuitState.HALF_OPEN
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

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(flaky)

        assert cb.state == CircuitState.OPEN

        # Backdate the last failure past reset_timeout.
        cb._last_failure_time = time.monotonic() - 1.0

        result = await cb.call(flaky)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens_circuit(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2, reset_timeout=0.1)

        async def always_failing() -> str:
            raise RuntimeError("still broken")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(always_failing)

        assert cb.state == CircuitState.OPEN

        # Backdate the last failure past reset_timeout.
        cb._last_failure_time = time.monotonic() - 1.0

        with pytest.raises(RuntimeError):
            await cb.call(always_failing)

        assert cb.state == CircuitState.OPEN


class TestReset:
    async def test_manual_reset(self) -> None:
        cb = CircuitBreaker(name="test", fail_threshold=2)

        async def failing() -> str:
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerConfig:
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

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)
        assert cb.failure_count == 2

        await cb.call(success)
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED


class TestStatePropertyIsPureRead:
    """Reading `.state` must not mutate the breaker or fire callbacks."""

    def test_state_property_is_pure_read(self) -> None:
        events: list[tuple[str, str, str]] = []

        def on_change(name: str, old: str, new: str) -> None:
            events.append((name, old, new))

        cb = CircuitBreaker(
            name="test", fail_threshold=1, reset_timeout=0.1, on_state_change=on_change
        )

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        events.clear()

        # Backdate past reset_timeout: a read must still not move the breaker to HALF_OPEN.
        cb._last_failure_time = time.monotonic() - 1.0
        snapshot_calls = cb._half_open_calls
        _ = cb.state
        _ = cb.state
        _ = cb.state
        assert events == []
        assert cb._half_open_calls == snapshot_calls
        assert cb.state == CircuitState.OPEN

        # The explicit admission path does transition.
        assert cb._maybe_transition_to_half_open() == CircuitState.HALF_OPEN
        assert events == [("test", "open", "half_open")]


class TestCallExclude:
    """``call(exclude=...)`` re-raises excluded errors uncharged and frees any HALF_OPEN slot."""

    async def test_excluded_exception_does_not_count_as_failure(self) -> None:
        cb = CircuitBreaker(name="t", fail_threshold=2, reset_timeout=60.0)

        class HarmlessError(Exception):
            pass

        async def raises() -> None:
            raise HarmlessError("rate limited, not a fault")

        for _ in range(5):
            with pytest.raises(HarmlessError):
                await cb.call(raises, exclude=(HarmlessError,))

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    async def test_excluded_exception_in_half_open_releases_slot(self) -> None:
        cb = CircuitBreaker(name="t", fail_threshold=1, reset_timeout=0.01, half_open_max_calls=1)

        async def fails() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fails)
        assert cb.state == CircuitState.OPEN

        await anyio.sleep(0.02)
        assert cb._maybe_transition_to_half_open() == CircuitState.HALF_OPEN

        class ExcludedError(Exception):
            pass

        async def excluded_fail() -> None:
            raise ExcludedError("not a fault")

        with pytest.raises(ExcludedError):
            await cb.call(excluded_fail, exclude=(ExcludedError,))

        # Slot must have been released so the next admission can proceed.
        assert cb._half_open_calls == 0


class TestSetStateNoop:
    """``_set_state`` to the current state is a no-op: ``on_state_change`` does not fire."""

    def test_set_state_to_same_state_does_not_fire_callback(self) -> None:
        cb = CircuitBreaker(name="t")
        events: list[tuple[str, str, str]] = []
        cb.on_state_change = lambda name, old, new: events.append((name, old, new))

        cb._set_state(CircuitState.CLOSED)
        assert events == []

        cb._set_state(CircuitState.OPEN)
        assert len(events) == 1

        cb._set_state(CircuitState.OPEN)
        assert len(events) == 1


class TestAllowRequestExplicitHalfOpenGate:
    """``allow_request()`` raises CircuitOpenError when HALF_OPEN slots are exhausted."""

    async def test_allow_request_raises_when_half_open_slots_exhausted(self) -> None:
        cb = CircuitBreaker(name="t", fail_threshold=1, reset_timeout=0.01, half_open_max_calls=1)

        async def fails() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(fails)
        assert cb.state == CircuitState.OPEN

        await anyio.sleep(0.02)
        assert cb._maybe_transition_to_half_open() == CircuitState.HALF_OPEN

        # First allow_request() consumes the only half-open slot.
        cb.allow_request()
        with pytest.raises(CircuitOpenError):
            cb.allow_request()
