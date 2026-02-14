"""Tests for the retry_with_policy module."""

from __future__ import annotations

import pytest

from agentloom.resilience.retry import RetryPolicy, retry_with_policy


class TestSuccessOnFirstTry:
    """Test that successful calls return immediately without retries."""

    async def test_returns_result_immediately(self) -> None:
        call_count = 0

        async def success() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        policy = RetryPolicy(max_retries=3, jitter=False)
        result = await retry_with_policy(success, policy, "test-op")

        assert result == "ok"
        assert call_count == 1

    async def test_no_retry_on_success(self) -> None:
        attempts: list[int] = []

        async def tracked_success() -> int:
            attempts.append(len(attempts) + 1)
            return 42

        policy = RetryPolicy(max_retries=5, jitter=False)
        result = await retry_with_policy(tracked_success, policy, "test-op")

        assert result == 42
        assert len(attempts) == 1


class TestSuccessOnRetry:
    """Test that intermittent failures are retried successfully."""

    async def test_succeeds_on_second_attempt(self) -> None:
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("temporary failure")
            return "recovered"

        policy = RetryPolicy(max_retries=3, backoff_base=0.01, jitter=False)
        result = await retry_with_policy(flaky, policy, "test-op")

        assert result == "recovered"
        assert call_count == 2

    async def test_succeeds_on_third_attempt(self) -> None:
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary failure")
            return "finally worked"

        policy = RetryPolicy(max_retries=3, backoff_base=0.01, jitter=False)
        result = await retry_with_policy(flaky, policy, "test-op")

        assert result == "finally worked"
        assert call_count == 3

    async def test_succeeds_on_last_retry(self) -> None:
        call_count = 0

        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError("temporary failure")
            return "last chance"

        policy = RetryPolicy(max_retries=3, backoff_base=0.01, jitter=False)
        result = await retry_with_policy(flaky, policy, "test-op")

        assert result == "last chance"
        # 1 initial + 3 retries = 4 attempts
        assert call_count == 4


class TestAllRetriesFail:
    """Test behavior when all retries are exhausted."""

    async def test_raises_last_exception(self) -> None:
        call_count = 0

        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"failure #{call_count}")

        policy = RetryPolicy(max_retries=2, backoff_base=0.01, jitter=False)

        with pytest.raises(RuntimeError, match="failure #3"):
            await retry_with_policy(always_fails, policy, "test-op")

        # 1 initial + 2 retries = 3 attempts
        assert call_count == 3

    async def test_zero_retries_fails_immediately(self) -> None:
        call_count = 0

        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("no retries")

        policy = RetryPolicy(max_retries=0, jitter=False)

        with pytest.raises(ValueError, match="no retries"):
            await retry_with_policy(always_fails, policy, "test-op")

        assert call_count == 1

    async def test_preserves_exception_type(self) -> None:
        async def type_error() -> str:
            raise TypeError("wrong type")

        policy = RetryPolicy(max_retries=1, backoff_base=0.01, jitter=False)

        with pytest.raises(TypeError, match="wrong type"):
            await retry_with_policy(type_error, policy, "test-op")


class TestRetryPolicyConfig:
    """Test RetryPolicy configuration and defaults."""

    def test_default_policy(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_base == 2.0
        assert policy.backoff_max == 60.0
        assert policy.jitter is True

    def test_custom_policy(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            backoff_base=1.5,
            backoff_max=30.0,
            jitter=False,
        )
        assert policy.max_retries == 5
        assert policy.backoff_base == 1.5
        assert policy.backoff_max == 30.0
        assert policy.jitter is False

    def test_retryable_status_codes(self) -> None:
        policy = RetryPolicy()
        assert 429 in policy.retryable_status_codes
        assert 500 in policy.retryable_status_codes
        assert 503 in policy.retryable_status_codes
