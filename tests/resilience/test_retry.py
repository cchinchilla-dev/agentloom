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


class TestRetryWithPolicyNonRetryable:
    """``retry_with_policy`` must short-circuit on exceptions whose status
    code is not in ``policy.retryable_status_codes``. The break path at
    ``retry.py:89-94`` is what enforces this — without it, a permanent
    4xx would burn the full retry budget."""

    async def test_non_retryable_status_breaks_immediately(self) -> None:
        from agentloom.exceptions import ProviderError
        from agentloom.resilience.retry import RetryPolicy, retry_with_policy

        attempts: list[int] = []

        async def always_400() -> str:
            attempts.append(1)
            raise ProviderError("mock", "permanent failure", status_code=400)

        policy = RetryPolicy(max_retries=3, backoff_base=1.0, backoff_max=0.0, jitter=False)
        with pytest.raises(ProviderError):
            await retry_with_policy(always_400, policy, "smoke")

        assert len(attempts) == 1, (
            f"non-retryable status must not consume the retry budget, got {len(attempts)} attempts"
        )

    async def test_retryable_status_consumes_budget(self) -> None:
        # Counterpart — a 429 must keep retrying until the budget runs out.
        from agentloom.exceptions import RateLimitError
        from agentloom.resilience.retry import RetryPolicy, retry_with_policy

        attempts: list[int] = []

        async def always_429() -> str:
            attempts.append(1)
            raise RateLimitError("mock", retry_after_s=0.0)

        policy = RetryPolicy(max_retries=2, backoff_base=1.0, backoff_max=0.0, jitter=False)
        with pytest.raises(RateLimitError):
            await retry_with_policy(always_429, policy, "smoke")

        assert len(attempts) == 3, "max_retries=2 means 3 total attempts"

    def test_is_retryable_extracts_status_from_httpx_response(self) -> None:
        # ``httpx.HTTPStatusError`` does not expose ``status_code`` directly
        # — the status lives on ``exc.response.status_code``. The helper
        # must dig into the response so a permanent 404 from an attachment
        # download (or any direct httpx call site) bails out instead of
        # being treated as a transient status-less failure.
        from types import SimpleNamespace

        from agentloom.resilience.retry import is_retryable_exception

        class FakeHTTPStatusError(Exception):
            def __init__(self, status: int) -> None:
                super().__init__(f"HTTP {status}")
                self.response = SimpleNamespace(status_code=status)

        # 404 is not in the default retryable list — must NOT retry.
        assert is_retryable_exception(FakeHTTPStatusError(404), [429, 500, 502, 503, 504]) is False
        # 503 IS in the list — must retry.
        assert is_retryable_exception(FakeHTTPStatusError(503), [429, 500, 502, 503, 504]) is True
