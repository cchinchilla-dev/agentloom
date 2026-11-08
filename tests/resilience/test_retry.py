"""Tests for the retry_with_policy module."""

from __future__ import annotations

import pytest

from agentloom.resilience.retry import RetryPolicy, retry_with_policy


class TestSuccessOnFirstTry:
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
    """A status outside ``retryable_status_codes`` stops retrying, so a 4xx burns no budget."""

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
        # ``httpx.HTTPStatusError`` keeps the status on ``exc.response``, not on the exception.
        from types import SimpleNamespace

        from agentloom.resilience.retry import is_retryable_exception

        class FakeHTTPStatusError(Exception):
            def __init__(self, status: int) -> None:
                super().__init__(f"HTTP {status}")
                self.response = SimpleNamespace(status_code=status)

        assert is_retryable_exception(FakeHTTPStatusError(404), [429, 500, 502, 503, 504]) is False
        assert is_retryable_exception(FakeHTTPStatusError(503), [429, 500, 502, 503, 504]) is True


class TestPermanentMarkerClassification:
    """An exception with ``is_retryable = False`` stops the retry loop at the first attempt."""

    def test_sandbox_violation_marked_non_retryable(self) -> None:
        from agentloom.exceptions import SandboxViolationError
        from agentloom.resilience.retry import is_retryable_exception

        exc = SandboxViolationError("shell_command", "denied")
        assert exc.is_retryable is False
        assert is_retryable_exception(exc, [429, 500]) is False

    def test_tool_not_found_marked_non_retryable(self) -> None:
        from agentloom.exceptions import ToolNotFoundError
        from agentloom.resilience.retry import is_retryable_exception

        exc = ToolNotFoundError("missing_tool", ["a", "b"])
        assert exc.is_retryable is False
        assert is_retryable_exception(exc, [429, 500]) is False
        # KeyError compatibility: call sites using ``except KeyError`` keep working.
        assert isinstance(exc, KeyError)

    def test_attachment_resolution_marked_non_retryable(self) -> None:
        from agentloom.exceptions import AttachmentResolutionError
        from agentloom.resilience.retry import is_retryable_exception

        exc = AttachmentResolutionError("file.png", "size limit")
        assert exc.is_retryable is False
        assert is_retryable_exception(exc, [429, 500]) is False
        # ValueError compatibility: ``except ValueError`` clauses keep working.
        assert isinstance(exc, ValueError)

    def test_template_error_marked_non_retryable(self) -> None:
        from agentloom.core.templates import TemplateError
        from agentloom.resilience.retry import is_retryable_exception

        exc = TemplateError("state.missing_key")
        assert exc.is_retryable is False
        assert is_retryable_exception(exc, [429, 500]) is False

    def test_validation_error_marked_non_retryable(self) -> None:
        from agentloom.exceptions import ValidationError
        from agentloom.resilience.retry import is_retryable_exception

        exc = ValidationError("bad workflow")
        assert exc.is_retryable is False
        assert is_retryable_exception(exc, [429, 500]) is False

    def test_pydantic_validation_error_marked_non_retryable(self) -> None:
        """Pydantic's ``ValidationError`` cannot carry ``is_retryable``, so it is special-cased."""
        from pydantic import BaseModel
        from pydantic import ValidationError as PydanticValidationError

        from agentloom.resilience.retry import is_retryable_exception

        class _M(BaseModel):
            x: int

        try:
            _M.model_validate({"x": "not-an-int"})
            raise AssertionError("Pydantic should have raised")
        except PydanticValidationError as exc:
            assert is_retryable_exception(exc, [429, 500]) is False

    def test_security_error_marked_non_retryable(self) -> None:
        from agentloom.exceptions import SecurityError
        from agentloom.resilience.retry import is_retryable_exception

        exc = SecurityError("expression refused", expression="getattr(...)")
        assert is_retryable_exception(exc, [429, 500]) is False

    def test_marker_walks_cause_chain(self) -> None:
        """The retry layer walks ``__cause__``, past the engine's ``StepError`` wrapper."""
        from agentloom.exceptions import SandboxViolationError, StepError
        from agentloom.resilience.retry import is_retryable_exception

        cause = SandboxViolationError("shell_command", "denied")
        try:
            try:
                raise cause
            except SandboxViolationError as inner:
                raise StepError("s1", "wrapped") from inner
        except StepError as wrapped:
            assert is_retryable_exception(wrapped, [429, 500]) is False

    def test_marker_walks_context_chain(self) -> None:
        """Implicit chaining through ``__context__`` is walked as well."""
        from agentloom.core.templates import TemplateError
        from agentloom.exceptions import StepError
        from agentloom.resilience.retry import is_retryable_exception

        try:
            try:
                raise TemplateError("state.x")
            except TemplateError:
                raise StepError("s1", "during cleanup")  # noqa: B904 — intentional implicit chain
        except StepError as wrapped:
            assert is_retryable_exception(wrapped, [429, 500]) is False

    def test_status_less_exception_still_transient_when_no_marker(self) -> None:
        """A status-less exception without a marker (a network error) is still transient."""
        from agentloom.resilience.retry import is_retryable_exception

        assert is_retryable_exception(RuntimeError("transient"), [429, 500]) is True
