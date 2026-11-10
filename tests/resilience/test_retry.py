"""Tests for the retry primitives."""

from __future__ import annotations


class TestRetryableStatus:
    """Only a status in ``retryable_status_codes`` is worth a retry."""

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
