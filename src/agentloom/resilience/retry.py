"""Retry primitives: exponential backoff with jitter, and retryability rules.

The engine's step retry loop (``WorkflowEngine._run_step``) reads them, so
retry semantics have a single source of truth.
"""

from __future__ import annotations

import random

DEFAULT_RETRYABLE_STATUS_CODES: list[int] = [429, 500, 502, 503, 504]


def compute_backoff(base: float, attempt: int, maximum: float, jitter: bool) -> float:
    """``base ** (attempt + 1)`` capped at ``maximum``, with optional ±25% jitter."""
    delay = min(base ** (attempt + 1), maximum)
    if jitter:
        delay *= 1.0 + random.uniform(-0.25, 0.25)
    return max(0.0, delay)


def extract_status_code(exc: BaseException) -> int | None:
    """Return the HTTP status carried by *exc*, or ``None`` if it has none.

    Handles three common shapes:

    - ``ProviderError`` / ``RateLimitError`` expose ``status_code`` directly.
    - ``httpx.HTTPStatusError`` (and its subclasses) carry the status under
      ``exc.response.status_code`` — the bare exception has no
      ``status_code`` attribute, so a naive ``getattr`` would silently miss
      it and treat the failure as transient.
    - Everything else (network errors, generic provider failures, parser
      hiccups) returns ``None`` and the caller can treat it as transient.
    """
    code = getattr(exc, "status_code", None)
    if code is not None:
        return int(code)
    response = getattr(exc, "response", None)
    if response is not None:
        rcode = getattr(response, "status_code", None)
        if rcode is not None:
            return int(rcode)
    return None


def _has_permanent_marker(exc: BaseException) -> bool:
    """Return True if *exc* or anything in its cause chain sets ``is_retryable = False``.

    Walks ``__cause__`` and ``__context__`` so a non-retryable error
    wrapped by ``StepError`` (or any other classifier the engine adds for
    step-id context) still surfaces as permanent. Pydantic
    ``ValidationError`` is special-cased because it's outside the
    AgentLoom class hierarchy and has no place to hang the attribute.
    """
    from pydantic import ValidationError as PydanticValidationError

    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if getattr(cursor, "is_retryable", None) is False:
            return True
        if isinstance(cursor, PydanticValidationError):
            return True
        # The explicit ``__cause__`` wins over the implicit ``__context__`` when both are set.
        cursor = cursor.__cause__ or cursor.__context__
    return False


def is_retryable_exception(exc: BaseException, codes: list[int]) -> bool:
    """Return True if *exc* should trigger a retry under *codes*.

    Decision order:

    1. **Explicit permanent marker.** If *exc* (or any cause in its
       chain) carries ``is_retryable = False`` — ``SandboxViolationError``,
       ``AttachmentResolutionError``, ``ToolNotFoundError``,
       ``TemplateError``, ``ValidationError``, Pydantic's
       ``ValidationError`` — bail out immediately.
    2. **HTTP status code.** ``ProviderError``, ``RateLimitError``, and
       ``httpx.HTTPStatusError`` expose a status either directly or via
       ``exc.response.status_code``; the code must be in *codes*.
    3. **Status-less default.** Network errors and generic provider
       hiccups are treated as transient and retried.
    """
    if _has_permanent_marker(exc):
        return False
    code = extract_status_code(exc)
    if code is None:
        return True
    return code in codes


def classify_failure(exc: BaseException, codes: list[int]) -> str:
    """Return the ``error_classification`` of a step that failed with *exc*.

    ``"transient"`` or ``"permanent"``, by the rule of
    :func:`is_retryable_exception`. Steps that return a failed result
    instead of raising set it, so the engine retries them under the same
    rule as an exception.
    """
    return "transient" if is_retryable_exception(exc, codes) else "permanent"
