"""Retry with exponential backoff and jitter.

The backoff and retryability primitives (`compute_backoff`,
`is_retryable_exception`) are imported by ``core.engine._execute_step``
so the engine and ``retry_with_policy`` share a single source of truth
for retry semantics — no parallel implementations to drift apart.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import anyio
from pydantic import BaseModel

T = TypeVar("T")
logger = logging.getLogger("agentloom.resilience")

DEFAULT_RETRYABLE_STATUS_CODES: list[int] = [429, 500, 502, 503, 504]


def compute_backoff(base: float, attempt: int, maximum: float, jitter: bool) -> float:
    """Exponential backoff capped at ``maximum`` with optional ±25% jitter.

    Used by both ``retry_with_policy`` and ``WorkflowEngine._execute_step``
    so step retries and gateway retries use the same waveform.
    """
    delay = min(base**attempt, maximum)
    if jitter:
        delay *= 1.0 + random.uniform(-0.25, 0.25)  # noqa: S311 — non-crypto jitter
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
    """Return True if *exc* (or anything in its cause chain) sets
    ``is_retryable = False``.

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
        # ``__cause__`` (explicit ``raise … from``) takes precedence over
        # ``__context__`` (implicit chain during exception handling) so
        # operator-marked causality wins when both are set.
        cursor = cursor.__cause__ or cursor.__context__
    return False


def is_retryable_exception(exc: BaseException, codes: list[int]) -> bool:
    """Return True if *exc* should trigger a retry under *codes*.

    Decision order:

    1. **Explicit permanent marker.** If *exc* (or any cause in its
       chain) carries ``is_retryable = False`` — ``SandboxViolationError``,
       ``AttachmentResolutionError``, ``ToolNotFoundError``,
       ``TemplateError``, ``ValidationError``, Pydantic's
       ``ValidationError`` — bail out immediately. Pre-0.5.0 the
       resilience layer burned the full retry budget on these and added
       10–127 s of backoff to a workflow that was never going to succeed.
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


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    jitter: bool = True
    retryable_status_codes: list[int] = DEFAULT_RETRYABLE_STATUS_CODES


async def retry_with_policy(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    policy: RetryPolicy,
    operation_name: str = "operation",
) -> T:
    """Execute an async callable with retry and exponential backoff.

    Args:
        coro_factory: Zero-argument callable returning a coroutine.
        policy: Retry configuration.
        operation_name: Name for logging.

    Returns:
        Result of the coroutine.

    Raises:
        The last exception if all retries are exhausted, or immediately
        if the exception is not retryable under ``policy.retryable_status_codes``.
    """
    last_exception: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exception = e

            if not is_retryable_exception(e, policy.retryable_status_codes):
                logger.debug(
                    "%s failed with non-retryable status %s; giving up",
                    operation_name,
                    extract_status_code(e),
                )
                break

            if attempt >= policy.max_retries:
                break

            backoff = compute_backoff(
                policy.backoff_base, attempt, policy.backoff_max, policy.jitter
            )
            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                operation_name,
                attempt + 1,
                policy.max_retries + 1,
                backoff,
                e,
            )

            await anyio.sleep(backoff)

    assert last_exception is not None
    raise last_exception
