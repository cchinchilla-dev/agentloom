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


def is_retryable_exception(exc: BaseException, codes: list[int]) -> bool:
    """Return True if *exc* should trigger a retry under *codes*.

    If *exc* exposes a ``status_code`` attribute (e.g. ``ProviderError``,
    ``RateLimitError``, ``httpx.HTTPStatusError``), the code must be in
    *codes*. Exceptions without a status code are retryable by default —
    a network error or generic provider failure is treated as transient.
    """
    code = getattr(exc, "status_code", None)
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
                    getattr(e, "status_code", None),
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
