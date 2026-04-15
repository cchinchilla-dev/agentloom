"""Retry with exponential backoff and jitter.

NOTE: retry_with_policy is not yet called by WorkflowEngine, which
reimplements retry inline.  Planned refactor will wire the engine
to use this function for consistent jitter and retryable-status-code
handling.
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


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    jitter: bool = True
    retryable_status_codes: list[int] = [429, 500, 502, 503, 504]


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
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exception = e

            if attempt >= policy.max_retries:
                break

            backoff = min(
                policy.backoff_base**attempt,
                policy.backoff_max,
            )

            if policy.jitter:
                backoff *= 1.0 + random.uniform(-0.25, 0.25)  # noqa: S311

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
