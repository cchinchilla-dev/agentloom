"""Shared HTTP error helpers for provider adapters.

Centralizes the 429/5xx mapping and extra-kwarg allowlisting so every
adapter handles ``Retry-After`` uniformly and never silently drops a
caller-supplied payload key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentloom.exceptions import ProviderError, RateLimitError

if TYPE_CHECKING:
    import httpx

# Passed through by the engine/gateway for bookkeeping; the HTTP layer
# should ignore rather than reject or forward these.
_PASSTHROUGH_KWARGS = frozenset({"step_id"})


def parse_retry_after(value: str | None) -> float | None:
    """Parse the ``Retry-After`` header as seconds. HTTP-date form is not
    supported — upstream providers we talk to use integer seconds."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def raise_for_status(provider: str, response: httpx.Response) -> None:
    """Map an HTTP response to the right exception type.

    429 → ``RateLimitError`` so the gateway can back off the rate-limiter
    bucket without counting the call against the circuit breaker.
    Other non-2xx → ``ProviderError``.
    """
    if response.status_code == 429:
        raise RateLimitError(
            provider,
            retry_after_s=parse_retry_after(response.headers.get("Retry-After")),
        )
    if response.status_code != 200:
        raise ProviderError(
            provider,
            f"API error {response.status_code}: {response.text}",
            status_code=response.status_code,
        )


def validate_extra_kwargs(
    provider: str,
    method: str,
    kwargs: dict[str, Any],
    allowlist: frozenset[str],
) -> dict[str, Any]:
    """Return only the allowlisted keys; raise on anything unknown."""
    unknown = set(kwargs) - allowlist - _PASSTHROUGH_KWARGS
    if unknown:
        raise TypeError(
            f"Unsupported parameters for {provider}.{method}: {sorted(unknown)}. "
            f"Allowed extras: {sorted(allowlist)}"
        )
    return {k: v for k, v in kwargs.items() if k in allowlist}
