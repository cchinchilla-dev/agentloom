"""The ``ResponseMeta`` that each adapter attaches to its responses (#166)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agentloom.contracts.stable import ResponseMeta

if TYPE_CHECKING:
    import httpx

# Header names arrive lowercased from httpx. Never widen this to auth headers.
_RATE_LIMIT_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")


def response_meta(
    sampling: Mapping[str, Any],
    headers: httpx.Headers,
    *,
    top_p_key: str = "top_p",
    request_id: str | None = None,
    system_fingerprint: str | None = None,
    model_version: str | None = None,
) -> ResponseMeta:
    """Metadata of a response whose headers just arrived.

    ``sampling`` is the part of the request body that holds the sampling
    parameters (the body itself, or the object an API nests them in), and
    ``top_p_key`` the name the API gives to ``top_p`` there.
    """
    return ResponseMeta(
        temperature=sampling.get("temperature"),
        top_p=sampling.get(top_p_key),
        seed=sampling.get("seed"),
        system_fingerprint=system_fingerprint,
        model_version=model_version,
        request_id=request_id,
        rate_limit={
            name: value
            for name, value in headers.items()
            if name.startswith(_RATE_LIMIT_PREFIXES) or name == "retry-after"
        },
        received_at=datetime.now(UTC),
    )
