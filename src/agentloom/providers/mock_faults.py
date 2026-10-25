"""Deterministic faults in a mock recording (#123).

An entry, or one turn of a step's list, can declare ``"fault": {...}``
instead of an answer::

    "answer": [{"fault": {"type": "http_error", "status_code": 500}}, {"content": "ok"}]

``MockProvider`` then fails that call every time the run reaches it, with the
exception a real adapter raises for that failure, so the gateway's circuit
breaker and fallback and the engine's retries take the path they take on a
real one.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from agentloom.exceptions import ProviderError, RateLimitError


class _Fault(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HttpErrorFault(_Fault):
    """The provider answers ``status_code``: 429 is a rate limit, anything else an API error."""

    type: Literal["http_error"]
    status_code: int = Field(ge=400, le=599)
    retry_after_s: float | None = Field(default=None, ge=0)
    body: str = ""


class TimeoutFault(_Fault):
    """No answer arrives: the call fails after ``after_ms``."""

    type: Literal["timeout"]
    after_ms: float = Field(default=0, ge=0)


class ConnectionResetFault(_Fault):
    """The connection fails before any answer."""

    type: Literal["connection_reset"]


class StreamTruncateFault(_Fault):
    """A stream delivers the entry's first ``after_chunks`` words, then the connection drops."""

    type: Literal["stream_truncate"]
    after_chunks: int = Field(ge=0)


MockFault = Annotated[
    HttpErrorFault | TimeoutFault | ConnectionResetFault | StreamTruncateFault,
    Field(discriminator="type"),
]
_FAULT: TypeAdapter[MockFault] = TypeAdapter(MockFault)


def parse_fault(raw: object, where: str) -> MockFault:
    """The fault an entry declares; ``ValueError`` naming ``where`` when it is malformed."""
    try:
        return _FAULT.validate_python(raw)
    except PydanticValidationError as exc:
        raise ValueError(f"{where} declares an invalid fault: {exc}") from exc


class StreamCutError(ProviderError):
    """A ``stream_truncate`` fault: the words delivered before the connection dropped."""

    def __init__(self, provider: str, chunks: list[str]) -> None:
        self.chunks = chunks
        super().__init__(
            provider, f"ReadError: the connection dropped after {len(chunks)} chunk(s) (mock fault)"
        )


def raise_fault(provider: str, fault: MockFault, content: str) -> NoReturn:
    """Raise what a real adapter raises for ``fault``; ``content`` feeds a cut stream."""
    if isinstance(fault, HttpErrorFault):
        if fault.status_code == 429:
            raise RateLimitError(provider, retry_after_s=fault.retry_after_s)
        raise ProviderError(
            provider, f"API error {fault.status_code}: {fault.body}", status_code=fault.status_code
        )
    if isinstance(fault, TimeoutFault):
        raise ProviderError(provider, "ReadTimeout: no answer (mock fault)")
    if isinstance(fault, ConnectionResetFault):
        raise ProviderError(provider, "ConnectError: connection reset (mock fault)")
    raise StreamCutError(provider, re.findall(r"\S+\s*", content)[: fault.after_chunks])
