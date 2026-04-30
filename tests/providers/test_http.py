"""Tests for the shared HTTP error helpers in `providers/_http.py`."""

from __future__ import annotations

import httpx
import pytest

from agentloom.exceptions import ProviderError, RateLimitError
from agentloom.providers._http import (
    parse_retry_after,
    raise_for_status,
    validate_extra_kwargs,
)


class TestParseRetryAfter:
    """`Retry-After` header parsing."""

    def test_returns_none_for_missing_header(self) -> None:
        assert parse_retry_after(None) is None

    def test_returns_none_for_empty_string(self) -> None:
        assert parse_retry_after("") is None

    def test_parses_integer_seconds(self) -> None:
        assert parse_retry_after("30") == 30.0

    def test_parses_float_seconds(self) -> None:
        assert parse_retry_after("12.5") == 12.5

    def test_returns_none_for_http_date(self) -> None:
        # HTTP-date form is documented as unsupported — must not crash.
        assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None

    def test_returns_none_for_garbage(self) -> None:
        assert parse_retry_after("not-a-number") is None


class TestRaiseForStatus:
    """HTTP status mapping."""

    def test_200_does_not_raise(self) -> None:
        resp = httpx.Response(status_code=200)
        raise_for_status("openai", resp)  # no exception

    def test_429_maps_to_rate_limit_error(self) -> None:
        resp = httpx.Response(status_code=429, headers={"Retry-After": "5"})
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status("openai", resp)
        assert exc_info.value.retry_after_s == 5.0

    def test_429_without_retry_after(self) -> None:
        resp = httpx.Response(status_code=429)
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status("anthropic", resp)
        assert exc_info.value.retry_after_s is None

    def test_500_maps_to_provider_error(self) -> None:
        resp = httpx.Response(status_code=500, text="server boom")
        with pytest.raises(ProviderError) as exc_info:
            raise_for_status("google", resp)
        assert "500" in str(exc_info.value)


class TestValidateExtraKwargs:
    """Allowlist enforcement and passthrough handling."""

    def test_returns_only_allowlisted_keys(self) -> None:
        out = validate_extra_kwargs(
            "openai",
            "complete",
            {"top_p": 0.9, "seed": 1},
            frozenset({"top_p", "seed"}),
        )
        assert out == {"top_p": 0.9, "seed": 1}

    def test_passthrough_step_id_not_in_output_or_error(self) -> None:
        out = validate_extra_kwargs(
            "openai",
            "complete",
            {"step_id": "my-step", "top_p": 0.5},
            frozenset({"top_p"}),
        )
        # step_id is silently dropped — must NOT appear in output.
        assert out == {"top_p": 0.5}

    def test_unknown_kwarg_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="Unsupported parameters for openai.complete"):
            validate_extra_kwargs(
                "openai",
                "complete",
                {"bogus": 1},
                frozenset({"top_p"}),
            )
