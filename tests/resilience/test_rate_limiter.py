"""Tests for token bucket rate limiter."""

from __future__ import annotations

import time

from agentloom.resilience.rate_limiter import RateLimiter


class TestRateLimiterInit:
    def test_default_config(self) -> None:
        rl = RateLimiter()
        assert rl.max_rpm == 60
        assert rl.max_tpm == 100_000

    def test_custom_config(self) -> None:
        rl = RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=5000)
        assert rl.max_rpm == 10
        assert rl.max_tpm == 5000

    def test_initial_bucket_is_full(self) -> None:
        rl = RateLimiter(max_requests_per_minute=30)
        assert rl._request_tokens == 30.0


class TestAcquireRPM:
    async def test_single_acquire(self) -> None:
        rl = RateLimiter(max_requests_per_minute=60)
        await rl.acquire()
        assert rl._request_tokens < 60.0

    async def test_multiple_acquires_consume_tokens(self) -> None:
        rl = RateLimiter(max_requests_per_minute=100)
        for _ in range(5):
            await rl.acquire()
        assert rl._request_tokens <= 95.1  # small refill drift is expected

    async def test_acquire_does_not_go_negative(self) -> None:
        rl = RateLimiter(max_requests_per_minute=3)
        for _ in range(3):
            await rl.acquire()
        # Bucket should be near zero but refill may add a tiny amount
        assert rl._request_tokens < 1.0


class TestAcquireTPM:
    async def test_token_count_consumed(self) -> None:
        rl = RateLimiter(max_requests_per_minute=100, max_tokens_per_minute=10_000)
        await rl.acquire(token_count=500)
        assert rl._token_tokens < 10_000

    async def test_zero_token_count_skips_tpm(self) -> None:
        rl = RateLimiter(max_tokens_per_minute=10_000)
        initial = rl._token_tokens
        await rl.acquire(token_count=0)
        # Token bucket should NOT be consumed (only refill drift)
        assert rl._token_tokens >= initial - 1.0


class TestRefill:
    def test_refill_adds_tokens(self) -> None:
        rl = RateLimiter(max_requests_per_minute=60)
        rl._request_tokens = 0.0
        # Simulate time passing
        rl._request_last_refill = time.monotonic() - 1.0  # 1 second ago
        rl._refill()
        # 60 RPM = 1 per second, so ~1 token should be refilled
        assert rl._request_tokens >= 0.9

    def test_refill_caps_at_max(self) -> None:
        rl = RateLimiter(max_requests_per_minute=60)
        rl._request_tokens = 59.0
        rl._request_last_refill = time.monotonic() - 10.0  # 10 seconds ago
        rl._refill()
        assert rl._request_tokens == 60.0

    def test_token_bucket_refills(self) -> None:
        rl = RateLimiter(max_tokens_per_minute=6000)
        rl._token_tokens = 0.0
        rl._token_last_refill = time.monotonic() - 1.0  # 1 second ago
        rl._refill()
        # 6000 TPM = 100 per second
        assert rl._token_tokens >= 90.0
