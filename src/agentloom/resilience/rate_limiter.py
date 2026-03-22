"""Token bucket rate limiter for provider API calls."""

from __future__ import annotations

import time

import anyio


class RateLimiter:
    """Token bucket rate limiter for controlling API request rates.

    Supports both request-based and token-based rate limiting.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_tokens_per_minute: int = 100_000,
    ) -> None:
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute

        # Request bucket
        self._request_tokens = float(max_requests_per_minute)
        self._request_last_refill = time.monotonic()

        # Token bucket
        self._token_tokens = float(max_tokens_per_minute)
        self._token_last_refill = time.monotonic()

        self._lock = anyio.Lock()

    async def consume_response_tokens(self, token_count: int) -> None:
        """Consume tokens from the bucket after receiving a response.

        Call this after a completion to account for response tokens
        against the TPM budget.
        """
        if token_count <= 0:
            return
        async with self._lock:
            self._refill()
            self._token_tokens = max(0.0, self._token_tokens - token_count)

    async def acquire(self, token_count: int = 0) -> None:
        """Acquire permission to make a request.

        Blocks until rate limit allows the request.

        Args:
            token_count: Estimated token count for this request (0 = request-only limiting).
        """
        while True:
            async with self._lock:
                self._refill()

                # Check request bucket
                if self._request_tokens < 1.0:
                    wait_time = (1.0 - self._request_tokens) / (self.max_rpm / 60.0)
                else:
                    # Check token bucket (if applicable)
                    if token_count > 0 and self._token_tokens < token_count:
                        wait_time = (token_count - self._token_tokens) / (self.max_tpm / 60.0)
                    else:
                        # Consume tokens
                        self._request_tokens -= 1.0
                        if token_count > 0:
                            self._token_tokens -= token_count
                        return

            # Wait outside the lock
            await anyio.sleep(min(wait_time, 5.0))

    def _refill(self) -> None:
        """Refill token buckets based on elapsed time."""
        now = time.monotonic()

        # Refill request bucket
        elapsed = now - self._request_last_refill
        self._request_tokens = min(
            float(self.max_rpm),
            self._request_tokens + elapsed * (self.max_rpm / 60.0),
        )
        self._request_last_refill = now

        # Refill token bucket
        elapsed_t = now - self._token_last_refill
        self._token_tokens = min(
            float(self.max_tpm),
            self._token_tokens + elapsed_t * (self.max_tpm / 60.0),
        )
        self._token_last_refill = now
