"""Provider Gateway — unified interface with fallback, circuit breaker, and rate limiting."""

from __future__ import annotations

import logging
from typing import Any

from agentloom.exceptions import CircuitOpenError, ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse
from agentloom.resilience.circuit_breaker import CircuitBreaker
from agentloom.resilience.rate_limiter import RateLimiter

logger = logging.getLogger("agentloom.gateway")


class ProviderEntry:
    """A registered provider with its associated resilience components."""

    def __init__(
        self,
        provider: BaseProvider,
        priority: int = 0,
        is_fallback: bool = False,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
        models: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.priority = priority
        self.is_fallback = is_fallback
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name=provider.name)
        self.rate_limiter = rate_limiter
        self.models = models or []


class ProviderGateway:
    """Central provider routing with fallback and resilience.

    Routes model requests to the appropriate provider. If a provider fails
    and fallbacks are configured, tries the next provider automatically.
    """

    def __init__(self) -> None:
        self._providers: list[ProviderEntry] = []
        self._model_mapping: dict[str, list[ProviderEntry]] = {}
        self._observer: Any | None = None

    def set_observer(self, observer: Any) -> None:
        """Attach an observer to receive circuit breaker state changes."""
        self._observer = observer
        # Wire existing circuit breakers to the observer
        for entry in self._providers:
            self._wire_circuit_callback(entry)

    def register(
        self,
        provider: BaseProvider,
        priority: int = 0,
        is_fallback: bool = False,
        models: list[str] | None = None,
        max_rpm: int = 60,
        max_tpm: int = 100_000,
        circuit_fail_threshold: int = 5,
        circuit_reset_timeout: float = 60.0,
    ) -> None:
        """Register a provider with its configuration."""
        entry = ProviderEntry(
            provider=provider,
            priority=priority,
            is_fallback=is_fallback,
            circuit_breaker=CircuitBreaker(
                name=provider.name,
                fail_threshold=circuit_fail_threshold,
                reset_timeout=circuit_reset_timeout,
            ),
            rate_limiter=RateLimiter(
                max_requests_per_minute=max_rpm,
                max_tokens_per_minute=max_tpm,
            ),
            models=models or [],
        )
        if self._observer:
            self._wire_circuit_callback(entry)
        self._providers.append(entry)
        self._providers.sort(key=lambda e: e.priority)

        for model in entry.models:
            self._model_mapping.setdefault(model, []).append(entry)
            self._model_mapping[model].sort(key=lambda e: e.priority)

    def _wire_circuit_callback(self, entry: ProviderEntry) -> None:
        """Connect a provider's circuit breaker to the observer."""
        obs = self._observer

        def _on_change(name: str, old: str, new: str) -> None:
            hook = getattr(obs, "on_circuit_state_change", None)
            if hook:
                hook(name, old, new)

        entry.circuit_breaker.on_state_change = _on_change

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Route a completion request to the appropriate provider with fallback."""
        # TODO: cache this lookup, rebuilding every call is wasteful
        candidates = self._get_candidates(model)

        if not candidates:
            raise ProviderError(
                "gateway",
                f"No provider registered for model '{model}'",
            )

        errors: list[str] = []

        for entry in candidates:
            try:
                if entry.rate_limiter:
                    # Estimate prompt tokens from message length (~4 chars per token)
                    estimated_tokens = sum(len(m.get("content", "")) for m in messages) // 4
                    await entry.rate_limiter.acquire(token_count=estimated_tokens)

                async def _call(e: ProviderEntry = entry) -> ProviderResponse:
                    return await e.provider.complete(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                response = await entry.circuit_breaker.call(_call)

                # Consume response tokens from rate limiter budget
                if entry.rate_limiter and response.usage.completion_tokens:
                    await entry.rate_limiter.consume_response_tokens(response.usage.completion_tokens)

                logger.debug(
                    "Provider '%s' responded for model '%s'",
                    entry.provider.name,
                    model,
                )
                return response

            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                errors.append(msg)
                logger.warning(msg)
                continue

            except Exception as e:
                msg = f"Provider '{entry.provider.name}' failed: {e}"
                errors.append(msg)
                logger.warning(msg)
                if self._observer:
                    error_hook = getattr(self._observer, "on_provider_error", None)
                    if error_hook:
                        error_hook(entry.provider.name, type(e).__name__)
                continue

        raise ProviderError(
            "gateway",
            f"All providers failed for model '{model}': " + "; ".join(errors),
        )

    def _get_candidates(self, model: str) -> list[ProviderEntry]:
        """Get provider candidates for a model, sorted by priority."""
        if model in self._model_mapping:
            return self._model_mapping[model]

        candidates = [e for e in self._providers if e.provider.supports_model(model)]

        fallbacks = [e for e in self._providers if e.is_fallback and e not in candidates]
        return candidates + fallbacks

    async def close(self) -> None:
        """Close all provider connections."""
        for entry in self._providers:
            await entry.provider.close()
