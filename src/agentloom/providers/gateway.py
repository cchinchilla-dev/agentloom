"""Provider Gateway — unified interface with fallback, circuit breaker, and rate limiting."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio

from agentloom.exceptions import CircuitOpenError, ProviderError
from agentloom.providers.base import BaseProvider, ProviderResponse, StreamResponse
from agentloom.providers.multimodal import estimate_content_tokens
from agentloom.resilience.circuit_breaker import CircuitBreaker, CircuitState
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
        self._candidate_cache: dict[str, list[ProviderEntry]] = {}
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

        self._candidate_cache.clear()

    def wrap_providers(self, factory: Callable[[BaseProvider], BaseProvider]) -> None:
        """Replace each registered provider with ``factory(provider)``.

        Useful for cross-cutting wrappers (e.g. ``RecordingProvider``) that
        need to intercept every registered provider without the caller
        knowing their concrete types or internal entry layout.
        """
        for entry in self._providers:
            entry.provider = factory(entry.provider)
        self._candidate_cache.clear()

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
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Route a completion request to the appropriate provider with fallback."""
        candidates = self._get_candidates(model)

        if not candidates:
            raise ProviderError(
                "gateway",
                f"No provider registered for model '{model}'",
            )

        errors: list[str] = []

        for entry in candidates:
            # Fast-fail when the circuit is open: do NOT consume the rate
            # limiter budget for traffic that will never reach the provider.
            if entry.circuit_breaker._maybe_transition_to_half_open() == CircuitState.OPEN:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                errors.append(msg)
                logger.warning(msg)
                continue

            try:
                if entry.rate_limiter:
                    # Estimate prompt tokens from message content
                    estimated_tokens = sum(
                        estimate_content_tokens(m.get("content", "")) for m in messages
                    )
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
                    resp_tokens = response.usage.completion_tokens
                    await entry.rate_limiter.consume_response_tokens(resp_tokens)

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

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        """Route a streaming request with fallback and resilience.

        NOTE: Mid-stream fallback is not supported.  Once a provider starts
        streaming successfully, the gateway commits to that provider.  If the
        connection drops mid-stream the error propagates to the caller and the
        engine's retry logic will re-attempt the entire step (potentially
        routing to a different provider if the circuit breaker has tripped).
        """
        candidates = self._get_candidates(model)

        if not candidates:
            raise ProviderError(
                "gateway",
                f"No provider registered for model '{model}'",
            )

        errors: list[str] = []

        for entry in candidates:
            try:
                entry.circuit_breaker.allow_request()
            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                errors.append(msg)
                logger.warning(msg)
                continue

            try:
                if entry.rate_limiter:
                    estimated_tokens = sum(
                        estimate_content_tokens(m.get("content", "")) for m in messages
                    )
                    await entry.rate_limiter.acquire(token_count=estimated_tokens)

                inner_sr = await entry.provider.stream(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # Wrap with resilience feedback
                outer_sr = StreamResponse(model=inner_sr.model, provider=inner_sr.provider)

                async def _wrapped_iter(
                    _inner: StreamResponse = inner_sr,
                    _entry: ProviderEntry = entry,
                    _outer: StreamResponse = outer_sr,
                ) -> AsyncIterator[str]:
                    cancelled_exc = anyio.get_cancelled_exc_class()
                    try:
                        raw_iter = _inner._iterator
                        if raw_iter is None:
                            _entry.circuit_breaker.record_success()
                            return
                        async for chunk in raw_iter:
                            yield chunk
                    except GeneratorExit:
                        # Caller aborted via aclose() — not a provider fault.
                        # The connection was established and producing chunks,
                        # so record a success to release the HALF_OPEN slot.
                        _entry.circuit_breaker.record_success()
                        raise
                    except cancelled_exc:
                        # Task-group / outer cancel scope — not a provider fault.
                        _entry.circuit_breaker.record_success()
                        raise
                    except Exception as exc:
                        _entry.circuit_breaker.record_failure()
                        if self._observer:
                            hook = getattr(self._observer, "on_provider_error", None)
                            if hook:
                                hook(_entry.provider.name, type(exc).__name__)
                        raise
                    else:
                        _entry.circuit_breaker.record_success()
                        if _entry.rate_limiter and _inner.usage.completion_tokens:
                            await _entry.rate_limiter.consume_response_tokens(
                                _inner.usage.completion_tokens
                            )
                    finally:
                        _outer.usage = _inner.usage
                        _outer.cost_usd = _inner.cost_usd
                        _outer.finish_reason = _inner.finish_reason
                        _outer.model = _inner.model

                outer_sr._set_iterator(_wrapped_iter())
                logger.debug(
                    "Provider '%s' streaming for model '%s'",
                    entry.provider.name,
                    model,
                )
                return outer_sr

            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                errors.append(msg)
                logger.warning(msg)
                continue

            except RateLimitError as e:
                # Throttled, not faulty. Do NOT record a breaker failure; the
                # provider is healthy. Try the next candidate (if any).
                msg = f"Provider '{entry.provider.name}' rate-limited: {e}"
                errors.append(msg)
                logger.warning(msg)
                continue

            except Exception as e:
                # Setup failures (rate limiter, provider.stream()) must feed
                # back to the circuit breaker — allow_request() already
                # incremented the half-open counter.
                entry.circuit_breaker.record_failure()
                msg = (
                    f"Provider '{entry.provider.name}' failed to start "
                    f"streaming for model '{model}': {e}"
                )
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

        if model in self._candidate_cache:
            return self._candidate_cache[model]

        candidates = [e for e in self._providers if e.provider.supports_model(model)]
        fallbacks = [e for e in self._providers if e.is_fallback and e not in candidates]
        result = candidates + fallbacks
        self._candidate_cache[model] = result
        return result

    async def close(self) -> None:
        """Close all provider connections."""
        for entry in self._providers:
            await entry.provider.close()
