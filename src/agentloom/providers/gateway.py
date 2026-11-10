"""Provider Gateway — unified interface with fallback, circuit breaker, and rate limiting."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, NoReturn

import anyio

from agentloom.core.attempts import record_retry_after, record_spend
from agentloom.core.protocols import call_hook
from agentloom.exceptions import (
    CircuitOpenError,
    ProviderError,
    RateLimitError,
    RecordingMismatchError,
    UnsupportedRequestError,
)
from agentloom.observability.schema import GenAIOperationName
from agentloom.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    ProviderResponse,
    StreamResponse,
)
from agentloom.providers.multimodal import estimate_content_tokens
from agentloom.resilience.circuit_breaker import CircuitBreaker, CircuitState
from agentloom.resilience.rate_limiter import RateLimiter
from agentloom.resilience.retry import extract_status_code

if TYPE_CHECKING:
    from agentloom.core.attempts import Attempt

logger = logging.getLogger("agentloom.gateway")

_EMBEDDINGS = GenAIOperationName.EMBEDDINGS.value


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
        is_primary: bool = False,
        model_aliases: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.priority = priority
        self.is_fallback = is_fallback
        self.is_primary = is_primary
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name=provider.name)
        self.rate_limiter = rate_limiter
        self.models = models or []
        self.model_aliases = model_aliases or {}

    def model_id(self, requested: str) -> str:
        """The id this provider knows *requested* by."""
        return self.model_aliases.get(requested, requested)

    def call_kwargs(
        self, kwargs: dict[str, Any], requested: str, step_id: str | None
    ) -> dict[str, Any]:
        """*kwargs* plus what only the recording and replaying providers take."""
        extended = dict(kwargs)
        if getattr(self.provider, "accepts_step_id", False):
            if step_id is not None:
                extended["step_id"] = step_id
            # A recording hashes the model the step asked for, whatever this provider calls it.
            if self.model_id(requested) != requested:
                extended["requested_model"] = requested
        return extended


def _release_half_open(breaker: CircuitBreaker) -> None:
    """Give back ``allow_request()``'s half-open slot after an outcome that says nothing."""
    if breaker.state == CircuitState.HALF_OPEN and breaker._half_open_calls > 0:
        breaker._half_open_calls -= 1


class _Failures:
    """Why each candidate failed, so the final error keeps a status they all share."""

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._statuses: set[int | None] = set()
        self._last: BaseException | None = None
        self._unsupported = True
        self._waits: list[float] = []

    def add(self, message: str, exc: BaseException | None = None) -> None:
        self._messages.append(message)
        if isinstance(exc, RateLimitError) and exc.retry_after_s is not None:
            self._waits.append(exc.retry_after_s)
        self._statuses.add(None if exc is None else extract_status_code(exc))
        self._last = exc
        self._unsupported = self._unsupported and isinstance(
            exc, (UnsupportedRequestError, NotImplementedError)
        )

    def raise_all(self, summary: str) -> NoReturn:
        """Raise the gateway's error: permanent if no candidate can serve it, else by status."""
        detail = f"{summary}: " + "; ".join(self._messages)
        if self._waits:
            record_retry_after(min(self._waits))  # the soonest a candidate takes the call again
        if self._unsupported:
            raise UnsupportedRequestError("gateway", detail)
        status = next(iter(self._statuses)) if len(self._statuses) == 1 else None
        error = ProviderError("gateway", detail, status_code=status)
        # Chain only on a shared status: a permanent last failure must not block another's retries.
        raise error from (self._last if status is not None else None)


class ProviderGateway:
    """Central provider routing with fallback and resilience.

    Routes model requests to the appropriate provider. If a provider fails
    and fallbacks are configured, tries the next provider automatically.
    """

    # LRU cap for the candidate cache: templated or multi-tenant model names are unbounded.
    _DEFAULT_CANDIDATE_CACHE_MAX = 1024

    def __init__(self, candidate_cache_max: int | None = None) -> None:
        self._providers: list[ProviderEntry] = []
        self._model_mapping: dict[str, list[ProviderEntry]] = {}
        self._candidate_cache: OrderedDict[str, list[ProviderEntry]] = OrderedDict()
        self._candidate_cache_max = candidate_cache_max or int(
            os.environ.get("AGENTLOOM_CANDIDATE_CACHE_MAX", self._DEFAULT_CANDIDATE_CACHE_MAX)
        )
        self._observer: Any | None = None

    def set_observer(self, observer: Any) -> None:
        """Attach an observer to receive circuit breaker state changes."""
        self._observer = observer
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
        is_primary: bool = False,
        model_aliases: dict[str, str] | None = None,
    ) -> None:
        """Register a provider with its configuration.

        ``is_primary`` puts it ahead of the others for any model it can serve,
        whatever its ``priority``. ``model_aliases`` maps a requested model id
        to this provider's id, and makes it a candidate for the requested one.
        """
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
            is_primary=is_primary,
            model_aliases=model_aliases,
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
            call_hook(obs, "on_circuit_state_change", name, old, new)

        entry.circuit_breaker.on_state_change = _on_change

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Route a completion request to the appropriate provider with fallback.

        ``step_id`` may be passed via kwargs so the observer can scope the
        per-attempt provider spans to the originating step. Only providers
        that set ``accepts_step_id`` receive it.
        """
        step_id: str | None = kwargs.pop("step_id", None)
        candidates = self._get_candidates(model)

        if not candidates:
            raise UnsupportedRequestError(
                "gateway",
                f"No provider registered for model '{model}'",
            )

        failures = _Failures()
        cancelled = anyio.get_cancelled_exc_class()

        for attempt_idx, entry in enumerate(candidates):
            call_hook(
                self._observer, "on_circuit_state", entry.provider.name, entry.circuit_breaker.state
            )
            # Check the circuit first, so an open one costs no rate-limiter budget.
            if entry.circuit_breaker._maybe_transition_to_half_open() == CircuitState.OPEN:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                failures.add(msg)
                logger.warning(msg)
                continue

            attempt_started = False
            attempt_start = 0.0
            if step_id is not None and getattr(self._observer, "on_provider_call_start", None):
                attempt_start = anyio.current_time()
                call_hook(
                    self._observer,
                    "on_provider_call_start",
                    step_id=step_id,
                    provider=entry.provider.name,
                    model=model,
                    attempt=attempt_idx,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                attempt_started = True

            try:
                if entry.rate_limiter:
                    estimated_tokens = sum(
                        estimate_content_tokens(m.get("content", "")) for m in messages
                    )
                    await entry.rate_limiter.acquire(token_count=estimated_tokens)

                async def _call(e: ProviderEntry = entry) -> ProviderResponse:
                    return await e.provider.complete(
                        messages=messages,
                        model=e.model_id(model),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **e.call_kwargs(kwargs, model, step_id),
                    )

                response = await entry.circuit_breaker.call(
                    _call, exclude=(RateLimitError, UnsupportedRequestError, RecordingMismatchError)
                )

                if entry.rate_limiter and response.usage.completion_tokens:
                    resp_tokens = response.usage.completion_tokens
                    await entry.rate_limiter.consume_response_tokens(resp_tokens)

                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=response.model or model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        reasoning_tokens=response.usage.reasoning_tokens,
                        finish_reason=response.finish_reason,
                        stream=False,
                    )

                record_spend(
                    response.cost_usd, entry.provider.name, response.model or model, response.usage
                )
                logger.debug(
                    "Provider '%s' responded for model '%s'",
                    entry.provider.name,
                    model,
                )
                return response

            except cancelled:
                # The caller's cancel says nothing about the provider, but the span must close.
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error=cancelled.__name__,
                        stream=False,
                    )
                raise

            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                failures.add(msg)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error="CircuitOpenError",
                        stream=False,
                    )
                continue

            except Exception as e:
                msg = f"Provider '{entry.provider.name}' failed: {e}"
                failures.add(msg, e)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error=type(e).__name__,
                        stream=False,
                    )
                call_hook(
                    self._observer, "on_provider_error", entry.provider.name, type(e).__name__
                )
                if isinstance(e, RecordingMismatchError):
                    raise  # a stale recording: no other provider answers in its place
                continue

        failures.raise_all(f"All providers failed for model '{model}'")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        """Route a streaming request with fallback and resilience.

        The first chunk is pulled here, so a provider that fails before
        sending one moves on to the next candidate. Once a chunk has arrived
        the gateway commits to that provider: a connection that drops later
        raises to the caller, and the engine's retry re-runs the whole step
        (on another provider if the circuit breaker has tripped).
        """
        step_id = kwargs.pop("step_id", None)
        candidates = self._get_candidates(model)

        if not candidates:
            raise UnsupportedRequestError(
                "gateway",
                f"No provider registered for model '{model}'",
            )

        failures = _Failures()
        cancelled = anyio.get_cancelled_exc_class()

        for attempt_idx, entry in enumerate(candidates):
            call_hook(
                self._observer, "on_circuit_state", entry.provider.name, entry.circuit_breaker.state
            )
            try:
                entry.circuit_breaker.allow_request()
            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                failures.add(msg)
                logger.warning(msg)
                continue

            attempt_started = False
            attempt_start = 0.0
            if step_id is not None and getattr(self._observer, "on_provider_call_start", None):
                attempt_start = anyio.current_time()
                call_hook(
                    self._observer,
                    "on_provider_call_start",
                    step_id=step_id,
                    provider=entry.provider.name,
                    model=model,
                    attempt=attempt_idx,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                attempt_started = True

            called = False  # a failure before the provider is called says nothing about it
            try:
                if entry.rate_limiter:
                    estimated_tokens = sum(
                        estimate_content_tokens(m.get("content", "")) for m in messages
                    )
                    await entry.rate_limiter.acquire(token_count=estimated_tokens)

                opened_at = anyio.current_time()
                called = True
                inner_sr = await entry.provider.stream(
                    messages=messages,
                    model=entry.model_id(model),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **entry.call_kwargs(kwargs, model, step_id),
                )

                # Pulled here, as adapters open the request on first iteration: a failure falls back
                first_chunk: list[str] = []
                try:
                    if inner_sr._iterator is not None:
                        first_chunk.append(await anext(inner_sr._iterator))
                except StopAsyncIteration:
                    pass  # no text at all is still an answer, settled in _wrapped_iter
                except BaseException:
                    if inner_sr.usage.total_tokens:  # what came before the failure was billed
                        record_spend(
                            inner_sr.cost_usd,
                            entry.provider.name,
                            inner_sr.model or model,
                            inner_sr.usage,
                        )
                    raise

                outer_sr = StreamResponse(model=inner_sr.model, provider=inner_sr.provider)
                if first_chunk:
                    outer_sr.time_to_first_token_ms = (anyio.current_time() - opened_at) * 1000.0

                async def _wrapped_iter(
                    _inner: StreamResponse = inner_sr,
                    _entry: ProviderEntry = entry,
                    _outer: StreamResponse = outer_sr,
                    _attempt_started: bool = attempt_started,
                    _attempt_start: float = attempt_start,
                    _attempt_idx: int = attempt_idx,
                    _step_id: str | None = step_id,
                    _observer: Any = self._observer,
                    _first_chunk: list[str] = first_chunk,
                ) -> AsyncIterator[str]:
                    cancelled_exc = anyio.get_cancelled_exc_class()
                    error_type: str | None = None
                    try:
                        raw_iter = _inner._iterator
                        if raw_iter is None:
                            _entry.circuit_breaker.record_success()
                            return
                        for chunk in _first_chunk:
                            yield chunk
                        async for chunk in raw_iter:
                            yield chunk
                    except GeneratorExit:
                        # Caller's aclose(): no provider fault; success frees the half-open slot.
                        _entry.circuit_breaker.record_success()
                        raise
                    except cancelled_exc:
                        # Task-group / outer cancel scope — not a provider fault.
                        _entry.circuit_breaker.record_success()
                        error_type = cancelled_exc.__name__
                        raise
                    except RateLimitError:
                        # Throttled, not faulty: only the half-open slot goes back.
                        _release_half_open(_entry.circuit_breaker)
                        error_type = "RateLimitError"
                        raise
                    except Exception as exc:
                        _entry.circuit_breaker.record_failure()
                        error_type = type(exc).__name__
                        call_hook(_observer, "on_provider_error", _entry.provider.name, error_type)
                        raise
                    else:
                        _entry.circuit_breaker.record_success()
                        if _entry.rate_limiter and _inner.usage.completion_tokens:
                            await _entry.rate_limiter.consume_response_tokens(
                                _inner.usage.completion_tokens
                            )
                    finally:
                        # Billed however the stream ends: usage that came before a failure counts.
                        record_spend(
                            _inner.cost_usd,
                            _entry.provider.name,
                            _inner.model or model,
                            _inner.usage,
                        )
                        _outer.usage = _inner.usage
                        _outer.cost_usd = _inner.cost_usd
                        _outer.reasoning_content = _inner.reasoning_content
                        _outer.tool_calls = _inner.tool_calls
                        _outer.finish_reason = _inner.finish_reason
                        _outer.model = _inner.model
                        _outer.meta = _inner.meta
                        if _attempt_started:
                            call_hook(
                                _observer,
                                "on_provider_call_end",
                                step_id=_step_id,
                                provider=_entry.provider.name,
                                model=_inner.model,
                                latency_s=anyio.current_time() - _attempt_start,
                                attempt=_attempt_idx,
                                prompt_tokens=_inner.usage.prompt_tokens,
                                completion_tokens=_inner.usage.completion_tokens,
                                reasoning_tokens=_inner.usage.reasoning_tokens,
                                finish_reason=_inner.finish_reason,
                                error=error_type,
                                stream=True,
                            )

                outer_sr._set_iterator(_wrapped_iter())
                logger.debug(
                    "Provider '%s' streaming for model '%s'",
                    entry.provider.name,
                    model,
                )
                return outer_sr

            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                failures.add(msg)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error="CircuitOpenError",
                        stream=True,
                    )
                continue

            except RateLimitError as e:
                # Throttled, not faulty: release allow_request()'s half-open slot without a failure.
                _release_half_open(entry.circuit_breaker)
                msg = f"Provider '{entry.provider.name}' rate-limited: {e}"
                failures.add(msg, e)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error="RateLimitError",
                        stream=True,
                    )
                continue

            except cancelled:
                # The caller's cancel says nothing about the provider, but the span must close.
                _release_half_open(entry.circuit_breaker)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error=cancelled.__name__,
                        stream=True,
                    )
                raise

            except Exception as e:
                # A provider's fault feeds the breaker; otherwise allow_request()'s slot goes back.
                if called and not isinstance(e, (UnsupportedRequestError, RecordingMismatchError)):
                    entry.circuit_breaker.record_failure()
                else:
                    _release_half_open(entry.circuit_breaker)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error=type(e).__name__,
                        stream=True,
                    )
                msg = (
                    f"Provider '{entry.provider.name}' failed to start "
                    f"streaming for model '{model}': {e}"
                )
                failures.add(msg, e)
                logger.warning(msg)
                call_hook(
                    self._observer, "on_provider_error", entry.provider.name, type(e).__name__
                )
                if isinstance(e, RecordingMismatchError):
                    raise  # a stale recording: no other provider answers in its place
                continue

        failures.raise_all(f"All providers failed for model '{model}'")

    async def embed(
        self,
        inputs: list[str],
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Route an embedding request with the same fallback stack as ``complete``.

        Providers that raise ``NotImplementedError`` (Anthropic) are
        skipped cleanly — they don't count as failures against the
        circuit breaker or the retry budget.
        """
        step_id: str | None = kwargs.pop("step_id", None)
        candidates = self._get_candidates(model)
        if not candidates:
            raise UnsupportedRequestError("gateway", f"No provider registered for model '{model}'")

        failures = _Failures()
        cancelled = anyio.get_cancelled_exc_class()
        for attempt_idx, entry in enumerate(candidates):
            call_hook(
                self._observer, "on_circuit_state", entry.provider.name, entry.circuit_breaker.state
            )
            if entry.circuit_breaker._maybe_transition_to_half_open() == CircuitState.OPEN:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                failures.add(msg)
                logger.warning(msg)
                continue

            attempt_started = False
            attempt_start = 0.0
            if step_id is not None and getattr(self._observer, "on_provider_call_start", None):
                attempt_start = anyio.current_time()
                call_hook(
                    self._observer,
                    "on_provider_call_start",
                    step_id=step_id,
                    provider=entry.provider.name,
                    model=model,
                    attempt=attempt_idx,
                    temperature=None,
                    max_tokens=None,
                    stream=False,
                    operation=_EMBEDDINGS,
                )
                attempt_started = True

            try:
                if entry.rate_limiter:
                    estimated_tokens = sum(estimate_content_tokens(t) for t in inputs)
                    await entry.rate_limiter.acquire(token_count=estimated_tokens)

                call_kwargs = entry.call_kwargs(kwargs, model, step_id)

                async def _call(
                    e: ProviderEntry = entry, ck: dict[str, Any] = call_kwargs
                ) -> EmbeddingResponse:
                    result = await e.provider.embed(
                        inputs=inputs, model=e.model_id(model), dimensions=dimensions, **ck
                    )
                    # Raised in the breaker: a count mismatch is a failure, not misaligned state.
                    if len(result.embeddings) != len(inputs):
                        raise ProviderError(
                            e.provider.name,
                            (
                                f"embed returned {len(result.embeddings)} vectors "
                                f"for {len(inputs)} inputs — provider contract violation"
                            ),
                        )
                    return result

                response = await entry.circuit_breaker.call(
                    _call, exclude=(RateLimitError, NotImplementedError, RecordingMismatchError)
                )
                if entry.rate_limiter and response.usage.prompt_tokens:
                    await entry.rate_limiter.consume_response_tokens(response.usage.prompt_tokens)

                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=response.model or model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=0,
                        reasoning_tokens=0,
                        finish_reason=None,
                        stream=False,
                        operation=_EMBEDDINGS,
                    )
                record_spend(
                    response.cost_usd, entry.provider.name, response.model or model, response.usage
                )
                logger.debug(
                    "Provider '%s' returned embeddings for model '%s'", entry.provider.name, model
                )
                return response

            except cancelled:
                # The caller's cancel says nothing about the provider, but the span must close.
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error=cancelled.__name__,
                        stream=False,
                        operation=_EMBEDDINGS,
                    )
                raise

            except NotImplementedError as e:
                # A missing capability, not a fault: no breaker hit, no ``on_provider_error``.
                msg = f"Provider '{entry.provider.name}' has no embeddings: {e}"
                failures.add(msg, e)
                logger.debug(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error="NotImplementedError",
                        stream=False,
                        operation=_EMBEDDINGS,
                    )
                continue
            except CircuitOpenError:
                msg = f"Provider '{entry.provider.name}' circuit is open"
                failures.add(msg)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error="CircuitOpenError",
                        stream=False,
                        operation=_EMBEDDINGS,
                    )
                continue
            except RateLimitError as e:
                # Throttled, not faulty: no breaker failure, no ``on_provider_error``.
                msg = f"Provider '{entry.provider.name}' rate-limited: {e}"
                failures.add(msg, e)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error="RateLimitError",
                        stream=False,
                        operation=_EMBEDDINGS,
                    )
                continue
            except Exception as e:
                msg = f"Provider '{entry.provider.name}' failed: {e}"
                failures.add(msg, e)
                logger.warning(msg)
                if attempt_started:
                    call_hook(
                        self._observer,
                        "on_provider_call_end",
                        step_id=step_id,
                        provider=entry.provider.name,
                        model=model,
                        latency_s=anyio.current_time() - attempt_start,
                        attempt=attempt_idx,
                        error=type(e).__name__,
                        stream=False,
                        operation=_EMBEDDINGS,
                    )
                call_hook(
                    self._observer, "on_provider_error", entry.provider.name, type(e).__name__
                )
                if isinstance(e, RecordingMismatchError):
                    raise  # a stale recording: no other provider answers in its place
                continue

        failures.raise_all(f"All providers failed for embed on model '{model}'")

    def _get_candidates(self, model: str) -> list[ProviderEntry]:
        """Get provider candidates for a model.

        Explicit ``_model_mapping`` registrations bypass the cache; any other
        model goes through the LRU-capped ``_candidate_cache``.
        """
        if model in self._model_mapping:
            return self._ordered(model, self._model_mapping[model])

        if model in self._candidate_cache:
            self._candidate_cache.move_to_end(model)
            return self._candidate_cache[model]

        result = self._ordered(
            model, [e for e in self._providers if e.provider.supports_model(model)]
        )
        self._candidate_cache[model] = result
        if len(self._candidate_cache) > self._candidate_cache_max:
            self._candidate_cache.popitem(last=False)  # evict LRU
        return result

    def _ordered(self, model: str, serving: list[ProviderEntry]) -> list[ProviderEntry]:
        """Call order: the primary if it can serve *model*, *serving* and aliases, the fallbacks."""
        aliased = [e for e in self._providers if model in e.model_aliases and e not in serving]
        candidates = sorted([*serving, *aliased], key=lambda e: e.priority)
        fallbacks = [e for e in self._providers if e.is_fallback and e not in candidates]
        # A primary that doesn't claim the name stays out: its 404s would open its circuit.
        primary = [
            e
            for e in self._providers
            if e.is_primary and (e in candidates or e.provider.supports_model(model))
        ]
        return primary + [e for e in candidates + fallbacks if e not in primary]

    async def begin_attempt(self, step_id: str, failed: Attempt | None) -> None:
        """Pass the engine's attempt signal to the providers that record."""
        for entry in self._providers:
            begin = getattr(entry.provider, "begin_attempt", None)
            if begin is not None:
                await begin(step_id, failed)

    async def close(self) -> None:
        """Close all provider connections."""
        for entry in self._providers:
            await entry.provider.close()
