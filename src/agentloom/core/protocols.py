"""Structural types for engine-level collaborators.

These Protocols describe the ``Any`` seams in :class:`StepContext` (each field
names its protocol) and keep the concrete implementations (``StateManager``,
``ProviderGateway``, ``ToolRegistry``, ``WorkflowObserver``, ``BaseCheckpointer``)
decoupled from step executors. They only describe shape and import nothing of
the package at runtime, so importing them causes no cycle. ``call_hook`` is the
one way the engine and the gateway reach an observer.

Only the methods that ``core/`` and ``steps/`` actually call on these
collaborators are included. A concrete implementation can add more; a
consumer that needs something not listed here should grow the protocol,
not cast back to ``Any``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentloom.core.results import StepResult

logger = logging.getLogger("agentloom.observability")


def call_hook(observer: object, hook: str, /, *args: object, **kwargs: object) -> None:
    """Call *observer*'s *hook*, if it has one; an exception is logged, never raised.

    Telemetry must not decide a run: a raising hook inside a step's attempt would retry its
    paid calls.
    """
    method = getattr(observer, hook, None)
    if method is None:
        return
    try:
        method(*args, **kwargs)
    except Exception:
        logger.warning("Observer hook '%s' failed", hook, exc_info=True)


@runtime_checkable
class StateManagerProtocol(Protocol):
    """The subset of :class:`StateManager` that step executors rely on."""

    async def get(self, key: str, default: Any = None) -> Any: ...
    async def set(self, key: str, value: Any) -> None: ...
    async def get_state_snapshot(self) -> dict[str, Any]: ...
    async def get_step_result(self, step_id: str) -> StepResult | None: ...
    async def set_step_result(self, step_id: str, result: StepResult) -> None: ...


@runtime_checkable
class GatewayProtocol(Protocol):
    """What a step expects from a provider gateway.

    Both ``complete`` and ``stream`` are kept intentionally narrow —
    concrete gateways accept more kwargs, but step code only forwards
    these.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any: ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """What step executors need from the tool registry — ``get`` only."""

    def get(self, name: str) -> Any: ...


@runtime_checkable
class ObserverProtocol(Protocol):
    """Workflow observer surface consumed by the engine.

    The engine calls these through :func:`call_hook`, so one an observer lacks
    is skipped (:class:`NoopObserver` has them all, as no-ops). The protocol
    documents the contract so observer authors can accept extra kwargs
    defensively.
    """

    def on_workflow_start(self, workflow_name: str, **kwargs: Any) -> None: ...

    def on_workflow_end(
        self,
        workflow_name: str,
        status: str,
        duration_ms: float,
        total_tokens: int,
        total_cost: float,
        **kwargs: Any,
    ) -> None: ...

    def on_step_start(
        self, step_id: str, step_type: str, stream: bool = False, **kwargs: Any
    ) -> None: ...

    def on_step_end(
        self,
        step_id: str,
        step_type: str,
        status: str,
        duration_ms: float,
        cost_usd: float = 0.0,
        **kwargs: Any,
    ) -> None: ...

    # Provider hooks too, so ``isinstance`` flags an observer that would miss token events

    def on_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        **kwargs: Any,
    ) -> None: ...

    def on_stream_response(
        self,
        provider: str,
        model: str,
        ttft_s: float,
        **kwargs: Any,
    ) -> None: ...


@runtime_checkable
class CheckpointerProtocol(Protocol):
    """Minimal checkpointer surface used by the engine."""

    async def save(self, data: Any) -> None: ...
    async def load(self, run_id: str) -> Any: ...


@runtime_checkable
class StreamCallbackProtocol(Protocol):
    """Signature of the per-chunk streaming callback."""

    def __call__(self, step_id: str, chunk: str) -> None: ...


__all__ = [
    "call_hook",
    "CheckpointerProtocol",
    "GatewayProtocol",
    "ObserverProtocol",
    "StateManagerProtocol",
    "StreamCallbackProtocol",
    "ToolRegistryProtocol",
]
