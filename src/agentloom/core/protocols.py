"""Structural types for engine-level collaborators.

These Protocols replace the ``Any`` seams in :class:`StepContext` and keep
the concrete implementations (``StateManager``, ``ProviderGateway``,
``ToolRegistry``, ``WorkflowObserver``, ``BaseCheckpointer``) decoupled
from step executors. Import sites use ``if TYPE_CHECKING`` so we avoid
runtime cycles — the Protocols themselves only describe shape.

Only the methods that ``core/`` and ``steps/`` actually call on these
collaborators are included. A concrete implementation can add more; a
consumer that needs something not listed here should grow the protocol,
not cast back to ``Any``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentloom.core.results import StepResult


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

    Every method is optional in practice — :class:`NoopObserver` implements
    all of them as no-ops and the engine relies on duck-typing. The
    protocol documents the contract so observer authors can add kwargs
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
        tokens: int = 0,
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
    "CheckpointerProtocol",
    "GatewayProtocol",
    "ObserverProtocol",
    "StateManagerProtocol",
    "StreamCallbackProtocol",
    "ToolRegistryProtocol",
]
