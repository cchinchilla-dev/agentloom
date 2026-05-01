"""Structural checks for the protocols module.

Exists so a reader can confirm at a glance which collaborators the engine
expects, and so renames on the concrete implementations surface here.
"""

from __future__ import annotations

from agentloom.core.protocols import (
    CheckpointerProtocol,
    GatewayProtocol,
    ObserverProtocol,
    StateManagerProtocol,
    StreamCallbackProtocol,
    ToolRegistryProtocol,
)
from agentloom.core.state import StateManager
from agentloom.observability.noop import NoopObserver
from agentloom.providers.gateway import ProviderGateway
from agentloom.tools.registry import ToolRegistry


class TestProtocolConformance:
    """Concrete implementations structurally satisfy the Protocols.

    ``runtime_checkable`` lets us use ``isinstance`` as a structural
    smoke test — it verifies that the required attributes exist and that
    callables are callable. Parameter and return-type signatures are
    **not** validated at runtime, so a failure here means an attribute
    or method was removed or renamed, not that signatures drifted.
    """

    def test_state_manager_conforms(self) -> None:
        assert isinstance(StateManager(), StateManagerProtocol)

    def test_gateway_conforms(self) -> None:
        assert isinstance(ProviderGateway(), GatewayProtocol)

    def test_noop_observer_conforms(self) -> None:
        assert isinstance(NoopObserver(), ObserverProtocol)

    def test_tool_registry_conforms(self) -> None:
        assert isinstance(ToolRegistry(), ToolRegistryProtocol)


class TestStreamCallback:
    def test_plain_callable_satisfies_protocol(self) -> None:
        def cb(step_id: str, chunk: str) -> None:
            return None

        # runtime_checkable Protocols only check attribute presence for
        # non-Callable structural types — Callable-Protocols rely on
        # ``__call__``, which any function has.
        assert isinstance(cb, StreamCallbackProtocol)


class TestCheckpointerProtocolAllowsNone:
    """Checkpointer is optional; the Protocol still enforces shape for
    concrete implementations."""

    def test_file_checkpointer_conforms(self, tmp_path) -> None:
        from agentloom.checkpointing.file import FileCheckpointer

        assert isinstance(FileCheckpointer(checkpoint_dir=tmp_path), CheckpointerProtocol)
