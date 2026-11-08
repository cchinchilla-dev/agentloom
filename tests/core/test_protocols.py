"""Structural checks that the engine's concrete collaborators satisfy its protocols."""

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

    ``isinstance`` on a ``runtime_checkable`` Protocol checks that the members exist, not their
    signatures: a failure here means a member was removed or renamed.
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

        # The Protocol's only member is ``__call__``, which any function has.
        assert isinstance(cb, StreamCallbackProtocol)


class TestCheckpointerProtocolAllowsNone:
    """The checkpointer is optional, but a concrete one must still fit the Protocol."""

    def test_file_checkpointer_conforms(self, tmp_path) -> None:
        from agentloom.checkpointing.file import FileCheckpointer

        assert isinstance(FileCheckpointer(checkpoint_dir=tmp_path), CheckpointerProtocol)
