"""AgentLoom: Production-ready agentic workflow orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

if TYPE_CHECKING:
    from agentloom.core.graph import GraphEdge, GraphNode, WorkflowGraph

__all__ = ["GraphEdge", "GraphNode", "WorkflowGraph"]


def __getattr__(name: str) -> Any:
    # Resolved on first access (PEP 562): ``import agentloom.contracts`` must not load the runtime.
    if name in __all__:
        from agentloom.core import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
