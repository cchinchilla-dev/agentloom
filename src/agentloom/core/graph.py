"""First-class graph API for workflow DAG analysis and export."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentloom.core.dag import DAG
from agentloom.core.models import StepType

if TYPE_CHECKING:
    from agentloom.core.models import StepDefinition, WorkflowDefinition


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """Immutable representation of a single node in the workflow graph."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: StepType
    depends_on: list[str] = Field(default_factory=list)
    label: str = ""

    @model_validator(mode="before")
    @classmethod
    def _default_label(cls, values: dict[str, object]) -> dict[str, object]:
        if not values.get("label"):
            values["label"] = values.get("id", "")
        return values


class GraphEdge(BaseModel):
    """Immutable representation of a directed edge in the workflow graph."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    label: str = ""


# ---------------------------------------------------------------------------
# WorkflowGraph
# ---------------------------------------------------------------------------


class WorkflowGraph:
    """Immutable first-class graph view of a workflow DAG.

    Wraps :class:`~agentloom.core.dag.DAG` and exposes path algorithms,
    layered analysis, and multiple export formats (DOT, PNML, Mermaid,
    NetworkX).

    Construct via :meth:`from_workflow` or :meth:`from_dag`; do not call
    ``__init__`` directly.
    """

    def __init__(
        self,
        dag: DAG,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        workflow: WorkflowDefinition | None = None,
    ) -> None:
        self._dag = dag
        self._nodes = nodes
        self._edges = edges
        self._workflow = workflow

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_workflow(cls, workflow: WorkflowDefinition) -> WorkflowGraph:
        """Build a :class:`WorkflowGraph` from a parsed workflow definition.

        Router step edges are labelled with their condition expressions; the
        default-target edge is labelled ``"default"``.  All other edges have
        an empty label.

        Args:
            workflow: Validated :class:`~agentloom.core.models.WorkflowDefinition`.

        Returns:
            A fully constructed, immutable :class:`WorkflowGraph`.
        """
        from agentloom.core.parser import WorkflowParser

        dag = WorkflowParser.build_dag(workflow)

        # Build node map: step_id -> GraphNode
        nodes: list[GraphNode] = [
            GraphNode(
                id=step.id,
                type=step.type,
                depends_on=list(step.depends_on),
                label=step.id,
            )
            for step in workflow.steps
        ]

        # Build edges with labels derived from router conditions
        edges: list[GraphEdge] = []
        router_targets: dict[str, dict[str, str]] = {}  # step_id -> {target -> label}

        for step in workflow.steps:
            if step.type == StepType.ROUTER:
                target_labels: dict[str, str] = {}
                for condition in step.conditions:
                    target_labels[condition.target] = condition.expression
                if step.default is not None:
                    target_labels[step.default] = "default"
                router_targets[step.id] = target_labels

        for node_id in dag.nodes:
            for successor in sorted(dag.successors(node_id)):
                label = ""
                if node_id in router_targets:
                    label = router_targets[node_id].get(successor, "")
                edges.append(GraphEdge(source=node_id, target=successor, label=label))

        edges.sort(key=lambda e: (e.source, e.target))
        return cls(dag=dag, nodes=nodes, edges=edges, workflow=workflow)

    @classmethod
    def from_dag(cls, dag: DAG) -> WorkflowGraph:
        """Build a :class:`WorkflowGraph` directly from a :class:`~agentloom.core.dag.DAG`.

        All nodes are given type :attr:`~agentloom.core.models.StepType.LLM_CALL`.
        No workflow reference is stored; :meth:`get_step_definition` always
        returns ``None``.

        Args:
            dag: A pre-built :class:`~agentloom.core.dag.DAG`.

        Returns:
            A fully constructed, immutable :class:`WorkflowGraph`.
        """
        nodes: list[GraphNode] = [
            GraphNode(id=node_id, type=StepType.LLM_CALL, label=node_id)
            for node_id in sorted(dag.nodes)
        ]

        edges: list[GraphEdge] = []
        for node_id in sorted(dag.nodes):
            for successor in sorted(dag.successors(node_id)):
                edges.append(GraphEdge(source=node_id, target=successor, label=""))

        edges.sort(key=lambda e: (e.source, e.target))
        return cls(dag=dag, nodes=nodes, edges=edges, workflow=None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[GraphNode]:
        """Nodes in topological order."""
        topo = self._dag.topological_sort()
        node_map = {n.id: n for n in self._nodes}
        return [node_map[nid] for nid in topo if nid in node_map]

    @property
    def edges(self) -> list[GraphEdge]:
        """Edges sorted by (source, target)."""
        return list(self._edges)

    @property
    def roots(self) -> list[str]:
        """Sorted node IDs with no predecessors (entry points)."""
        return sorted(node.id for node in self._nodes if not self._dag.predecessors(node.id))

    @property
    def leaves(self) -> list[str]:
        """Sorted node IDs with no successors (terminal steps)."""
        return sorted(node.id for node in self._nodes if not self._dag.successors(node.id))

    @property
    def layers(self) -> list[list[str]]:
        """Nodes grouped into parallel-execution layers."""
        return self._dag.execution_layers()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_step_definition(self, node_id: str) -> StepDefinition | None:
        """Return the :class:`~agentloom.core.models.StepDefinition` for *node_id*.

        Returns ``None`` if this graph was built from a bare DAG (no workflow
        reference) or if the step ID is not found.
        """
        if self._workflow is None:
            return None
        return self._workflow.get_step(node_id)
