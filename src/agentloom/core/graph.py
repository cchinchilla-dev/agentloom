"""First-class graph API for workflow DAG analysis and export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import ModuleType
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentloom.compat import is_available, try_import
from agentloom.core.dag import DAG
from agentloom.core.models import StepType

if TYPE_CHECKING:
    from agentloom.core.models import StepDefinition, WorkflowDefinition


class GraphNode(BaseModel):
    """Immutable representation of a single node in the workflow graph."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: StepType
    depends_on: list[str] = Field(default_factory=list)
    label: str = ""

    @model_validator(mode="before")
    @classmethod
    def _default_label(cls, values: object) -> object:
        if isinstance(values, dict) and not values.get("label"):
            values["label"] = values.get("id", "")
        return values


class GraphEdge(BaseModel):
    """Immutable representation of a directed edge in the workflow graph."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    label: str = ""


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

    def get_step_definition(self, node_id: str) -> StepDefinition | None:
        """Return the :class:`~agentloom.core.models.StepDefinition` for *node_id*.

        Returns ``None`` if this graph was built from a bare DAG (no workflow
        reference) or if the step ID is not found.
        """
        if self._workflow is None:
            return None
        return self._workflow.get_step(node_id)

    def all_paths(self) -> list[list[str]]:
        """Return all simple paths from every root to every leaf.

        The result is deterministically sorted.
        """
        leaves = set(self.leaves)
        result: list[list[str]] = []

        def dfs(node: str, current: list[str]) -> None:
            current.append(node)
            if node in leaves:
                result.append(list(current))
            else:
                for successor in sorted(self._dag.successors(node)):
                    dfs(successor, current)
            current.pop()

        for root in self.roots:
            dfs(root, [])

        return sorted(result)

    _MAX_PRIME_PATHS = 10_000

    def prime_paths(self, *, max_paths: int = _MAX_PRIME_PATHS) -> list[list[str]]:
        """Return all prime (maximal simple) paths in the graph.

        A prime path is a maximal simple path: it cannot be extended at either
        end without repeating a node, and it is not a contiguous sub-sequence
        of any longer path in the result set.

        Args:
            max_paths: Safety limit on intermediate path count.  Raises
                :class:`ValueError` if exceeded (default 10 000).

        The result is deterministically sorted.
        """
        # Start with all single-node paths.
        paths: list[list[str]] = [[node_id] for node_id in sorted(self._dag.nodes)]

        # Iteratively extend each path.
        changed = True
        while changed:
            changed = False
            new_paths: list[list[str]] = []
            for path in paths:
                last = path[-1]
                successors = sorted(self._dag.successors(last))
                extended = False
                for succ in successors:
                    if succ not in path:  # keep it simple (acyclic)
                        new_paths.append(path + [succ])
                        extended = True
                        changed = True
                if not extended:
                    new_paths.append(path)
            paths = new_paths
            if len(paths) > max_paths:
                raise ValueError(
                    f"Prime path enumeration exceeded {max_paths} intermediate paths. "
                    f"Graph is too complex; pass a higher max_paths to override."
                )

        # De-duplicate.
        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []
        for path in paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)

        # Filter: remove any path that is a contiguous sub-sequence of another.
        def is_subpath(candidate: list[str], other: list[str]) -> bool:
            if len(candidate) >= len(other):
                return False
            n = len(candidate)
            return any(other[i : i + n] == candidate for i in range(len(other) - n + 1))

        prime: list[list[str]] = []
        for i, path in enumerate(unique):
            dominated = any(is_subpath(path, other) for j, other in enumerate(unique) if i != j)
            if not dominated:
                prime.append(path)

        return sorted(prime)

    def critical_path(self) -> list[str]:
        """Return the longest path through the graph (by hop count).

        Uses topological-order dynamic programming.  Returns an empty list
        for an empty graph.
        """
        topo = self._dag.topological_sort()
        if not topo:
            return []

        dist: dict[str, int] = {}
        prev: dict[str, str | None] = {}

        for node in topo:
            preds = self._dag.predecessors(node)
            if not preds:
                dist[node] = 1
                prev[node] = None
            else:
                best_pred = max(sorted(preds), key=lambda p: (dist.get(p, 0), p))
                dist[node] = dist.get(best_pred, 0) + 1
                prev[node] = best_pred

        # Node with the greatest distance is the end of the critical path.
        end = max(topo, key=lambda n: dist.get(n, 0))

        path: list[str] = []
        current: str | None = end
        while current is not None:
            path.append(current)
            current = prev.get(current)

        path.reverse()
        return path

    def to_dict(self) -> dict[str, object]:
        """Serialise the graph to a plain dictionary.

        Includes node/edge lists, root/leaf sets, execution layers, and the
        critical path.
        """
        return {
            "nodes": [node.model_dump() for node in self.nodes],
            "edges": [edge.model_dump() for edge in self.edges],
            "roots": self.roots,
            "leaves": self.leaves,
            "layers": self.layers,
            "critical_path": self.critical_path(),
        }

    def to_dot(self) -> str:
        """Render the graph as a Graphviz DOT string."""
        _SHAPES: dict[StepType, str] = {
            StepType.LLM_CALL: "box, style=rounded",
            StepType.TOOL: "trapezium",
            StepType.ROUTER: "diamond",
            StepType.SUBWORKFLOW: "doubleoctagon",
        }

        def _dot_escape(text: str) -> str:
            return text.replace("\\", "\\\\").replace('"', '\\"')

        lines: list[str] = ["digraph workflow {", "    rankdir=LR;"]

        for node in self.nodes:
            shape = _SHAPES.get(node.type, "box")
            lines.append(
                f'    "{_dot_escape(node.id)}" [label="{_dot_escape(node.label)}", shape={shape}];'
            )

        for edge in self.edges:
            src = _dot_escape(edge.source)
            tgt = _dot_escape(edge.target)
            if edge.label:
                lines.append(f'    "{src}" -> "{tgt}" [label="{_dot_escape(edge.label)}"];')
            else:
                lines.append(f'    "{src}" -> "{tgt}";')

        lines.append("}")
        return "\n".join(lines)

    def to_pnml(self) -> str:
        """Render the graph as a PNML (Petri Net Markup Language) string."""
        root_el = ET.Element("pnml")
        net = ET.SubElement(
            root_el,
            "net",
            attrib={
                "id": "workflow",
                "type": "http://www.pnml.org/version-2009/grammar/pnmlcoremodel",
            },
        )

        # One place per node (prefixed to avoid collisions with transition IDs).
        for node in self.nodes:
            place_id = f"p_{node.id}"
            place = ET.SubElement(net, "place", attrib={"id": place_id})
            name_el = ET.SubElement(place, "name")
            ET.SubElement(name_el, "text").text = node.id

        # One transition + two arcs per edge (index-based IDs to avoid collisions).
        for idx, edge in enumerate(self.edges):
            t_id = f"t{idx}"
            label_text = f"{edge.source}->{edge.target}"
            transition = ET.SubElement(net, "transition", attrib={"id": t_id})
            name_el = ET.SubElement(transition, "name")
            ET.SubElement(name_el, "text").text = label_text

            ET.SubElement(
                net,
                "arc",
                attrib={"id": f"a{idx}_in", "source": f"p_{edge.source}", "target": t_id},
            )
            ET.SubElement(
                net,
                "arc",
                attrib={"id": f"a{idx}_out", "source": t_id, "target": f"p_{edge.target}"},
            )

        ET.indent(root_el, space="  ")
        return "<?xml version='1.0' encoding='UTF-8'?>\n" + ET.tostring(root_el, encoding="unicode")

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid flowchart string."""

        def _mermaid_id(text: str) -> str:
            """Sanitise a node ID for Mermaid (alphanumeric + underscore only)."""
            return "".join(c if c.isalnum() or c == "_" else "_" for c in text)

        def _mermaid_label(text: str) -> str:
            """Escape special characters in a Mermaid label."""
            return (
                text.replace('"', "#quot;")
                .replace("|", "#vert;")
                .replace("[", "#lsqb;")
                .replace("]", "#rsqb;")
                .replace("{", "#lbrace;")
                .replace("}", "#rbrace;")
            )

        lines: list[str] = ["graph TD"]

        # Node declarations with shape based on type.
        for node in self.nodes:
            nid = _mermaid_id(node.id)
            lbl = _mermaid_label(node.label)
            if node.type == StepType.ROUTER:
                lines.append(f"    {nid}{{{lbl}}}")
            elif node.type == StepType.TOOL:
                lines.append(f"    {nid}[/{lbl}/]")
            elif node.type == StepType.SUBWORKFLOW:
                lines.append(f"    {nid}[[{lbl}]]")
            else:
                lines.append(f'    {nid}["{lbl}"]')

        # Edges.
        for edge in self.edges:
            src = _mermaid_id(edge.source)
            tgt = _mermaid_id(edge.target)
            if edge.label:
                safe_lbl = _mermaid_label(edge.label)
                lines.append(f"    {src} -->|{safe_lbl}| {tgt}")
            else:
                lines.append(f"    {src} --> {tgt}")

        return "\n".join(lines)

    def to_networkx(self) -> object:
        """Return a ``networkx.DiGraph`` representation of this workflow graph.

        Requires the ``graph`` optional extra::

            pip install agentloom[graph]

        Node attributes: ``type`` (:class:`str`), ``label`` (:class:`str`).
        Edge attributes: ``label`` (:class:`str`).

        Raises:
            ImportError: If *networkx* is not installed.
        """
        nx_or_proxy = try_import("networkx", extra="graph")
        if not is_available(nx_or_proxy):
            # nx_or_proxy is a MissingDependencyProxy; delegate to its _raise().
            getattr(nx_or_proxy, "_raise")()
            return None  # pragma: no cover

        # Narrow to ModuleType so mypy is satisfied for attribute access.
        nx: ModuleType = nx_or_proxy  # type: ignore[assignment]
        graph = nx.DiGraph()

        for node in self.nodes:
            graph.add_node(node.id, type=str(node.type), label=node.label)

        for edge in self.edges:
            graph.add_edge(edge.source, edge.target, label=edge.label)

        return graph
