"""Directed Acyclic Graph for workflow step dependencies."""

from __future__ import annotations

from collections import defaultdict

from agentloom.exceptions import ValidationError


class DAG:
    """A directed acyclic graph representing step dependencies.

    Nodes are step IDs (strings). Edges represent "depends on" relationships:
    an edge from A to B means B depends on A (A must complete before B).
    """

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: dict[str, set[str]] = defaultdict(set)  # node -> set of successors
        self._reverse: dict[str, set[str]] = defaultdict(set)  # node -> set of predecessors

    def add_node(self, node_id: str) -> None:
        """Add a node to the graph."""
        self._nodes.add(node_id)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge: to_node depends on from_node."""
        self._nodes.add(from_node)
        self._nodes.add(to_node)
        self._edges[from_node].add(to_node)
        self._reverse[to_node].add(from_node)

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    def predecessors(self, node_id: str) -> set[str]:
        """Return the set of nodes that must complete before this node."""
        return set(self._reverse.get(node_id, set()))

    def successors(self, node_id: str) -> set[str]:
        """Return the set of nodes that depend on this node."""
        return set(self._edges.get(node_id, set()))

    def validate(self) -> list[str]:
        """Validate the DAG. Returns a list of error messages (empty if valid)."""
        errors: list[str] = []

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in self._nodes}
        cycle_path: list[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            cycle_path.append(node)
            for succ in self._edges.get(node, set()):
                if color[succ] == GRAY:
                    cycle_start = cycle_path.index(succ)
                    cycle = cycle_path[cycle_start:] + [succ]
                    errors.append(f"Cycle detected: {' -> '.join(cycle)}")
                    return True
                if color[succ] == WHITE and dfs(succ):
                    return True
            cycle_path.pop()
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                dfs(node)

        for node, succs in self._edges.items():
            for succ in succs:
                if succ not in self._nodes:
                    errors.append(f"Edge to non-existent node: {node} -> {succ}")

        return errors

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order. Raises ValidationError if cyclic."""
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for succ in self._edges.get(node, set()):
                in_degree[succ] = in_degree.get(succ, 0) + 1

        # NOTE: using sorted list as a priority queue — O(n log n) per iteration.
        # Fine for typical workflow sizes (<100 nodes). For large DAGs, replace
        # with heapq or collections.deque.
        queue = [n for n in self._nodes if in_degree[n] == 0]
        queue.sort()  # Deterministic order for same-priority nodes
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for succ in sorted(self._edges.get(node, set())):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
            queue.sort()

        if len(result) != len(self._nodes):
            raise ValidationError("Workflow DAG contains a cycle")

        return result

    def execution_layers(self) -> list[list[str]]:
        """Return nodes grouped into layers for parallel execution.

        Each layer contains nodes that can execute concurrently.
        All nodes in a layer have their dependencies satisfied by previous layers.
        """
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for succ in self._edges.get(node, set()):
                in_degree[succ] = in_degree.get(succ, 0) + 1

        layers: list[list[str]] = []
        remaining = set(self._nodes)

        while remaining:
            layer = sorted(n for n in remaining if in_degree[n] == 0)
            if not layer:
                raise ValidationError("Workflow DAG contains a cycle")
            layers.append(layer)
            remaining -= set(layer)
            for node in layer:
                for succ in self._edges.get(node, set()):
                    in_degree[succ] -= 1

        return layers

    def get_ready_nodes(self, completed: set[str]) -> list[str]:
        """Return nodes whose dependencies are all in the completed set."""
        ready = []
        for node in self._nodes:
            if node in completed:
                continue
            deps = self._reverse.get(node, set())
            if deps <= completed:
                ready.append(node)
        return sorted(ready)

    @classmethod
    def from_steps(cls, steps: list[tuple[str, list[str]]]) -> DAG:
        """Build a DAG from a list of (step_id, depends_on) tuples."""
        dag = cls()
        for step_id, deps in steps:
            dag.add_node(step_id)
            for dep in deps:
                dag.add_edge(dep, step_id)
        return dag
