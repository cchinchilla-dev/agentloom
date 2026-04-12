# Graph API

Programmatic access to the workflow DAG for analysis, visualization, and test coverage.

## Quick start

```python
from agentloom import WorkflowGraph
from agentloom.core.parser import WorkflowParser

workflow = WorkflowParser.from_yaml("examples/03_router_workflow.yaml")
graph = WorkflowGraph.from_workflow(workflow)
```

You can also build a graph from a bare `DAG` object (without a full workflow definition):

```python
from agentloom.core.dag import DAG
from agentloom.core.graph import WorkflowGraph

dag = DAG()
dag.add_node("a")
dag.add_node("b")
dag.add_edge("a", "b")
graph = WorkflowGraph.from_dag(dag)
```

## Data models

### GraphNode

Frozen Pydantic model representing a step in the graph:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Step identifier |
| `type` | `str` | Step type (`llm_call`, `tool`, `router`, `subworkflow`) |
| `depends_on` | `list[str]` | IDs of predecessor steps |
| `label` | `str` | Display label |

### GraphEdge

Frozen Pydantic model representing a dependency edge:

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Source step ID |
| `target` | `str` | Target step ID |
| `label` | `str` | Edge label (e.g., router condition) |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `graph.nodes` | `list[GraphNode]` | Nodes in topological order |
| `graph.edges` | `list[GraphEdge]` | Edges sorted by (source, target) |
| `graph.roots` | `list[str]` | Entry-point step IDs (no predecessors) |
| `graph.leaves` | `list[str]` | Terminal step IDs (no successors) |
| `graph.layers` | `list[list[str]]` | Steps grouped into parallel-execution layers |

## Analysis methods

### `critical_path()`

Returns the longest path by hop count — the latency bottleneck:

```python
graph.critical_path()
# ["fetch_data", "analyze", "summarize", "report"]
```

### `all_paths()`

Every simple path from root to leaf:

```python
for path in graph.all_paths():
    print(" -> ".join(path))
# fetch_data -> analyze -> report
# fetch_data -> analyze -> summarize -> report
```

### `prime_paths()`

Maximal simple paths for test coverage. These are paths that cannot be extended without repeating a node:

```python
graph.prime_paths(max_paths=10000)
```

### `get_step_definition(node_id)`

Retrieve the original step config for a node:

```python
step = graph.get_step_definition("classify")
# StepDefinition(id="classify", type="llm_call", ...)
```

## Export formats

=== "Graphviz DOT"

    ```python
    dot = graph.to_dot()
    with open("workflow.dot", "w") as f:
        f.write(dot)
    # Then: dot -Tpng workflow.dot -o workflow.png
    ```

    Node shapes by step type: rounded box (LLM), trapezium (tool), diamond (router), doubleoctagon (subworkflow).

=== "Mermaid"

    ```python
    mermaid = graph.to_mermaid()
    # Paste into https://mermaid.live or use in docs
    ```

    Generates `graph TD` format with edge labels for router conditions.

=== "PNML"

    ```python
    pnml = graph.to_pnml()
    # Petri Net Markup Language (places + transitions)
    ```

=== "Dict (JSON)"

    ```python
    data = graph.to_dict()
    # {"nodes": [...], "edges": [...], "roots": [...], "leaves": [...],
    #  "layers": [...], "critical_path": [...]}
    ```

=== "NetworkX"

    ```bash
    pip install agentloom[graph]
    ```

    ```python
    nx_graph = graph.to_networkx()  # networkx.DiGraph
    # Nodes have "type" and "label" attributes
    # Edges have "label" attribute
    ```
