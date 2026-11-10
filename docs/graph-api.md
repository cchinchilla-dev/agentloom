---
title: "Graph API"
description: "Read the DAG of an AgentLoom workflow from Python: its layers and paths, and exports as Graphviz DOT, Mermaid, PNML, JSON or a NetworkX graph."
order: 14
section: "Running"
---

# Graph API

`WorkflowGraph` reads the DAG of a workflow: its steps and dependencies, the layers they run in and the paths through them. It exports the graph as Graphviz DOT, Mermaid, PNML, a dictionary or a NetworkX graph.

## Getting a graph

```python
from agentloom import WorkflowGraph
from agentloom.core.parser import WorkflowParser

workflow = WorkflowParser.from_yaml("examples/03_router_workflow.yaml")
graph = WorkflowGraph.from_workflow(workflow)

graph.layers          # [['classify'], ['route'], ['handle_billing', 'handle_general', 'handle_technical']]
graph.critical_path() # ['classify', 'route', 'handle_billing']
print(graph.to_mermaid())
```

- `WorkflowGraph.from_workflow(workflow)` takes a parsed workflow or one built with the [Python DSL](python-dsl.md). It refuses a cycle with a `ValidationError`, but doesn't check that dependencies exist: validate a DSL build first, with `WorkflowParser.from_dict(workflow.model_dump())`.
- `WorkflowGraph.from_dag(dag)` takes a bare `DAG` from `agentloom.core.dag`. Its nodes have no step behind them: each is typed `llm_call`, and `get_step_definition` returns `None`.

`WorkflowGraph`, `GraphNode` and `GraphEdge` import from `agentloom` itself, and need nothing beyond the core package, except `to_networkx`.

## Nodes and edges

| Model | Field | What it is |
|-------|-------|------------|
| `GraphNode` | `id` | The step's id. |
| | `type` | The step type: `llm_call`, `tool`, `router`, `subworkflow`, `approval_gate`, `embed` or `agent`. |
| | `depends_on` | The step's dependencies. |
| | `label` | The step's id. |
| `GraphEdge` | `source`, `target` | An edge from a step to one that depends on it. |
| | `label` | On a router's edges, the condition that leads to the target, or `default`. Empty elsewhere. |

- **Edges come from `depends_on` only.** A router's conditions label the edges to its targets; a target that doesn't list the router in `depends_on` gets no edge from it.
- **A router's edge has one label.** When several conditions lead to one target, the last one names the edge, and `default` wins over all of them.
- **Every branch is a path,** though a run takes only one of a router's branches.
- **A subworkflow is one node.** Its own steps aren't expanded.

Both models are immutable.

## Structure

| Property | What it holds |
|----------|---------------|
| `nodes` | The nodes, in the order they can run; steps that could run together are sorted by id. |
| `edges` | The edges, sorted by source and target. |
| `roots` | The steps that depend on nothing. |
| `leaves` | The steps nothing depends on. |
| `layers` | The steps grouped in the layers the engine runs them in. See [Dependencies and layers](concepts.md#dependencies-and-layers). |

## Paths

- **`critical_path()`** is the longest chain of steps, counted in steps. It knows nothing of durations or costs, so it isn't a latency estimate. Between chains of the same length, the step ids decide which one it returns, not how likely a branch is: in example 03, all three branches have three steps, and it returns the billing one.
- **`all_paths()`** lists every chain from a root to a leaf, each of a router's branches included.
- **`prime_paths(max_paths=10_000)`** lists the paths that no longer path contains. In a DAG, these are the same as `all_paths()`. `max_paths` caps the paths the search keeps along the way, which start at one per step, and a graph that needs more raises a `ValueError`.
- **`get_step_definition(node_id)`** returns the step behind a node, or `None` for an unknown id.

## Exports

<!-- tabs -->

**DOT**

```python
print(graph.to_dot())
```

```text
digraph workflow {
    rankdir=LR;
    "classify" [label="classify", shape=box, style=rounded];
    "route" [label="route", shape=diamond];
    "handle_billing" [label="handle_billing", shape=box, style=rounded];
    "handle_general" [label="handle_general", shape=box, style=rounded];
    "handle_technical" [label="handle_technical", shape=box, style=rounded];
    "classify" -> "route";
    "route" -> "handle_billing" [label="state.classification.lower() == 'billing'"];
    "route" -> "handle_general" [label="default"];
    "route" -> "handle_technical" [label="state.classification.lower() == 'technical'"];
}
```

**Mermaid**

```python
print(graph.to_mermaid())
```

```text
graph TD
    classify["classify"]
    route{route}
    handle_billing["handle_billing"]
    handle_general["handle_general"]
    handle_technical["handle_technical"]
    classify --> route
    route -->|state.classification.lower() == 'billing'| handle_billing
    route -->|default| handle_general
    route -->|state.classification.lower() == 'technical'| handle_technical
```

**Dictionary**

```python
import json
json.dumps(graph.to_dict())
```

`to_dict()` returns `nodes`, `edges`, `roots`, `leaves`, `layers` and `critical_path`, ready for `json.dumps`.

<!-- /tabs -->

| Export | What it gives |
|--------|---------------|
| `to_dot()` | Graphviz DOT, left to right. Routers are diamonds, tools trapeziums, subworkflows double octagons, LLM calls rounded boxes, and other steps plain boxes. Render it with `dot -Tsvg`. |
| `to_mermaid()` | A Mermaid flowchart, without a code fence. Routers are diamonds, tools parallelograms and subworkflows subroutine boxes. Ids keep only letters, digits and `_`. Steps that would end up with the same id get a numbered one: `a-b` and `a_b` become `a_b` and `a_b_2`. |
| `to_pnml()` | A Petri net in PNML, as XML: a place for each step and a transition for each edge, named `source->target`. Edge labels are left out. |
| `to_dict()` | A dictionary of nodes, edges, roots, leaves, layers and the critical path. |
| `to_networkx()` | A `networkx.DiGraph`, with each node's `type` and `label` and each edge's `label` as attributes. |

`to_networkx()` needs the `graph` extra:

```bash
pip install "agentloom[graph]"
```

Without it, the call raises an `ImportError` that names the extra.

## The visualize command

`agentloom visualize` draws a workflow as text or as Mermaid without this API. Its Mermaid differs: steps keep the order of the file, ids aren't changed, and edge labels are cut to 20 characters, so two conditions that start alike can read the same. For DOT, PNML, path analysis or exact labels, use `WorkflowGraph`. See [CLI](cli.md#visualize).

## What's next

<!-- cards -->

- **[Python DSL](python-dsl.md)** — Build and run workflows from Python.
- **[Concepts](concepts.md)** — Steps, dependencies and the layers they run in.
- **[CLI](cli.md)** — `validate` and `visualize` from the command line.

<!-- /cards -->
