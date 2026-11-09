---
title: "Architecture"
description: "How the AgentLoom engine turns a workflow into layers, runs them, manages state, routes calls through the gateway and finishes a run."
order: 18
section: "Operations"
---

# Architecture

How the engine runs a workflow: the modules it is made of, the path of a run from the YAML file to its result, and which parts are public. [Concepts](concepts.md) introduces the ideas this page builds on.

## Modules

| Module | What it does |
|--------|--------------|
| `cli/` | The commands. Each one parses the workflow, sets up the gateway, the observer, the tools and the checkpointer, and runs the engine. |
| `core/models.py`, `core/parser.py` | The workflow's Pydantic models, and loading YAML or a dictionary into them, with the reference, cycle and output checks. |
| `core/dag.py` | The dependency graph: cycles, layers, topological order, the steps after a step. |
| `core/engine.py` | `WorkflowEngine`: layers, concurrency, attempts, timeouts, the budget, pauses, routers, failures, the checkpoint and the history record. |
| `core/state.py`, `core/templates.py`, `core/redact.py` | The state, rendering prompts from it, and masking redacted keys. |
| `core/results.py` | `StepResult`, `WorkflowResult` and their statuses. |
| `core/protocols.py` | The interfaces the engine uses for its collaborators. |
| `steps/` | One executor per step type, found through `StepRegistry`. |
| `providers/` | `ProviderGateway`, the provider adapters, the mock, recording and replay providers, attachments and prices. |
| `resilience/` | Circuit breakers, rate limiters, retry waits and the budget. |
| `tools/` | `ToolRegistry`, the `@tool` decorator, the built-in tools and `ToolSandbox`, which also checks webhooks and attachments. |
| `checkpointing/`, `history/` | Checkpoint files, and run history records. |
| `observability/` | The observer, OpenTelemetry tracing and metrics, and their no-op stand-ins. |
| `webhooks/` | Approval gate notifications. |
| `contracts/` | The public import surface. See [Public contracts](#public-contracts). |

## The path of a run

What `agentloom run workflow.yaml` does:

1. **Parse.** The YAML becomes a `WorkflowDefinition`; its references (`depends_on`, router targets, subworkflow files) are checked, then cycles, then steps that write the same `output`.
2. **Set up.** The command applies its options to `config`, creates one `ProviderGateway`, registers the providers the environment sets up, or the mock, and adds an observer if the `observability` extra is installed, the built-in tools behind the sandbox, and a checkpointer with `--checkpoint`.
3. **Layer.** The engine builds the graph again and sorts the steps into layers.
4. **Run each layer.** Its steps start together, up to `max_concurrent_steps`, under the run's `timeout`.
5. **Run each step.** The budget is checked, then each attempt calls the step's executor under the step's `timeout`. The executor takes a snapshot of the state, renders its templates and calls the gateway or its tool.
6. **Call a provider.** The gateway tries the providers that serve the model in order, skipping those whose circuit is open, waiting for each one's rate limit, and records what the call cost.
7. **Record.** A step that succeeds writes its `output` to the state. The engine stores its result, charges its cost to the budget and tells the observer.
8. **Between layers,** the engine stops for a pause, skips the steps after a failure, and notes which branches the routers chose.
9. **Finish.** The engine builds the `WorkflowResult`, saves the checkpoint, masks redacted keys, writes the history record and tells the observer. The command prints the result and closes the gateway.

## Scheduling

- **Layers are levels of the graph:** a step joins the first layer after all its dependencies. Within a layer, steps are sorted by id.
- **A layer waits for the whole previous layer,** not only for the steps it depends on: a step whose dependency took 0.1 seconds still waits for an unrelated 10-second step of that layer.
- **`max_concurrent_steps`** caps the steps running at once within each layer, 10 by default. A subworkflow runs in an engine of its own, with its own cap.
- **A failed step doesn't stop its siblings.** A pause takes effect once the layer has finished. A budget breach stops the steps still running.
- One gateway serves the whole run, so parallel steps share each provider's circuit breaker and rate limit.

## State

`StateManager` holds the state behind one lock. Each attempt of a step reads a copy of the state taken when it starts, and the step writes its `output` only when it succeeds, along with `steps.<id>`. A failed attempt writes nothing to the state, but what its tools did isn't undone.

Steps of the same layer don't see each other's writes, as each read the state when it started. With more steps in a layer than `max_concurrent_steps`, the steps that wait for a slot start after some siblings finished, and see their outputs: what they render can differ from run to run. Two steps of one layer that write the same `output` key get a warning when the workflow loads, or an error with `config.strict_outputs: true`.

## Routers and skipped steps

After each layer, every router that succeeded activates the step its `output` names. In the next layers:

- **A step that depends on a router** runs only if the router activated it.
- **Every step after a branch that wasn't taken** is skipped, even one that also depends on the branch that was: a join of two branches is skipped unless both ran.
- **A router that failed** activates nothing. Under `skip_downstream`, its branches are skipped as after any failure; under `continue`, they all run.

Under `skip_downstream`, every step after a failed or timed-out step is skipped with `skipped due to upstream failure: <step>`, naming the closest failed step. A resumed run restores the choices of the routers that finished.

## Attempts and retries

Retries happen in the engine, per step: each attempt runs the whole step again, and [Reliability](reliability.md#what-is-retried) lists what is retried. The gateway doesn't retry: a call tries each of the model's providers once, and fails when they all fail.

The budget is checked before each step starts and charged after each attempt, failed ones included.

## The gateway

`ProviderGateway.register(provider, priority=0, is_fallback=False, models=None, max_rpm=60, max_tpm=100_000, circuit_fail_threshold=5, circuit_reset_timeout=60.0)` adds a provider. For each model, the gateway picks its candidates:

1. **A model listed in a provider's `models`** goes to those providers only, without fallbacks.
2. **Any other model** goes to the providers that serve it, by `priority`, lowest first, and then to the fallback providers.

For each candidate, an open circuit skips it, the rate limiter may make the call wait, and the call goes through the circuit breaker; a failure moves on to the next candidate. [Reliability](reliability.md#provider-failures) covers fallbacks, circuit breakers and rate limits from a workflow's side.

## Where redaction applies

Keys marked `redact` in `state_schema`, and those `AGENTLOOM_REDACT_STATE_KEYS` names, are masked as `<REDACTED:sha256=…>` in:

- the checkpoint: the state, the step results and the workflow definition;
- the `WorkflowResult`: its `final_state` and `step_results`, and so `--json`;
- prompts captured with `capture_prompts`;
- approval gate notifications;
- subworkflows, which inherit the policy.

The state stays as it is in memory, so prompts get the real values. History records hold no state. Recordings and the inter-agent event log aren't redacted. See [Redaction](state-and-templates.md#redaction).

## The end of a run

However a run ends, `success`, `failed`, `paused`, `budget_exceeded` or `timeout`, or with an unexpected error, the engine saves its checkpoint, once, when it has a checkpointer, then writes its history record. There is no checkpoint per step. A checkpoint or record that can't be written is logged, and the run goes on. A run stopped from outside, such as with Ctrl+C, writes neither.

A subworkflow saves its own checkpoint and record, under `<run id>.<step id>`.

## Collaborator protocols

The engine reaches its collaborators through the `typing.Protocol` interfaces of `agentloom.core.protocols`, so a test or an embedding program can pass its own:

| Protocol | For |
|----------|-----|
| `StateManagerProtocol` | Reading and writing the state. |
| `GatewayProtocol` | Calling providers. |
| `ToolRegistryProtocol` | Looking up tools. |
| `ObserverProtocol` | The run's events. See [Observability](observability.md#from-python). |
| `CheckpointerProtocol` | Saving and loading checkpoints. |
| `StreamCallbackProtocol` | Receiving a streamed reply's chunks. |

Each is `runtime_checkable`. The step retries and their waits come from `agentloom.resilience.retry`: `compute_backoff` and `is_retryable_exception`.

## Public contracts

`agentloom.contracts` is the import surface for code that reads AgentLoom's output or wraps its workflows without running them. Importing it loads neither the engine nor httpx, anyio or PyYAML.

| Module | Holds | Stability |
|--------|-------|-----------|
| `agentloom.contracts.stable` | `Message`, `Attachment`, `ToolCall`, `ToolDefinition`, `TokenUsage`, `ProviderResponse`, `ResponseMeta`, `WorkflowHandle` | Semantic versioning. |
| `agentloom.contracts.experimental` | The span, attribute and metric names (`SpanName`, `SpanAttr`, `MetricName`, `GenAIOperationName`, `GenAIProviderName`); the inter-agent event models and their JSON Schema (`SCHEMA_PATH`, `SCHEMA_URI`) | May change in a minor release; importing it warns once. |

The engine's own modules use the same classes: `agentloom.contracts.stable.Message` is `agentloom.core.models.Message`. The top-level `agentloom` package loads `Agent`, `AgentResult`, `WorkflowGraph`, `GraphNode` and `GraphEdge` on first use. Every other module can be imported, without a stability promise. `WorkflowHandle`, and the `WorkflowAdapter` that runs a workflow through it, are in [Python DSL](python-dsl.md#as-a-function).

## What's next

<!-- cards -->

- **[Concepts](concepts.md)** — Workflows, steps, layers and state, from a workflow author's side.
- **[Reliability](reliability.md)** — Retries, timeouts, budgets and checkpoints.
- **[Python DSL](python-dsl.md)** — Building and running workflows from Python.

<!-- /cards -->
