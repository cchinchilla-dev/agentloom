---
title: "Python DSL"
description: "Build AgentLoom workflows in Python or load them from YAML, then run them with the engine, the providers and the tools you choose."
order: 13
section: "Running"
---

# Python DSL

Workflows can be built in Python as well as written in YAML. Either way, a Python program runs them with `WorkflowEngine`, with the providers, tools and checkpoints it sets up itself.

## Building a workflow

```python
from agentloom.core.dsl import workflow

wf = (
    workflow("support-reply", model="gpt-4o-mini", budget_usd=0.10)
    .set_state(message="My order arrived damaged")
    .add_llm_step(
        "classify",
        system_prompt="Reply with one word: billing, technical or general.",
        prompt="Classify: {state.message}",
        output="category",
    )
    .add_llm_step(
        "reply",
        depends_on=["classify"],
        prompt="Write a short {state.category} support reply to: {state.message}",
        output="reply",
        temperature=0.3,
        retry={"max_retries": 2},
    )
    .build()
)
```

`workflow(name, description="", **config)` starts a builder. Its keyword arguments are the keys of a file's [`config`](workflow-files.md#config), such as `model`, `budget_usd`, `timeout`, `on_step_failure` or `sandbox`. Every method returns the builder, and `build()` returns the workflow.

| Method | What it adds |
|--------|--------------|
| `set_state(**values)` | Keys of the initial state. Several calls add up. |
| `set_state_schema(schema)` | Per-key [state settings](state-and-templates.md#redaction), such as `{"api_key": {"redact": True}}`. Keys may be globs. Several calls add up. |
| `add_llm_step(step_id, prompt, system_prompt=None, model=None, output=None, depends_on=None, attachments=None, stream=None, **fields)` | An [`llm_call`](llm-calls.md). `**fields` takes any other field of the step: `temperature`, `max_tokens`, `timeout`, `retry`, `thinking`, `tools`, `tool_choice`, `response_schema`, `conversation`, `speaker`. |
| `add_tool_step(step_id, tool_name, tool_args=None, output=None, depends_on=None)` | A [`tool`](tools-and-sandbox.md#the-tool-step) step. |
| `add_router_step(step_id, conditions, default=None, depends_on=None)` | A [`router`](steps.md#router). `conditions` is a list of `(expression, target)` pairs. |
| `add_subworkflow_step(step_id, workflow_path=None, workflow_inline=None, output=None, depends_on=None)` | A [`subworkflow`](steps.md#subworkflow). |
| `add_embed_step(step_id, inputs, model=None, dimensions=None, output=None, depends_on=None)` | An `embed` step. `inputs` is a state path. |
| `add_approval_gate_step(step_id, timeout_seconds=None, on_timeout=None, notify=None, output=None, depends_on=None)` | An `approval_gate` step. |
| `add_agent_step(step_id, agent, prompt=None, conversation=None, output=None, depends_on=None, attachments=None)` | An [`agent`](agents.md) step. `agent` is an `AgentDefinition` or the path of its YAML file. |

```python
.add_router_step(
    "route",
    depends_on=["classify"],
    conditions=[
        ("state.category.strip().lower() == 'billing'", "handle_billing"),
        ("state.category.strip().lower() == 'technical'", "handle_technical"),
    ],
    default="handle_general",
)
```

- A wrong field fails at once, at the `workflow()` or `add_…` call that sets it: an unknown key, a value of the wrong type, `tools` and `response_schema` on one step.
- A Pydantic `response_schema` names its class by path, as in YAML. In a script, `"__main__:Answer"` names a class defined in that script.

### What the builder doesn't cover

- `version` can't be set.
- Only `add_llm_step` takes extra fields: the other steps get no `timeout` or `retry`, and a subworkflow no `isolated_state`, `input` or `return_keys`.

For those, write the workflow as a dictionary, the same shape as the YAML file, and load it with `WorkflowParser.from_dict`. Or add to a build and validate the whole again:

```python
from agentloom.core.models import WorkflowDefinition

wf = WorkflowDefinition(**{
    **wf.model_dump(),
    "version": "2.0",
    "steps": [*wf.model_dump()["steps"],
              {"id": "ping", "type": "tool", "tool_name": "http_request", "depends_on": ["reply"],
               "retry": {"max_retries": 0}}],
})
```

### Validating a build

`build()` checks each field, that step ids are unique, that dependencies, router targets and subworkflow files exist, that there are no cycles, and that steps of one layer don't share a conversation. It raises `ValidationError`. Only parallel steps writing one `output` key aren't checked.

To get that warning too, run the checks that `agentloom validate` runs:

```python
from agentloom.core.parser import WorkflowParser

WorkflowParser.from_dict(wf.model_dump())   # raises ValidationError, warns on shared output keys
```

## Loading YAML

```python
from agentloom.core.parser import WorkflowParser

wf = WorkflowParser.from_yaml("pipeline.yaml")
wf = WorkflowParser.from_dict({"name": "inline", "steps": [...]})
```

Both run every check that `agentloom validate` runs, and raise `agentloom.exceptions.ValidationError`.

`from_yaml` takes a path or YAML text. A string with no file behind it is read as YAML, unless it looks like a path, one line without a `:`, which fails with `Workflow file not found`. A `Path` is always a path. Text too long to be a file name is read as YAML.

## Running a workflow

```python
import anyio

from agentloom.core.engine import WorkflowEngine
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.openai import OpenAIProvider

async def main() -> None:
    gateway = ProviderGateway()
    gateway.register(OpenAIProvider(), priority=0)   # reads OPENAI_API_KEY
    try:
        result = await WorkflowEngine(wf, provider_gateway=gateway).run()
    finally:
        await gateway.close()
    print(result.status, result.final_state["reply"])

anyio.run(main)
```

`run()` is asynchronous: call it from a coroutine, with `anyio.run` or `asyncio.run`.

| `WorkflowEngine` argument | What it is |
|---------------------------|------------|
| `workflow` | The workflow, built or loaded. |
| `provider_gateway` | The providers, below. Without one, every `llm_call`, `agent` and `embed` step fails, after its retries. |
| `tool_registry` | The tools. See [Tools of your own](tools-and-sandbox.md#tools-of-your-own). Without one, every `tool` step fails. |
| `state_manager` | A `StateManager(initial_state={...})` from `agentloom.core.state`, to start from another state. It replaces the workflow's `state` rather than adding to it. |
| `checkpointer`, `run_id` | Saving the run to resume it. See [Reliability](reliability.md#from-python). |
| `on_stream_chunk` | A function called with `(step_id, chunk)` for each piece of a streamed reply. |
| `observer` | Hooks called as the run goes. Subclass `NoopObserver` from `agentloom.observability.noop` and override the hooks you need. See [Observability](observability.md). |

- **The gateway decides which providers answer.** From Python, `config.provider` has no effect. `config.model` is still the model of every step that doesn't name its own.
- **A run that doesn't succeed returns a result,** with its status, as it does from the CLI. `run()` raises only for a cycle in a workflow that wasn't validated, an observer that lacks a hook, or a cancellation.
- **Every run writes a record** to `agentloom_runs/<run_id>.json`, in the current directory, or under `AGENTLOOM_RUNS_DIR`. See [Observability](observability.md).
- **The engine doesn't close the providers:** call `await gateway.close()` when done.

### The provider gateway

```python
gateway.register(provider, priority=0, is_fallback=False, models=None,
                 max_rpm=60, max_tpm=100_000,
                 circuit_fail_threshold=5, circuit_reset_timeout=60.0,
                 is_primary=False, model_aliases=None)
```

- A model goes to the providers that list it in `models`, or, if no provider lists it, to every provider that serves it by name; then to those registered with `is_fallback=True`. `priority` orders each group, lowest first.
- `is_primary=True` puts the provider ahead of the rest for any model it can serve. `model_aliases` maps a requested model name to this provider's own, and makes it a candidate for the requested one.
- A streamed call waits for the first chunk inside `gateway.stream()`, so a provider that fails before sending one is replaced by the next. Iterate the stream it returns, as an abandoned one holds its connection open.
- `max_rpm` and `max_tpm` set the provider's rate limits, and `circuit_fail_threshold` and `circuit_reset_timeout` its circuit breaker. See [Provider failures](reliability.md#provider-failures).

| Provider | Import from | Models it serves by name |
|----------|-------------|--------------------------|
| `OpenAIProvider` | `agentloom.providers.openai` | Names that start with `gpt-`, `o1`, `o3`, `o4`, `text-embedding-3-` or `text-embedding-ada-`. |
| `AnthropicProvider` | `agentloom.providers.anthropic` | Names that contain `claude`. |
| `GoogleProvider` | `agentloom.providers.google` | Names that contain `gemini`, and `embedding-…`. |
| `OllamaProvider` | `agentloom.providers.ollama` | Any. |
| `GenericOpenAIProvider(base_url)` | `agentloom.providers.generic_openai` | Any. |
| `MockProvider(responses_file=...)` | `agentloom.providers.mock` | Any. Replays a recording; see [Testing & replay](testing-and-replay.md). |

Without an `api_key`, OpenAI, Anthropic and Google read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`, and Ollama reads its address from `OLLAMA_BASE_URL`. The other `*_BASE_URL` variables apply only to the CLI: pass `base_url` instead. [Providers](providers.md) covers each one.

### Results

`run()` returns a `WorkflowResult`:

| Field | What it holds |
|-------|---------------|
| `status` | `success`, `failed`, `timeout`, `budget_exceeded` or `paused`. |
| `final_state` | The state at the end, with `steps.<id>` for each step that succeeded. |
| `step_results` | A `StepResult` per step that ran: `status`, `output`, `error`, `duration_ms`, `token_usage`, `cost_usd`, `model`, `provider`, `error_classification` and more. |
| `total_tokens`, `total_cost_usd`, `total_duration_ms` | The run's totals. |
| `error` | The first failed step's error. |
| `run_id`, `workflow_name` | The run and its workflow. |

[Redacted keys](state-and-templates.md#redaction) are redacted in `final_state` and `step_results` too.

### As a function

`WorkflowAdapter(workflow, provider_gateway, tool_registry=None, observer=None)`, from `agentloom.core.handle`, wraps a workflow as a call: `await adapter.run(inputs)` starts from the workflow's state with `inputs` on top, and returns the final state, or raises `WorkflowError` when the run doesn't succeed. It takes no checkpointer.

## The state from Python

`StateManager` holds a run's state. From a step or a tool of your own:

- `await state.get("user.profile.email")` and `await state.set("report.summary", text)` read and write dotted paths, by the rules of [Writing state](state-and-templates.md#writing-state).
- `await state.update("counter", lambda n: (n or 0) + 1)` reads and writes under one lock, and returns the new value. A `get` followed by a `set` doesn't: fifty parallel increments written that way leave `1`, while fifty `update` calls leave `50`.
- `await state.get_state_snapshot()` returns a copy of the whole state.

A write that the rules refuse raises `StateWriteError`, from `agentloom.exceptions`. When it comes from a step's `output`, the step fails at once, without retries: the same write would be refused again.

## Imports

`agentloom` itself exports `Agent`, `AgentResult`, `WorkflowGraph`, `GraphNode` and `GraphEdge`. The rest comes from its modules:

| Name | Import from |
|------|-------------|
| `workflow` | `agentloom.core.dsl` |
| `WorkflowParser` | `agentloom.core.parser` |
| `WorkflowEngine` | `agentloom.core.engine` |
| `WorkflowResult`, `StepResult` | `agentloom.core.results` |
| `WorkflowDefinition`, `StepDefinition` | `agentloom.core.models` |
| `ProviderGateway` | `agentloom.providers.gateway` |
| `ToolRegistry`, `BaseTool`, `tool` | `agentloom.tools` |
| `FileCheckpointer` | `agentloom.checkpointing` |
| `ValidationError`, `WorkflowError`, `StateWriteError` | `agentloom.exceptions` |

`agentloom.contracts.stable` follows semantic versioning: `Attachment`, `Message`, `ProviderResponse`, `ResponseMeta`, `TokenUsage`, `ToolCall`, `ToolDefinition` and `WorkflowHandle` change only in a major version. `agentloom.contracts.experimental` warns on import that it may change. No other module makes that promise.

## What's next

<!-- cards -->

- **[Graph API](graph-api.md)** — Read, analyse and export a workflow's DAG.
- **[Tools & sandbox](tools-and-sandbox.md)** — Tools of your own, registered from Python.
- **[Testing & replay](testing-and-replay.md)** — Running workflows offline against recordings.

<!-- /cards -->
