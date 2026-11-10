---
title: "Concepts"
description: "The ideas behind AgentLoom: workflows as DAGs of steps, a shared state, layers that run in parallel, one gateway to every provider, and runs."
order: 2
section: "Start"
---

# Concepts

A workflow is a graph of steps over a shared state. The engine runs the graph layer by layer, every model call goes through one gateway, and each run ends with a result and a record.

## Workflows

A workflow is a YAML file, or the same structure built with the [Python DSL](python-dsl.md):

```yaml
name: support-triage          # required
version: "1.0"
config:                       # defaults for every step
  provider: openai
  model: gpt-4o-mini
  budget_usd: 0.50
state:                        # the initial state
  ticket: "My invoice is wrong"
steps:                        # the graph
  - id: classify
    type: llm_call
    prompt: "Classify this ticket as billing or technical: {state.ticket}"
    output: category
```

`config` sets what steps inherit: the provider and model (OpenAI and `gpt-4o-mini` unless stated), the budget, the timeout, how many steps run at once and what happens when one fails. Unknown keys at the top level, in `config`, in a step and in their nested blocks are refused, so a misspelled field fails at load time rather than being ignored; `agentloom validate` checks a file without running it.

[Workflow files](workflow-files.md) lists every field of the file and of `config`.

## Steps

Each step has an `id`, a `type` and the fields of that type. There are seven:

| Type | What it does |
|------|--------------|
| `llm_call` | One model call: a prompt or a conversation, with optional attachments, tools and structured output. |
| `router` | Chooses which branch runs next, from conditions on the state. |
| `tool` | Runs a tool, built in or your own, with arguments taken from the state. |
| `subworkflow` | Runs another workflow as a single step. |
| `approval_gate` | Pauses the run until someone approves or rejects it. |
| `embed` | Turns text from the state into vectors. |
| `agent` | Runs a turn of a reusable agent: a system prompt, tools and a loop policy defined once. |

Every type also takes `depends_on` (the steps that must finish first), `output` (the state key the result goes to), `retry` and `timeout`. [Steps](steps.md) covers each type; [LLM calls](llm-calls.md), [Agents](agents.md) and [Tools & sandbox](tools-and-sandbox.md) take the three that need a page of their own.

## Dependencies and layers

`depends_on` is what makes the graph. AgentLoom checks it when the file loads: an unknown step id or a cycle is an error. Then the engine sorts the steps into **layers**: a step joins the first layer after all of its dependencies.

```yaml
steps:
  - id: fetch
    type: tool
    tool_name: http_request
    tool_args: { url: "https://example.com/report.json" }
    output: report
  - id: summarize
    type: llm_call
    depends_on: [fetch]
    prompt: "Summarize: {state.report.body}"
    output: summary
  - id: classify
    type: llm_call
    depends_on: [fetch]
    prompt: "Classify: {state.report.body}"
    output: label
  - id: publish
    type: llm_call
    depends_on: [summarize, classify]
    prompt: "Write a note from {state.summary} ({state.label})"
    output: note
```

```text
layer 1   fetch
layer 2   classify   summarize      (in parallel)
layer 3   publish
```

Steps in the same layer run at the same time, up to `config.max_concurrent_steps` (10 by default). The next layer starts when the whole layer has finished.

> [!IMPORTANT]
> Dependencies come only from `depends_on`, never from templates. A step that reads `{state.summary}` without depending on the step that writes it can run in the same layer or an earlier one, and read the key before it exists.

A `router` adds branches: the steps that depend on it run only if it chose them, and whatever depends on a branch that didn't run is skipped too.

## State

The state is one dictionary shared by the whole run. It starts from the file's `state:` block plus any `--state key=value` given to `agentloom run`, and grows as steps write their results:

- A step's `output:` names the key its result goes to. Dotted paths such as `report.summary` write into nested dictionaries.
- Templates read the state: `{state.ticket}`, `{state.items[0]}` or `{state.report.summary}`.
- Each successful step also lands under `steps.<id>`, so `{state.steps.classify.output}` reads a result without naming its key.
- A step reads the state as it stood when its layer started. Steps in the same layer don't see each other's writes, however `max_concurrent_steps` staggers them; the next layer sees them all.

A template that names a missing key renders an empty string, so a typo shows up as a blank in the prompt rather than an error. [State & templates](state-and-templates.md) has the full syntax.

## Providers and the gateway

Steps never call a provider directly. They name a model, and the **gateway** finds the providers that serve it, `config.provider` first: `claude*` models go to Anthropic, `gemini*` to Google, `gpt-*` and the `o` series to OpenAI, anything to Ollama or a `generic_openai` endpoint. On the way it applies each provider's rate limit and circuit breaker, and records the tokens and cost of every call.

| Provider | Serves |
|----------|--------|
| `openai`, `anthropic`, `google` | Their own models, with the key from the environment. |
| `ollama` | Any model pulled into a local Ollama. |
| `generic_openai` | Any OpenAI-compatible server: vLLM, LM Studio, a hosted gateway. |
| `mock` | Recorded or placeholder responses, offline, for tests. |

[Providers](providers.md) covers models, attachments, reasoning and pricing.

## Runs

Every run gets a `run_id`, twelve hexadecimal characters, and ends in one of five states:

| Status | Meaning |
|--------|---------|
| `success` | Every step that ran succeeded. |
| `failed` | At least one step failed or timed out. |
| `timeout` | The workflow hit `config.timeout`. |
| `budget_exceeded` | The spend reached `config.budget_usd`. |
| `paused` | An approval gate is waiting; `agentloom resume` continues the run. |

A step that fails is first retried: three more times by default, with exponential backoff. Errors that can't succeed on a second try, like a rejected key or an unknown model, are not retried. If the step still fails, `config.on_step_failure` decides the rest: `skip_downstream` (the default) skips every step that depends on it, while `continue` runs them anyway. Either way, the run ends `failed`. [Reliability](reliability.md) covers retries, timeouts, budgets and resuming a run.

The run returns a result with the final state, each step's status, duration, tokens and cost, and the totals. `agentloom run` prints a summary of it, or all of it with `--json`, and saves a record to `agentloom_runs/`. With `--checkpoint` it also saves the state it stopped with, for [`agentloom resume`](cli.md#resume).

## What deterministic means here

The model fills in text; it never decides what runs. The steps, their order and the routing rules are all in the file, so two runs of the same workflow take the same path whenever the routers see the same state. Record a run once with `--record` and [`agentloom replay`](testing-and-replay.md) repeats it offline with the same responses, for tests and debugging.

## What's next

<!-- cards -->

- **[Workflow files](workflow-files.md)** — Every key of a workflow file, every config option and what validate checks.
- **[Steps](steps.md)** — The seven step types, with routers, subworkflows, approval gates and embeddings in full.
- **[State & templates](state-and-templates.md)** — How steps write to the state, and the template syntax that reads it.
- **[Providers](providers.md)** — Providers, models, attachments, reasoning and pricing.
- **[Examples](examples.md)** — Workflows that put these pieces together.
- **[Architecture](architecture.md)** — How the engine schedules, runs and records a workflow.

<!-- /cards -->
