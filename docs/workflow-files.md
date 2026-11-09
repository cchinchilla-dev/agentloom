---
title: "Workflow files"
description: "The structure of an AgentLoom workflow file: top-level keys, every config option, the fields all steps share, and what validate checks before a run."
order: 4
section: "Workflows"
---

# Workflow files

A workflow file is YAML with a name, defaults in `config`, an initial `state` and a list of `steps`. This page covers the file itself and the fields every step shares; [Steps](steps.md) covers each step type.

## File structure

```yaml
name: support-triage          # required
version: "1.0"
description: "Classify a ticket and draft a reply"
config:                       # defaults for the whole run
  model: gpt-4o-mini
  budget_usd: 0.50
state:                        # the initial state
  ticket: "My invoice is wrong"
state_schema:                 # state keys kept out of checkpoints, records and traces
  api_key: { redact: true }
steps:                        # required: the graph
  - id: classify
    type: llm_call
    prompt: "Classify this ticket as billing or technical: {state.ticket}"
    output: category
```

| Key | Default | What it is |
|-----|---------|------------|
| `name` | required | Names the run in the summary, the run record and the traces. |
| `version` | `"1.0"` | Free text, which `validate` prints. Quote it: an unquoted `1.0` is a number, and the file doesn't load. |
| `description` | empty | Free text, for whoever reads the file. |
| `config` | the defaults below | Defaults for the run: provider, model, budget, timeouts, failure policy. |
| `state` | empty | The state the run starts with. See [State & templates](state-and-templates.md). |
| `state_schema` | empty | State keys whose values never reach a checkpoint, a record or a trace. See [Redaction](state-and-templates.md#redaction). |
| `steps` | required | The steps, in any order: `depends_on`, not their position in the file, decides when each one runs. |

### Unknown keys

AgentLoom checks every key when the file loads. A misspelled key at the top level, in `config` or in a step stops the load, and the error names it:

```text
Validation FAILED:
Invalid workflow definition: 1 validation error for WorkflowDefinition
steps.0.promt
  Extra inputs are not permitted [type=extra_forbidden, input_value='Summarize…', input_type=str]
```

Nested blocks are looser. Inside `retry`, `sandbox`, `notify`, `thinking`, `attachments`, `tools` and a router's `conditions`, an unknown key is ignored: `sandbox: {allowed_domain: [...]}` loads without an error, and the setting is lost. Check the spelling of those against their reference.

A field of another step type is accepted too, and ignored: a `prompt` on a router does nothing.

## Config

Every key is optional.

| Key | Default | What it sets |
|-----|---------|--------------|
| `provider` | `openai` | The provider tried first. The model name decides which providers can serve a call; see [Providers](providers.md). |
| `model` | `gpt-4o-mini` | The model of every step that doesn't name its own. |
| `base_url` | none | The endpoint of `provider: generic_openai`. Required there, and refused with any other provider. |
| `api_key_env` | none | The environment variable that holds the key for `generic_openai`. Without it, requests carry no key. |
| `budget_usd` | no limit | The most the run may spend, in US dollars. See [Reliability](reliability.md). |
| `timeout` | no limit | Seconds for the whole run; `0` also means no limit. Past it, the steps it cut short end `timeout`, no later step starts, and the run ends `timeout`. |
| `max_concurrent_steps` | `10` | How many steps of one layer run at once, from 1 to 1024. |
| `stream` | `false` | Stream model replies as they are written. A step's own `stream` overrides it. |
| `on_step_failure` | `skip_downstream` | What a failed step does to the steps after it: `skip_downstream` skips them, `continue` runs them anyway. See [Reliability](reliability.md). |
| `strict_outputs` | `false` | Refuse the file, rather than warn, when steps that can run in parallel write the same `output` key. See [Parallel writes](#parallel-writes-to-one-key). |
| `sandbox` | disabled | Limits on what the built-in tools may run, read, write and reach, and on webhook destinations. See [Tools & sandbox](tools-and-sandbox.md). |
| `capture_prompts` | `false` | Add each rendered prompt and system prompt to the trace of its `llm_call`. See [Observability](observability.md). |
| `responses_file` | none | The recording that `provider: mock` replays. See [Testing & Replay](testing-and-replay.md). |
| `latency_model` | `constant` | How the `mock` provider simulates latency: `constant`, `normal` or `replay`. |
| `latency_ms` | `0` | The `mock` provider's simulated latency per call. |
| `max_retries` | `3` | Retries for each step whose `retry` doesn't set `max_retries`. A subworkflow's steps take the child's value. |

Four options of `agentloom run` override `config` for one run: `--provider`, `--model`, `--budget` and `--stream`. They change the top-level workflow only; a [subworkflow](steps.md#subworkflow) keeps its own `config`.

## Fields every step has

| Field | Default | What it does |
|-------|---------|--------------|
| `id` | required | Names the step in `depends_on`, in router targets, under `steps.<id>` in the state and in the run's results. Unique within the file. |
| `type` | required | One of the seven [step types](steps.md#step-types-at-a-glance). |
| `depends_on` | `[]` | The steps that must finish before this one starts. See [Dependencies and layers](concepts.md#dependencies-and-layers). |
| `output` | none | The state key the step's result goes to. A dotted path such as `report.summary` writes into nested dictionaries; see [Writing state](state-and-templates.md#writing-state). |
| `timeout` | no limit | Seconds for each attempt; `0` also means no limit. An attempt that runs out is retried like a failed one. |
| `retry` | `config.max_retries` retries | How failed attempts are retried: `max_retries`, `backoff_base`, `backoff_max`, `jitter` and `retryable_status_codes`. See [Reliability](reliability.md). |

`retry` and `timeout` apply to every step type, a router or a subworkflow included. The other fields belong to one type or a few, and each type's reference lists them.

## What validate checks

`agentloom validate` loads a file the way `agentloom run` does, without calling any provider, and prints the plan:

```text
Workflow 'support-triage' is valid.
  Version:  1.0
  Steps:    4
  Layers:   3
  Provider: openai
  Model:    gpt-4o-mini
  Budget:   $0.50

  Execution plan:
    Layer 0: fetch
    Layer 1: classify, summarize (parallel)
    Layer 2: publish
```

**Checked when the file loads,** by `validate` and before every run:

- YAML syntax, the required keys (`name`, `steps`, and each step's `id` and `type`) and the type of every value.
- Unknown keys at the top level, in `config` and in steps.
- Duplicate step ids.
- `depends_on` entries and router targets (`target` and `default`) that name no step.
- Cycles in `depends_on`.
- A `workflow_path` that doesn't exist, relative to the directory you run from.
- `generic_openai` without `base_url`, or `base_url` and `api_key_env` with any other provider.
- `tools` and `response_schema` on the same `llm_call`.
- Agent steps without an `agent`, or with a setting that comes from the agent's definition.
- Bounds: `max_concurrent_steps` from 1 to 1024, a `timeout` of 0 or more, `max_tool_iterations` and `dimensions` of 1 or more.
- Steps that can run in parallel and write the same `output` key: a warning, or an error with `strict_outputs`.

**Checked only when the step runs,** so `validate` passes them:

- A router with neither `conditions` nor `default`, and the router's expressions themselves.
- A `tool` step without `tool_name`, or naming a tool that isn't registered.
- A subworkflow with neither `workflow_path` nor `workflow_inline`, and the child workflow's own contents.
- An `embed` step without `inputs`.
- An `llm_call` with neither `prompt` nor `conversation`.

A file that fails to load makes `validate` exit with code 1 and print the reason.

## Parallel writes to one key

Two steps that can run in parallel and write the same `output` key race: whichever finishes last wins. The parser warns when it finds a pair:

```text
Parallel steps share an output key under last-writer-wins semantics:
output='reply' written by ['answer', 'escalate']. Set config.strict_outputs: true
to promote this to a parse error, or chain the steps via depends_on if the
overwrite is intentional.
```

A chain is exempt: when one step depends on the other, directly or through others, the second overwrites the first on purpose. The branches of a router trigger the warning too, although only one of them runs, because the parser doesn't follow routing. Give each branch its own key, or leave the warning.

With `strict_outputs: true`, the warning becomes an error and the file doesn't load.

## A complete file

```yaml
name: classify-and-respond
version: "1.0"
description: "Classify a message, then answer it or hand it to a person"

config:
  model: gpt-4o-mini
  budget_usd: 0.50
  timeout: 120

state:
  message: "How do I export my invoices?"

steps:
  - id: classify
    type: llm_call
    system_prompt: "Reply with one word: question, complaint or request."
    prompt: "{state.message}"
    output: category

  - id: route
    type: router
    depends_on: [classify]
    conditions:
      - expression: "state.category.strip().lower() == 'question'"
        target: answer
    default: escalate

  - id: answer
    type: llm_call
    depends_on: [route]
    prompt: "Answer briefly: {state.message}"
    output: reply

  - id: escalate
    type: llm_call
    depends_on: [route]
    prompt: "Summarize this message in two lines for a support agent: {state.message}"
    output: handoff
```

Three layers: `classify`, then `route`, then whichever of `answer` and `escalate` the router chose. `agentloom run classify.yaml --state message="The app crashes on login"` runs it with another message.

## What's next

<!-- cards -->

- **[Steps](steps.md)** — Every step type, with routers, subworkflows, approval gates and embeddings in full.
- **[State & templates](state-and-templates.md)** — How the state starts, how steps write to it and how templates read it.
- **[Reliability](reliability.md)** — Retries, timeouts, failure policies, budgets and checkpoints.

<!-- /cards -->
