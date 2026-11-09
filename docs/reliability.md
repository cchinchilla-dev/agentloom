---
title: "Reliability"
description: "How AgentLoom retries failed steps, enforces timeouts and budgets, handles the steps after a failure, and saves runs to resume them."
order: 10
section: "Workflows"
---

# Reliability

What happens when a step fails: how it is retried, how long it may take, what it does to the steps after it, how much a run may spend, and how to pick a run up again from where it stopped.

## Retries

Each step retries its own failed attempts, with waits that grow between them:

```yaml
- id: summarize
  type: llm_call
  prompt: "Summarize: {state.report}"
  retry:
    max_retries: 5
    backoff_base: 2
    backoff_max: 30
    retryable_status_codes: [429, 500, 502, 503, 504, 529]
  timeout: 60
```

| Field | Default | What it is |
|-------|---------|------------|
| `max_retries` | `config.max_retries`, `3` by default | Retries after the first attempt: `3` allows four attempts, `0` only one. |
| `backoff_base` | `2` | How fast the waits grow. |
| `backoff_max` | `60` | The longest wait, in seconds. |
| `jitter` | `true` | Varies each wait by up to 25% either way, so parallel steps don't retry in step. |
| `retryable_status_codes` | `429`, `500`, `502`, `503`, `504` | The provider statuses worth another attempt. |

- **The waits** are 1 second before the first retry, then `backoff_base`, `backoff_base²` and so on, up to `backoff_max`. Jitter applies after that cap. With the defaults, a step waits about 1, 2 and 4 seconds.
- **Every step has its own `retry`,** a router or a subworkflow included. `config.max_retries` sets `max_retries` for each step that doesn't set its own; a subworkflow's steps take their child's `config.max_retries`.
- **An attempt runs the whole step again.** It renders its prompts from the state as it is then, runs its tools again, and runs a subworkflow from its first step. Nothing a failed attempt did is undone, so tools with side effects should be safe to repeat.
- **Retries multiply in subworkflows.** A child's steps retry inside each attempt of the `subworkflow` step: two retries in the child and one in the parent make up to six attempts.
- A provider's `Retry-After` header isn't waited for: the step's own waits apply.
- A negative `max_retries` fails when the workflow loads. The other values aren't checked, and a misspelled key under `retry` is ignored.

A failed step's result says whether its last failure was `transient` or `permanent`, but not how many attempts it took. The log has a line for each retry: `Step 'summarize' failed (attempt 2/6), retrying in 2.1s`.

### What is retried

A failure is either permanent, and fails the step at once, or transient, and is retried. For a provider error, the status decides: one listed in `retryable_status_codes` is transient, any other status is permanent, and an error without a status, such as a dropped connection, a provider timeout or an open circuit breaker, is transient. A model no provider serves, or a request none of them can take, is permanent.

| Step | Fails at once | Retried |
|------|---------------|---------|
| `llm_call` | A provider status not listed, a broken prompt template, a structured reply that never validates, a conversation budget too small for the turn, an unusable conversation, an attachment that is refused, can't be read or that no provider can take. See [LLM calls](llm-calls.md#when-a-call-fails). | Listed statuses, errors without a status. |
| `agent` | A broken constraint, the iteration cap, a missing tool. See [Agents](agents.md#retries-and-results). | `max_latency_ms`, and the provider errors an `llm_call` retries. |
| `tool` | A tool that isn't registered, arguments that can't be resolved, a sandbox refusal. | Any other error the tool raises. |
| `router` | Every failure: an expression it doesn't allow, a syntax error, a condition that raises, no match and no `default`. | None of its own. |
| `subworkflow` | A child that can't be loaded, that failed permanently, or that went over its own budget. | Any other failure of the child. |
| `embed` | A provider status not listed, a missing input or one of the wrong type, no provider with embeddings. | Listed statuses, errors without a status. |
| `approval_gate` | A gate that expired with no `on_timeout`, a decision other than `approved` or `rejected`. | None of its own. |
| Any step | Going over the run's budget. | The step's `timeout`. |

AgentLoom can't tell which of a tool's own errors would pass on a later attempt. For a tool that fails the same way each time, set `retry: {max_retries: 0}`.

## Timeouts

- **A step's `timeout`** bounds each attempt, in seconds, for any step type; `0` or no value means no limit. An attempt that runs out ends `timeout` and is retried while attempts remain, whatever `retryable_status_codes` says. The waits between attempts don't count toward it.
- **`config.timeout`** bounds the whole run. When it runs out, the steps running at that moment end `timeout`, with `Workflow timed out after 300.0s`, later steps don't start and have no result, and the run ends `timeout`. It cuts a wait between retries short too.
- **An agent's `constraints.max_latency_ms`** bounds its turn as `timeout` does, and the shorter one applies. Running out fails the turn as a transient failure, rather than ending it `timeout`.
- **Provider requests** time out after 60 seconds, or 120 for Ollama, which can't be changed. A provider timeout is an error without a status: it is retried, and counts against the provider's circuit breaker.

A step that ends `timeout` counts as failed for the steps after it and for the run's status.

## When a step fails

`config.on_step_failure` decides what a failed or timed-out step does to the steps after it:

- **`skip_downstream`**, the default, skips every step that depends on it, directly or through other steps, with `skipped due to upstream failure: fetch`. That includes a step that also depends on steps that succeeded. Steps on other branches still run, and the steps of the same layer finish.
- **`continue`** runs them anyway, on what the state holds.

Under `continue`, keep in mind that a failed step writes nothing:

- its `output` key and `steps.<id>` don't exist, so `{state.summary}` renders empty, with a warning;
- a router can't tell it failed: `steps.fetch.status` reads `None`, not `failed`;
- a router that fails runs all of its branches.

Either way, a run with a failed step ends `failed`.

### Statuses and exit codes

- **A step** ends `success`, `failed`, `skipped`, `timeout` or `paused`. A step that never started has no result.
- **A run** ends `success`, `failed`, `timeout`, `budget_exceeded` or `paused`. When several apply, `paused` wins over `budget_exceeded`, which wins over `timeout`, which wins over `failed`.

`agentloom run` exits with:

| Code | When |
|------|------|
| `0` | The run succeeded or paused. |
| `1` | The run failed, timed out or went over budget, or the workflow couldn't load. |
| `2` | The options contradict each other, such as `--record-tools` without `--record`. |
| `130` | The run was interrupted with Ctrl-C or SIGTERM. |

`agentloom resume` exits with `0` or `1` the same way. Its summary marks a timed-out step `[FAIL]`, like a failed one. With `--json`, the output starts with a line such as `Running workflow: …`, and `Run ID: …` with `--checkpoint`, before the JSON.

## Budgets

`config.budget_usd`, or `agentloom run --budget 0.50`, caps what a run may spend, in US dollars.

- **When it's checked.** Before each step starts, a step is refused once the spend has reached the limit. After each attempt, its cost is added, and the run stops if the total is over. There is no check before each call inside a step, so one step with a tool loop or several re-asks can go over by all its calls; an [agent](agents.md#constraints) has `max_cost_usd` for that.
- **When it's exceeded.** The step whose cost crossed the limit keeps its result. The other steps running in its layer are stopped and have no result, though what they spent still counts. Later steps don't start. The run ends `budget_exceeded`, with `Budget exceeded: spent $0.5120 of $0.5000 limit`, and `agentloom run` exits with `1`.
- **What counts.** Every call a step makes: failed attempts, each turn of a tool loop, re-asks for valid JSON, conversation summaries, agents called as tools and embeddings.
- **Subworkflows** share the parent's budget, unless the child sets its own `budget_usd`. A child that goes over its own budget fails its step without a retry, and its cost still counts toward the parent's budget.

### Prices

Costs come from a table of prices per 1,000 input and output tokens that ships with AgentLoom, in `agentloom/providers/pricing.yaml`. A model is matched by its name, or by the longest entry its name starts with, so a dated name without an entry of its own, such as `gpt-4o-mini-2024-07-18`, takes the price of `gpt-4o-mini`. Reasoning tokens are billed at the output price.

A model without an entry costs 0, so a budget never stops it: local Ollama models, and any model newer than the table. `AGENTLOOM_PRICING_FILE=/path/to/pricing.yaml` replaces the whole table: copy the bundled file and add your models to it:

```yaml
my-finetuned-model:
  input: 0.0008
  output: 0.0032
```

## Provider failures

Each attempt goes through the provider gateway, which has protections of its own. [Providers](providers.md) covers them in full; this is how they combine with retries:

- **Fallback.** An attempt tries the providers that serve the model, in order, then the fallback providers. Only when all of them fail does the attempt fail, with a status only if they all returned the same one. Every provider receives the same model name, and a [streamed](llm-calls.md#streaming) call doesn't fall back at all.
- **Circuit breaker.** Five failed calls in a row to one provider, from any step, open its circuit, and calls skip that provider for 60 seconds; one call then tests it. 429 responses don't count, except on a streamed call, and neither does a request the provider can't take. With no other provider for the model, an open circuit fails each attempt as transient, and the step goes on retrying until its retries run out.
- **Rate limits.** Each provider is held to 60 requests and 100,000 tokens a minute, the `mock` provider included when `agentloom run` sets it up.

A resumed run starts with closed circuits and fresh rate limits.

## Checkpoints and resume

### Saving a run

```bash
agentloom run pipeline.yaml --checkpoint
agentloom run pipeline.yaml --checkpoint --checkpoint-dir runs/checkpoints
```

With `--checkpoint`, the run prints its `Run ID` and saves a checkpoint to `.agentloom/checkpoints/<run_id>.json`, or to `--checkpoint-dir`. The checkpoint holds the workflow as it ran, the command line's overrides included, the state, every step's result, the status and the spend, with [redacted keys](state-and-templates.md#redaction) replaced. A subworkflow saves its own, as `<run_id>.<step_id>.json`.

The checkpoint is written once, when the run ends: `success`, `failed`, `timeout`, `budget_exceeded` or `paused`, or `cancelled` when Ctrl-C or SIGTERM interrupts it. `agentloom resume` continues a cancelled run after its finished steps; a step that was running when it stopped runs again from the start, and is paid again. A subworkflow cut short with its parent leaves the checkpoint to the parent.

> [!WARNING]
> A process that is killed outright, with SIGKILL or by running out of memory, saves nothing: there is no checkpoint to resume, and the work it did is lost. A resumed run killed so leaves its previous checkpoint as it was.

`agentloom runs` lists the checkpoints, with each run's id, workflow, status and last update; `--json` prints them whole. A resumed run updates its own checkpoint, under the same id.

Whoever can write to the checkpoint directory decides what a resume runs: keep it as private as the workflows themselves.

### Resuming

```bash
agentloom resume 3f9c2a7b1e04
```

- **The checkpoint is what runs,** not the YAML file: changes made to the file since don't apply. Providers are set up again, so API keys are read from the environment, and a `mock` provider reads its recording again.
- **Steps that succeeded keep their results.** Every other step runs: failed, timed out, skipped, paused and never started.
- **The budget goes on from what was spent,** and `config.timeout` starts over.
- **Redacted keys come back redacted:** steps that read them receive `<REDACTED:sha256=…>`.

Any run with a checkpoint can be resumed. A successful one has nothing left to run. One that went over budget goes over again at once: `agentloom resume` has no `--budget`, and a higher limit has to be set from Python.

After a router, steps two or more levels down a branch it didn't take run on resume, as if the branch had been taken.

`agentloom resume` takes `--checkpoint-dir`, `--provider`, `--model`, `--approve`, `--reject`, `--lite`, `--json` and `--stream`. See [CLI](cli.md).

### Approval gates

An [approval gate](steps.md#approval_gate) pauses the run and waits for a decision:

```bash
agentloom run publish.yaml --checkpoint
agentloom resume 3f9c2a7b1e04 --approve
agentloom resume 3f9c2a7b1e04 --reject
```

- The run needs `--checkpoint`. Without it, the run still pauses and prints the commands to resume it, but there is nothing to resume.
- The decision becomes the gate's `output`, `approved` or `rejected`, and the gate succeeds either way: steps after it run unless a [router](steps.md#router) acts on the decision.
- A resume without `--approve` or `--reject` pauses at the gate again, and sends its webhook again.
- Gates that pause in the same layer take one resume each.
- A gate with `timeout_seconds` that nobody decides in time takes its `on_timeout`, or fails the run. See [When nobody decides](steps.md#when-nobody-decides).
- `agentloom callback-server` resumes paused runs over HTTP, from the same checkpoint directory. See [CLI](cli.md).

### From Python

```python
from agentloom.checkpointing import FileCheckpointer
from agentloom.core.engine import WorkflowEngine

checkpointer = FileCheckpointer(".agentloom/checkpoints")
engine = WorkflowEngine(workflow, provider_gateway=gateway, tool_registry=registry,
                        checkpointer=checkpointer, run_id="nightly-report")
result = await engine.run()

data = await checkpointer.load("nightly-report")
engine = await WorkflowEngine.from_checkpoint(
    data, checkpointer, provider_gateway=gateway, tool_registry=registry,
    approval_decisions={data.paused_step_id: "approved"},   # for a paused run
)
result = await engine.run()
```

- `run_id` is optional; without it, the engine makes one up.
- Before resuming, `data` can be changed: `data.workflow_definition["config"]["budget_usd"]` raises the limit of a run that went over it, and `data.state` takes back the values of redacted keys.
- A checkpoint store of your own subclasses `BaseCheckpointer`, with async `save`, `load`, which raises `KeyError` for an unknown run, `list_runs` and `delete`.
- Checkpoints need a `WorkflowEngine`: `WorkflowAdapter` runs without them.

## What's next

<!-- cards -->

- **[Providers](providers.md)** — Fallback, circuit breakers and rate limits in full.
- **[Observability](observability.md)** — Traces, metrics and the record of every run.
- **[CLI](cli.md)** — `run`, `resume`, `runs` and `callback-server`, option by option.

<!-- /cards -->
