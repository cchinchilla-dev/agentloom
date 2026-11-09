---
title: "CLI reference"
description: "Every agentloom command and option, from running and validating a workflow to replaying, resuming and inspecting runs."
order: 12
section: "Running"
---

# CLI reference

`agentloom` runs, checks, replays and resumes workflows. Every command, and `agentloom` itself, takes `--help`.

| Command | What it does |
|---------|--------------|
| [`run`](#run) | Runs a workflow. |
| [`validate`](#validate) | Checks a workflow without running it, and prints its plan. |
| [`visualize`](#visualize) | Draws a workflow's DAG, as text or Mermaid. |
| [`replay`](#replay) | Runs a workflow against a recording, offline. |
| [`verify-determinism`](#verify-determinism) | Replays a recording several times and compares the results. |
| [`resume`](#resume) | Continues a run from its checkpoint. |
| [`runs`](#runs) | Lists saved checkpoints. |
| [`history`](#history) | Lists the record of every run. |
| [`callback-server`](#callback-server) | Approves and rejects paused runs over HTTP. |
| [`agent inspect`](#agent-inspect) | Prints an agent's definition as it loads. |
| [`info`](#info) | Prints the version, dependencies and configured providers. |

## Common behaviour

- **There is no `--version` or `-h`.** `agentloom info` prints the version, and `--help` the help.
- **Warnings go to stderr,** such as a retry or a parse warning. There is no option for more detail: `AGENTLOOM_LOG_LEVEL` is read but changes nothing.
- **Paths are checked:** a workflow, agent or recording that doesn't exist is a usage error, exit `2`.
- **`--json` isn't always pure JSON.** `run`, `replay` and `resume` print a first line such as `Running workflow: …` before it. `runs` and `history` print plain text when they find nothing.

| Exit code | When |
|-----------|------|
| `0` | Success, a paused run, a valid workflow, or replays that agree. |
| `1` | A run that failed, timed out or went over budget; a workflow, recording or config file that doesn't load; a failed check; a conflict between options in `replay` and `resume`. |
| `2` | A usage error: an unknown option, a missing argument, a value out of range, a conflict between options in `run`. |
| `130` | A run interrupted with Ctrl-C. |

Every run, replay and resume writes a record to `agentloom_runs/<run_id>.json`, or under `AGENTLOOM_RUNS_DIR`: see [`history`](#history).

## `run`

```bash
agentloom run [OPTIONS] WORKFLOW_PATH
```

| Option | What it does |
|--------|--------------|
| `-s, --state KEY=VALUE` | Sets a top-level state key, once per key. JSON-looking values, such as `[1, 2]`, `5` or `true`, are parsed; others stay strings. See [The initial state](state-and-templates.md#the-initial-state). |
| `-p, --provider NAME` | Sets `config.provider`: the provider tried first among those that serve the model, unless a config file sets up the providers. See [Providers](providers.md#which-provider-answers). |
| `-m, --model NAME` | Sets `config.model`. A step's own `model` wins over it. |
| `-b, --budget USD` | Sets `config.budget_usd`. |
| `--config FILE` | The [config file](providers.md#config-file) whose `providers:` block sets up the providers. By default, the file `AGENTLOOM_CONFIG` names, then `./agentloom.yaml`. |
| `--stream` | Prints model replies as they arrive. See [Streaming](llm-calls.md#streaming). |
| `--json` | Prints the whole result as JSON. |
| `--checkpoint` | Saves a checkpoint when the run ends, and prints its `Run ID`. See [Checkpoints and resume](reliability.md#checkpoints-and-resume). |
| `--checkpoint-dir DIR` | Where checkpoints go. `.agentloom/checkpoints` by default. |
| `--mock-responses FILE` | Answers from a recording with the `mock` provider; a step it lacks gets `Mock response`, with a warning. |
| `--record FILE` | Records every provider call to `FILE`, for [`replay`](#replay). |
| `--record-tools` | With `--record`, records the results of the model's tool calls too. |
| `--lite` | Runs without OpenTelemetry, even when the `observability` extra is installed. |

The summary shows the status, duration, tokens and cost of the run and of each step, and the commands to resume a paused run. With `--json`, the result has `status`, `step_results`, `final_state`, `total_tokens`, `total_cost_usd`, `error` and `run_id`.

- `--record` needs a live provider: it refuses a workflow with `provider: mock`, and `--mock-responses`. These conflicts exit with `2`.
- `--mock-responses` with a file that doesn't exist isn't an error: every step gets `Mock response`, and the run succeeds.
- A missing API key isn't caught before the run: the first step fails with `No provider registered for model …`, after its retries.

## `validate`

```bash
agentloom validate WORKFLOW_PATH
```

For `examples/03_router_workflow.yaml`:

```text
Workflow 'customer-support-router' is valid.
  Version:  1.0
  Steps:    5
  Layers:   3
  Provider: openai
  Model:    gpt-4o-mini
  Budget:   $0.05

  Execution plan:
    Layer 0: classify
    Layer 1: route
    Layer 2: handle_billing, handle_general, handle_technical (parallel)
```

It loads the file as `run` does, agent files included, and checks every field, dependency, router target and subworkflow path, and that there is no cycle. Steps that can run in parallel and write the same `output` key get a warning. An error prints `Validation FAILED:` and exits with `1`.

It doesn't check what only a run finds: an inline subworkflow's steps, a Pydantic schema's import, tool names, prompt templates, keys or recordings. See [What validate checks](workflow-files.md#what-validate-checks).

## `visualize`

```bash
agentloom visualize [--format ascii|mermaid] WORKFLOW_PATH
```

`ascii`, the default, prints one row per layer, with a box per step, such as `[IF/ELSE: route]`. `mermaid` prints a Mermaid flowchart inside a ` ```mermaid ` fence, ready for a Markdown file. Both go to stdout.

Router edges in the Mermaid output carry the condition's first 20 characters, so two conditions that start alike look the same. For exact labels, Graphviz DOT or PNML, use the [Graph API](graph-api.md).

## `replay`

```bash
agentloom replay [OPTIONS] WORKFLOW_PATH --recording FILE
```

Runs a workflow against a recording made with `run --record`, without calling any provider. See [Testing & replay](testing-and-replay.md).

| Option | What it does |
|--------|--------------|
| `-r, --recording FILE` | The recording. Required. |
| `-s, --state KEY=VALUE` | As in `run`. |
| `--json` | As in `run`; with a cut, it also lists each call and where it came from. |
| `--allow-default-fallback` | A step that the recording lacks gets `Mock response`, instead of failing. |
| `--tools live\|replay\|verify` | Model tool calls: `live` runs them (the default), `replay` serves the results recorded with `--record-tools`, and `verify` runs them and fails when a result differs. `tool` steps always run. |
| `--live-from STEP` | Runs that step, and every step after it, against the live providers. |
| `--force-live STEP` | Runs that step live, once per option. |
| `--record FILE` | With a cut, records this run to a new file. |
| `--record-tools` | With `--record`, records tool results too. |
| `--config FILE` | As in `run`, for the live providers of a cut. |
| `--observability` | Turns OpenTelemetry on, which `replay` leaves off. |

- **Replay is strict.** A step the recording lacks fails the run. A changed prompt fails too, for entries that carry a `request_hash`, as `run --record` writes them; older recordings without one replay a changed prompt with its recorded answer.
- **With `--allow-default-fallback`,** a step in the recording gets its recorded answer, even if its prompt changed.
- **A cut serves the recording through the live providers,** so their keys must be set, for the steps before the cut as well as after it. Steps before the cut replay until a call misses, and go live from there.
- Conflicting options, such as `--record` without a cut, exit with `1`.

## `verify-determinism`

```bash
agentloom verify-determinism [OPTIONS] WORKFLOW_PATH --recording FILE
```

Replays the recording `--runs` times and compares the runs: the status, each step's status, output and error, the final state and the event log. When the recording carries an [event log](testing-and-replay.md#inter-agent-events), the replayed one must match it too.

| Option | What it does |
|--------|--------------|
| `-r, --recording FILE` | The recording. Required. |
| `--runs N` | How many replays, at least 2. `2` by default. |
| `-s, --state KEY=VALUE` | As in `run`. |
| `--tools live\|replay\|verify` | As in `replay`. With `live`, tools run once per replay. |
| `--events-out FILE` | Writes the first replay's event log as JSON Lines. |

It prints `Deterministic: 2 replays of … give the same result` and exits with `0`, or prints the first difference as a diff and exits with `1`. It doesn't check that the replays succeed: replays that fail the same way agree, when the recording has no event log.

## `resume`

```bash
agentloom resume [OPTIONS] RUN_ID
```

Continues a run from its checkpoint. [Checkpoints and resume](reliability.md#checkpoints-and-resume) covers what runs again.

| Option | What it does |
|--------|--------------|
| `--approve`, `--reject` | The decision for the paused [approval gate](steps.md#approval_gate). |
| `--checkpoint-dir DIR` | Where the checkpoint is. `.agentloom/checkpoints` by default. |
| `-p, --provider NAME`, `-m, --model NAME`, `--config FILE` | As in `run`. |
| `--stream`, `--json`, `--lite` | As in `run`. |

- A run whose workflow says `provider: mock` resumes against its recording.
- `resume` has no `--budget` or `--state`.
- An unknown run, `--approve` with `--reject`, or a decision for a run that isn't paused exits with `1`.
- A decision for a gate that [expired](steps.md#when-nobody-decides) is ignored, and `resume` says so: the gate takes its `on_timeout`.

## `runs`

```bash
agentloom runs [--checkpoint-dir DIR] [--json]
```

Lists the checkpoints, with each run's id, workflow, status and last update, in UTC, and when a paused run's gate [expires](steps.md#when-nobody-decides). Subworkflows have rows of their own, as `<run_id>.<step_id>`. `--json` prints the checkpoints whole, state and definition included.

## `history`

```bash
agentloom history [OPTIONS]
```

Lists the record of every run that ended, checkpointed or not, newest first.

| Option | What it does |
|--------|--------------|
| `--runs-dir DIR` | Where the records are. `AGENTLOOM_RUNS_DIR`, else `./agentloom_runs`. |
| `--workflow NAME` | Only that workflow. |
| `--provider NAME` | Only runs that used that provider, by name alone: `anthropic`, not `anthropic/claude-…`. |
| `--since DATE`, `--until DATE` | Only runs from `--since`, and before `--until`. A date is midnight UTC; a time without an offset is local time. |
| `--min-cost USD`, `--max-cost USD` | Only runs within that cost. |
| `--limit N` | At most N runs. `20` by default. |
| `--json` | The records whole. |

A record holds the run's id, time, workflow, a hash of its definition, status, the providers and models it used, cost, tokens, steps run, duration and error. Records are never deleted, and a resumed run replaces the record of its first part. A replay records the provider of the recording, not `mock`. [Observability](observability.md) covers the record.

## `callback-server`

```bash
agentloom callback-server [OPTIONS]
```

An HTTP server that resumes paused runs from their checkpoints, for approvals that come from another system.

| Option | What it does |
|--------|--------------|
| `--host HOST` | The address to listen on. `0.0.0.0` by default. |
| `--port PORT` | The port. `8642` by default; `0` picks a free one. |
| `--token TOKEN` | A shared secret that requests must send in the `X-AgentLoom-Token` header. |
| `--checkpoint-dir DIR` | Where the checkpoints are. `.agentloom/checkpoints` by default. |
| `--config FILE` | As in `run`, for the runs it resumes. The server reads it when it starts, and stops if it doesn't load. |
| `--lite` | As in `run`. |

| Endpoint | What it does |
|----------|--------------|
| `GET /pending` | Lists the paused runs, with `expires_at` for a gate that has `timeout_seconds`. |
| `POST /approve/<run_id>` | Approves the run's gate and resumes it. |
| `POST /reject/<run_id>` | Rejects it and resumes it. |
| `POST /webhook` | Receives an approval gate's notification and prints it. |

- `/approve` and `/reject` answer `202` and then resume the run. The outcome reaches the checkpoint and the run's record, not the caller. An unknown run answers `404`, a run that isn't paused `409`, and a config file that stopped loading after the server started `500`.
- A decision for a run the server is already resuming answers `409`.
- A decision for a gate that [expired](steps.md#when-nobody-decides) answers `202` with `expired_at`: the gate takes its `on_timeout`.
- The server resumes each run whose gate expired, without a request: it looks every 30 seconds, and at the next expiry it knows of.
- With `--token`, every endpoint but `/webhook` answers `401` without the right header. A server on an address other than loopback, without a token, prints a warning: anyone who reaches it can decide. The token is a command-line argument, visible to other users of the machine in its process list.
- Runs with `provider: mock` can't resume here: the server sets up only live providers. Use `agentloom resume`.
- A gate's webhook can't reach a server on `localhost` or a private address unless the workflow sets `sandbox.allow_internal_webhook_targets: true`. See [The webhook](steps.md#the-webhook).

## `agent inspect`

```bash
agentloom agent inspect [--json] AGENT_PATH
```

Prints an [agent](agents.md#inspecting-an-agent)'s definition as it loads, without running it: its model and prices, system prompt, tools with their parameters, the agents it calls, its loop policy, output schema and constraints. A file that isn't a valid agent prints the errors and exits with `1`.

## `info`

```bash
agentloom info
```

Prints the AgentLoom version, the Python version and platform, the core dependencies and the `observability` extra, and the providers: OpenAI, Anthropic and Google as configured when their key is set, and Ollama's address. It never prints a key.

## What's next

<!-- cards -->

- **[Testing & replay](testing-and-replay.md)** — Recordings, replay, cuts and fault injection.
- **[Reliability](reliability.md)** — Exit statuses, checkpoints and resuming runs.
- **[Providers](providers.md)** — The keys and endpoints `run` reads.

<!-- /cards -->
