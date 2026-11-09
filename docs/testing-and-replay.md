---
title: "Testing & replay"
description: "Record a run against real providers and replay it offline: strict replays, cut points, tool results, injected faults and the inter-agent event log."
order: 15
section: "Running"
---

# Testing & replay

Record a run once against real providers, then replay it as often as you need: no network, no keys and no cost, with the same answers every time. A strict replay also fails when a prompt has changed since the recording, so it works as a test.

## Which to use

| To | Use |
|----|-----|
| Try a workflow's wiring without keys | `agentloom run workflow.yaml --provider mock`: every call answers `Mock response`. |
| Replay a real run offline, failing on any change | `agentloom replay workflow.yaml -r recording.json` |
| Check that a run gives the same result every time | `agentloom verify-determinism workflow.yaml -r recording.json` |
| Run part of a workflow live and replay the rest | `agentloom replay … --live-from STEP` |
| See how a workflow handles provider failures | A recording with [fault entries](#injecting-faults). |
| Test a workflow from Python | `MockProvider(responses_file=…, strict=True)`. See [Testing from Python](#testing-from-python). |

## Recording a run

```bash
agentloom run workflow.yaml --record recordings/run1.json
```

The run calls its providers as usual and saves each answer to the file, creating its folders if needed:

- **Every model call,** fallbacks included: plain calls, streamed calls once the stream ends, and embeddings. A stream that fails or is cut short isn't recorded.
- **The run's [event log](#inter-agent-events),** under `_inter_agent_events`.
- **Tool results,** with `--record-tools`. See [Recording tool results](#recording-tool-results).

A step that was retried keeps the calls of its last attempt only. A recording made with `--stream` replays without it, and the other way round.

The file is written after every call, so a run cut short keeps the calls it made, but not its event log, which is written when the run ends. `--record` refuses a workflow whose provider is `mock`.

**Recording into an existing file** replaces the entries of the steps this run called and keeps the rest. The event log is replaced whole, so it no longer matches entries kept from an earlier run, and `verify-determinism` fails on such a file. Record to a new file when you mean to verify it.

> [!CAUTION]
> A recording holds the answers verbatim. A `state_schema` key marked `redact` is masked in checkpoints and the run history, not here: an answer that repeats a secret, and a tool's arguments and results, are saved as they were. Prompts aren't saved, only their hash. Read a recording before you commit it.

### What a recording holds

A recording is a JSON object with an entry per step:

```json
{
  "_version": 2,
  "summarize": {
    "content": "Revenue grew 12% on subscription renewals.",
    "model": "gpt-4o-mini",
    "provider": "openai",
    "usage": {"prompt_tokens": 412, "completion_tokens": 18, "total_tokens": 430, "reasoning_tokens": 0},
    "cost_usd": 0.0000726,
    "latency_ms": 812.4,
    "finish_reason": "stop",
    "request_hash": "6054bd…"
  }
}
```

| Field | What it is |
|-------|------------|
| `content` | The answer; for a streamed call, its chunks joined. |
| `model`, `provider` | What answered. A replay reports this provider, not `mock`. |
| `usage`, `cost_usd` | Tokens and cost. A replay reports them again, and `budget_usd` counts them. |
| `latency_ms` | How long the call took. Only the `replay` [latency model](#latency) reads it. |
| `finish_reason` | As the provider sent it, such as `end_turn` for Anthropic. |
| `request_hash` | A hash of the request, which strict replay checks. See [Strict replay](#strict-replay). |
| `tool_calls` | The tools the model asked for. |
| `meta` | What the call sent and got back besides the answer, such as `seed` or `request_id`, and `attempt` for a retried step. |
| `embeddings` | The vectors of an `embed` step. |

No field is required: an entry written by hand can hold `content` alone. Keys starting with `_` aren't steps: `_version`, `_tools` and `_inter_agent_events`. A file without `_version` loads; one with a version below `2` is refused.

- **Entries are keyed by step id.** A conversation's summary call is recorded under `<step>::summarize`, and the calls of an [agent used as a tool](agents.md#agents-as-tools) under `<step>::<tool>#<n>`. A subworkflow's steps use their own id, so they share an entry with a top-level step of the same id.
- **A step that calls the model several times,** in a tool loop or to re-ask for a valid structured reply, gets a list of turns, in call order.
- **A call made from Python without a step id** is keyed by a hash of its messages, model, temperature, `max_tokens` and other arguments.

## Replaying a run

```bash
agentloom replay workflow.yaml -r recordings/run1.json
```

Every model call is answered from the recording, and no provider is called, even with its key set. The model's tool calls still run, unless you replay their [recorded results](#recording-tool-results). Each replay writes a run history record, and OpenTelemetry stays off unless you pass `--observability`.

The [CLI reference](cli.md#replay) lists every option.

### Strict replay

`replay` is strict: a call the recording can't answer fails its step at once, without retries, and the run exits with `1`.

- **A step missing from the recording** fails with `No recorded response for step 'review'`.
- **A changed request** fails with `the replayed request (hash …) does not match the recording (hash …)`. Any change counts: the prompt, the state it renders, the model, the temperature, the tools or the schema.

Only entries with a `request_hash` can catch a change. `run --record` writes one on every model call, but not on embeddings: an `embed` step replays its recorded vectors whatever its input, and fails, after its retries, when the number of inputs changes. Entries written by hand, such as the recordings of examples 35 to 40, carry none either.

In a list of turns, strict replay picks the turn whose hash matches the call: a retried step starts its turns again. A list without hashes is served in order, one turn per call, and repeats its last turn once it runs out.

### Lenient replay

Lenient replay answers what it can and never fails a call: a missing step gets `Mock response`, or `{}` for a step with a `response_schema`, with a warning, and a changed request gets its recorded answer, without one. Lists are served in order, hashes ignored.

| | Missing step | Changed request | Missing file |
|---|---|---|---|
| `replay` | Fails | Fails | Exit `2` |
| `verify-determinism` | Fails | Fails | Exit `2` |
| `replay --allow-default-fallback` | Placeholder | Recorded answer | Exit `2` |
| `run --mock-responses FILE` | Placeholder | Recorded answer | Placeholders for every call |
| `provider: mock` in the workflow | Placeholder | Recorded answer | Placeholders for every call |

For CI, use `replay` or `verify-determinism`: a lenient run passes on a stale recording, and on a mistyped path.

### The `mock` provider in YAML

A workflow can replay a recording with plain `agentloom run`:

```yaml
config:
  provider: mock
  model: mock-model
  responses_file: recordings/run1.json
  latency_model: constant
  latency_ms: 0
```

- **The replay is lenient.** For a strict one, run `agentloom replay` on the same recording.
- **`responses_file` is relative to the working directory,** not to the workflow file. Run from the folder the path assumes: from anywhere else, the file isn't found, and every call answers the placeholder.
- A recording that doesn't parse fails the run with a traceback.
- Like any provider set up by `agentloom run`, the mock is held to 60 requests a minute.

Examples 35 to 41 run this way against the recordings in `recordings/`. [`examples/32_yaml_mock.yaml`](../examples/32_yaml_mock.yaml) names `recordings/byzantine.json`, which isn't in the repository: it runs on placeholders.

### Latency

Only a `provider: mock` workflow and a `MockProvider` built in Python wait before answering. `replay`, `run --mock-responses` and `verify-determinism` answer at once, whatever the workflow sets.

| `latency_model` | Wait per call |
|-----------------|---------------|
| `constant`, the default | `latency_ms`. |
| `normal` | A random wait around `latency_ms`, with a spread of 10% of it, and at least 1 ms. In YAML it can't be seeded; in Python, `seed=` makes it repeat. |
| `replay` | The entry's recorded `latency_ms`, or the configured `latency_ms` for an entry without one. |

## Recording tool results

By default a replay runs the model's tool calls again. To replay them too, record their results:

```bash
agentloom run workflow.yaml --record recordings/run1.json --record-tools
agentloom replay workflow.yaml -r recordings/run1.json --tools replay
```

`--record-tools` saves, under `_tools`, each tool call the model made, per step and in order: its name, its arguments, the result the model saw, and whether it succeeded. Only model tool calls are recorded, from an `llm_call` or `agent` step's `tools:`. A `tool` step always runs, in every mode.

| `--tools` | Tools run? | A call the recording lacks | A different result |
|-----------|------------|----------------------------|--------------------|
| `live`, the default | Yes | — | Caught only when the model's next turn no longer matches its `request_hash`. |
| `replay` | No: the recorded result is served. | Fails the step. | — |
| `verify` | Yes, and the result is compared. | Fails the step. | Fails the step. |

- **A call matches** the next recorded call of its step with the same name and arguments; recorded calls skipped on the way aren't served. A retried step starts its recorded calls again.
- **With `--allow-default-fallback`,** a missing or different call logs a warning and uses the live result.
- `verify-determinism` takes the same `--tools` option. With `live`, tools run once per replay.

## Replaying up to a cut point

Replay a run up to a step and call the providers from there, to try a new prompt or model on the end of a workflow without paying for its start:

```bash
agentloom replay workflow.yaml -r recordings/run1.json --live-from summarize
```

- **`--live-from STEP`** runs that step and every step that depends on it live. **`--force-live STEP`** runs one step live, and can be repeated.
- **Other steps replay** until one of their calls misses the recording, and run live from that call on.
- **The recording is served through the live providers,** so the workflow's providers must be set up, with their keys, for the replayed steps as well as the live ones.
- **`--record FILE`** saves the run to a new recording: replayed calls as recorded, live calls as made. With `--record-tools`, tools too.
- **`--json`** adds a `calls` list: for each call, its step, whether it was replayed or live, and why (`hit`, `miss` or `forced`).

A cut can't be combined with `--allow-default-fallback`.

## Injecting faults

An entry, or a turn of a list, can hold a `fault` instead of an answer. Every replay then fails that call the same way, which tests retries and fallbacks without waiting for a real outage. [`examples/40_fault_replay.yaml`](../examples/40_fault_replay.yaml) replays this recording:

```json
{
  "_version": 2,
  "draft": [
    {"fault": {"type": "http_error", "status_code": 429, "retry_after_s": 0.1}},
    {"fault": {"type": "http_error", "status_code": 503, "body": "upstream overloaded"}},
    {"content": "Release 1.4 is in final testing and ships Thursday."}
  ]
}
```

The first two attempts of `draft` fail, and the third gets the answer: the step's `retry` allows two retries.

| `type` | Fields | What the call does |
|--------|--------|--------------------|
| `http_error` | `status_code` (400 to 599, required), `retry_after_s`, `body` | Fails with that status, as the provider would. A 429 is a rate limit. |
| `timeout` | `after_ms` | Waits, then fails as a provider timeout. |
| `connection_reset` | | Fails as a dropped connection. |
| `stream_truncate` | `after_chunks` (required) | Streams the first `after_chunks` words of the entry's `content`, then fails. A call that doesn't stream fails at once. |

Each failure is retried or not as a real one would be: see [What is retried](reliability.md#what-is-retried). A malformed fault is refused when the recording loads.

- **A list without hashes,** as written by hand, is served in order, one turn per call, and a retry carries on from the next turn. So a retry doesn't go back to the start of a tool loop: with `[a turn that calls a tool, a fault, an answer]`, the retried step gets the answer first and never calls the tool.
- **A single entry with a fault** fails every attempt: retries can't recover.
- **In a recorded list,** with hashes, under strict replay, a fault turn fires when the call reaches it, and the retry starts the step's turns again.
- A 429 doesn't wait for its `retry_after_s`: the step's own waits apply.
- Words a `stream_truncate` fault delivered are already printed when the retry streams its answer.
- `embed` steps ignore faults.
- Faults fire in the replayed steps of a cut too, without sending them live.

## Inter-agent events

`run --record` saves the run's event log under `_inter_agent_events`: what each step received, which tools it called, what it answered and what it wrote to the state. It is the same in every run that does the same thing, so `verify-determinism` compares it across replays.

For each attempt of a step, the log has:

- a `task_sent` from each finished step it depends on, and its own `task_received`;
- a `tool_call` and a `tool_return` for each tool call, a `tool` step's included;
- when it succeeds, a `response_sent`, and a `state_transition` for each state key it wrote.

An agent used as a tool adds a `task_sent` from its caller, then its own `task_received`, tool events and `response_sent`. A subworkflow's steps are named `<step>.<child step>`. A retried step keeps the events of its last attempt only.

| Field | What it is |
|-------|------------|
| `type` | One of the six event types. |
| `event_id` | An id derived from the event's place in the run, the same in every replay. |
| `agent_id` | The step's `speaker`, its agent's name for an `agent` step, or else its id. A `task_sent` carries the sender's. |
| `step_id` | The step. |
| `parent_event_ids` | The events that caused this one. |
| `payload_hash` | A 16-character hash of what the event carried: a tool's arguments or result, a step's output, a state value. |
| `timestamp_ns` | A clock reading for this process, left out of every comparison. |
| `recipient_agent_id`, `tool_name`, `key` | On `task_sent`, tool events and `state_transition`. |

Events are ordered by their causes, then by step and type, so steps that run in parallel give the same log whichever finishes first.

From Python, `agentloom.contracts.experimental` exports the event models, `InterAgentEvent` among them, and `SCHEMA_PATH`, the event log's JSON Schema. Importing it warns once that the module may change.

### Verifying a recording

```bash
agentloom verify-determinism workflow.yaml -r recordings/run1.json --runs 3
```

It replays the recording strictly `--runs` times and compares the runs: the status, each step's status, output and error, the final state and the event log, which must also match the recorded one. `--events-out FILE` writes the first replay's log as JSON Lines. A difference prints a diff and exits with `1`.

It doesn't check that the replays succeed: replays that fail the same way agree when the recording has no event log. See the [CLI reference](cli.md#verify-determinism).

## Testing from Python

AgentLoom ships no pytest plugin or fixtures. A test builds the engine with a strict `MockProvider`:

```python
import anyio
import pytest

from agentloom.core.engine import WorkflowEngine
from agentloom.core.parser import WorkflowParser
from agentloom.core.results import WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.mock import MockProvider


@pytest.fixture(autouse=True)
def _history_in_tmp(tmp_path, monkeypatch):
    # Every run writes a history record
    monkeypatch.setenv("AGENTLOOM_RUNS_DIR", str(tmp_path / "runs"))


async def replay(workflow, recording, **state):
    wf = WorkflowParser.from_yaml(workflow)
    gateway = ProviderGateway()
    gateway.register(MockProvider(responses_file=recording, strict=True))
    engine = WorkflowEngine(
        wf,
        state_manager=StateManager(initial_state={**wf.state, **state}),
        provider_gateway=gateway,
    )
    try:
        return await engine.run()
    finally:
        await gateway.close()


def test_replays():
    result = anyio.run(replay, "workflows/report.yaml", "recordings/report.json")
    assert result.status is WorkflowStatus.SUCCESS


def test_a_changed_topic_misses_the_recording():
    result = anyio.run(lambda: replay("workflows/report.yaml", "recordings/report.json", topic="dogs"))
    assert "does not match the recording" in result.error
```

- **Pass `strict=True`.** A `MockProvider` is lenient by default, and a test then passes on `Mock response` with only a logged warning.
- **Assert on `result.final_state`,** where keys marked `redact` read as `<REDACTED:sha256=…>`: check those through a step's output instead.
- **Paths are relative to the working directory:** `responses_file`, sandbox paths and agent files. `monkeypatch.chdir` sets it for a test.
- `@pytest.mark.anyio` on an `async def` test works too, with the plugin that comes with anyio.

`MockProvider` takes `latency_model` and `latency_ms` as the YAML does, plus `seed`, which makes `normal` waits repeat, and `default_response`, the placeholder text. Its `calls` list records each call it answered and how: `step_id`, `prompt_hash` or `default`.

To drive the CLI from a test, `typer.testing.CliRunner` invokes `agentloom.cli.main.app`. With `--json`, the output starts with a `Running workflow:` line before the JSON.

## Metrics

With OpenTelemetry on, replays and recordings report:

| Metric | Labels | Counts |
|--------|--------|--------|
| `agentloom_mock_replays_total` | `workflow`, `matched_by` | Calls a recording answered, by how: `step_id`, `prompt_hash` or `default`. |
| `agentloom_mock_faults_total` | `workflow`, `fault_type` | Faults fired. |
| `agentloom_recording_captures_total` | `provider`, `model` | Calls recorded. |
| `agentloom_recording_latency_seconds` | `provider`, `model` | How long recorded calls took. |

`replay` reports them only with `--observability`, and `verify-determinism` never does. The bundled Grafana dashboard has a "Mock & Replay" row. See [Observability](observability.md).

## Limits

- Two processes recording to the same file at once can lose each other's entries.
- A changed request goes unnoticed in lenient replays, in entries without a `request_hash` and in embeddings.
- Steps that share an id across a subworkflow share an entry: a lenient replay serves them in call order.

## What's next

<!-- cards -->

- **[CLI](cli.md)** — Every option of `run --record`, `replay` and `verify-determinism`.
- **[Reliability](reliability.md)** — The retries and fallbacks that injected faults exercise.
- **[Observability](observability.md)** — Traces and metrics of live and replayed runs.

<!-- /cards -->
