# Testing & Replay

AgentLoom ships two providers designed for **offline, deterministic** execution of workflows: `RecordingProvider` captures real LLM responses to disk, and `MockProvider` replays them. Together they enable reproducible tests, CI without API keys, and statistical evaluation without paying per-run.

## When to use which

| Goal | Use |
|---|---|
| Reproducible unit/integration tests | `MockProvider` with a committed JSON fixture |
| CI runs without API keys / network | `MockProvider` |
| Capturing a real run to replay later | `RecordingProvider` wrapping the real provider |
| Statistical eval over a fixed response set | `MockProvider` with `latency_model: replay` |
| Debugging a production incident offline | Record in prod → replay locally |
| Re-running only the tail of a recorded run | `agentloom replay --live-from <step>` |
| Checking that a recording replays the same every time | `agentloom verify-determinism` |

## Recording a run

Wrap any real provider and capture every completion to a JSON file:

```bash
agentloom run workflow.yaml --record recordings/run1.json
```

The file is flushed **per call**, so a crashed workflow still leaves a partial recording. Re-running against the same path keeps the entries it does not touch; a step recorded again replaces its old entry.

The captured format is directly loadable by `MockProvider`:

```json
{
  "_version": 2,
  "summarize": {
    "content": "The article argues that...",
    "model": "claude-sonnet-4-20250514",
    "provider": "anthropic",
    "usage": {"prompt_tokens": 412, "completion_tokens": 88, "total_tokens": 500},
    "cost_usd": 0.00264,
    "latency_ms": 1843.2,
    "finish_reason": "stop",
    "request_hash": "9f2c41…",
    "meta": {"temperature": 0.2, "request_id": "req_011C…", "received_at": "2026-10-24T12:00:01Z"}
  }
}
```

Entries sit at the top level alongside the `_version` metadata key, keyed by the step's `step_id` when available, or by the request hash (SHA-256 of model + temperature + max_tokens + extra + serialized messages) otherwise. Streaming responses are keyed under the same hash as the equivalent `complete()` call and persist the joined chunk content in the same entry shape, so a recording captures both modes uniformly.

A step called more than once in one run (the turns of a `tools:` loop, or a step inside a loop) is stored as a list of such entries in call order. A turn in which the model asked for tools carries `tool_calls`, each with `id`, `name` and `arguments`. When the engine retries a step, the recorder drops the turns of the failed attempt, those of the steps of a subworkflow that ran inside it included, so the recording holds the attempt that finished.

An entry also keeps `meta`, the [response metadata](providers.md#response-metadata) that the provider reported: the sampling parameters sent, the provider's identifiers, its rate-limit headers and when the response arrived. Only the fields set are written. When the engine retried the step, `meta.attempt` holds the attempt number of the step that made the call; a step inside a subworkflow counts its own attempts. An entry with nothing to report has no `meta`. The request hash leaves `meta` out, so recordings with and without it replay the same, and the mock provider serves it back on `ProviderResponse.meta`.

!!! info "v1 ↔ v2 compatibility"
    The reader treats any top-level key starting with `_` as metadata, so v1 recordings (0.4.x, no underscore-prefixed keys) load against 0.5.0+ without errors. Recordings keyed by `step_id` continue to replay unchanged. v1 recordings keyed only by the **legacy messages-only request hash** will not match under 0.5.0+, because the fallback hash inputs now include `model`, `temperature`, `max_tokens`, and `extra` — regenerate them against 0.5.0+ to replay. New recordings are always written with `_version: 2`.

## Replaying a run

Two ways, with **different strictness**:

```bash
# Dedicated subcommand — strict by default, observability off
agentloom replay workflow.yaml --recording recordings/run1.json

# Via run --mock-responses — lenient (one-line warning on a miss)
agentloom run workflow.yaml --mock-responses recordings/run1.json
```

Every `llm_call` step resolves from the JSON. No network, no API key, no cost. Latency is simulated according to `latency_model`.

### Strict mode

`agentloom replay` runs the mock provider in **strict mode**. A request that misses the recording — or matches a step whose prompt, system prompt, model, or tools spec drifted since capture — raises `RecordingMismatchError` and fails the workflow. This is what makes a green replay trustworthy in CI: pre-0.5.0 the mock provider silently fell through to the placeholder `"Mock response"`, so a replay could pass while answering a prompt the recording no longer matched.

Each recorded entry carries a `request_hash`; on replay the mock compares it against the live request and refuses on a mismatch. A mismatch fails its step on the first attempt: the step's retries do not apply, the gateway does not try another provider, and a subworkflow around the step is not run again. Recording files are also validated against the canonical schema when loaded — a malformed file, or one with `_version` below 2, is rejected at startup with a re-record hint.

Pass `--allow-default-fallback` to restore the lenient behaviour (placeholder response on a miss):

```bash
agentloom replay workflow.yaml --recording rec.json --allow-default-fallback
```

`agentloom run --provider mock` and `--mock-responses` stay lenient — a miss returns the placeholder default and logs a one-line warning, so ad-hoc mock runs stay frictionless.

### YAML-configured MockProvider

You can also declare the mock provider directly in the workflow config, avoiding CLI flags entirely:

```yaml
config:
  provider: mock
  model: mock-model
  responses_file: fixtures/responses.json
  latency_model: constant   # constant | normal | replay
  latency_ms: 0
```

Run it like any normal workflow — `agentloom run workflow.yaml`. Useful for committed fixtures and CI without plumbing flags through shell scripts.

## Recording tool results

Replay serves the recorded LLM responses, but the tools of a `tools:` loop run again. To replay a run whose tools read the clock, call an API or touch files, record their results too:

```bash
agentloom run workflow.yaml --record recordings/run1.json --record-tools
agentloom replay workflow.yaml --recording recordings/run1.json --tools replay
```

`--record-tools` stores each tool call under `_tools`, keyed by step, in call order:

```json
{
  "_version": 2,
  "_tools": {
    "ask": [
      {"name": "http_request", "arguments": {"url": "https://example.com"}, "result": "<html>…", "success": true}
    ]
  }
}
```

`agentloom replay --tools` takes one of three modes:

| Mode | Tools | A call missing from the recording, or a different result |
|---|---|---|
| `live` (default) | run | not checked here; the next turn's `request_hash` catches a changed result |
| `replay` | do not run: the recorded result is served | a missing call raises `RecordingMismatchError` with the step, the call number and the tool |
| `verify` | run, and their result and `success` flag are compared | raises `RecordingMismatchError` with the call and the hashes of both results |

A call matches the next recorded call of its step with the same tool name and arguments, so a tool called twice gets its two results in order. When the engine retries a step, the failed attempt leaves no trace, nor do the steps of a subworkflow that ran inside it: recording keeps the calls of the attempt that finished, and replay serves again the calls that the failed attempt took. With `--allow-default-fallback`, a missing call runs live and a different result logs a warning. Steps of `type: tool` always run live. An [agent called as a tool](workflow-yaml.md#agent) is not on the tape: its turn replays from the recording in every mode, and the tape holds its own tools' calls under its key.

## Replaying up to a cut point

To run only the tail of a workflow after a fix, keep the recorded answers of the steps before it and run the rest live:

```bash
agentloom replay workflow.yaml --recording recordings/run1.json --live-from summarize
agentloom replay workflow.yaml --recording recordings/run1.json --force-live classify --force-live route
```

- `--live-from <step>` runs that step and every step that depends on it live. A step that does not depend on it, a parallel branch included, is still served from the recording.
- `--force-live <step>` (repeatable) runs a step live even when the recording matches it.
- Any other step is served from the recording while its requests match. From its first call that misses (its prompt, model or tools changed, or it makes more calls than were recorded) it runs live, and its dependents, whose prompts change with it, follow on their own.

Both flags name top-level steps; the steps of a subworkflow follow their parent. The live part uses the workflow's own providers, so their keys must be set. With `--tools replay`, a replayed step gets its recorded tool results and a live step runs its tools; with `--tools live`, a replayed step whose tools return something new misses its next turn and runs live from there. `--allow-default-fallback` does not combine with a cut.

The run lists every call with where it came from and why, and reports a replayed step that left recorded turns unused:

```text
Calls (replay up to the cut, live after it):
  classify #1: replay (hit)
  summarize #1: live (forced)
  report #1: live (miss)
```

With `--json`, the list is under `calls`, one `{step_id, call, source, reason}` per call. From Python, wrap each live provider with `gateway.wrap_providers(lambda p: ReplayOrLiveProvider(p, cut))`, where `cut = ReplayCut(MockProvider(responses_file=..., strict=True), forced)` (`agentloom.providers.replay_or_live`), and read `cut.calls` after the run.

To keep replaying after the fix, record the cut run to a new file:

```bash
agentloom replay workflow.yaml --recording recordings/run1.json --live-from summarize \
  --record recordings/run2.json --record-tools --tools replay
```

`run2.json` holds every call of that run, in order within each step: a call served from `run1.json` is copied as recorded there (its latency and usage included), and a live call is captured as `agentloom run --record` captures it, so a step that left the recording midway keeps its replayed turns and its live ones. With `--record-tools`, the tool results go under `_tools` too, the replayed ones as served and the live ones as they ran. `run2.json` replays strictly on its own, and `run1.json` is never written: `--record` refuses the source file, a link to it included. `--record` needs `--live-from` or `--force-live`, since a replay without a cut calls nothing.

## Injecting faults

A recording can fail a call on purpose, the same way on every run. Any entry, or any turn of a step's list, takes a `fault` instead of an answer:

```json
{
  "draft": [
    {"fault": {"type": "http_error", "status_code": 429, "retry_after_s": 1}},
    {"fault": {"type": "timeout", "after_ms": 500}},
    {"content": "Release 1.4 ships Thursday."}
  ]
}
```

| `type` | Parameters | Raises |
|---|---|---|
| `http_error` | `status_code` (400–599), `retry_after_s`, `body` | `RateLimitError` for 429, with `retry_after_s`; `ProviderError` with the status otherwise |
| `timeout` | `after_ms` (default 0) | `ProviderError` (`ReadTimeout`) after waiting `after_ms` |
| `connection_reset` | — | `ProviderError` (`ConnectError`) |
| `stream_truncate` | `after_chunks` | a stream yields the entry's first `after_chunks` words of `content`, then `ProviderError` (`ReadError`); a plain call fails at once |

Each fault raises what the provider adapters raise for that failure, so the gateway's circuit breaker and fallback and the step's `retry` policy take the path a real failure takes: the example above fails twice and the third attempt answers when the step allows two retries (`examples/40_fault_replay.yaml`). A malformed fault (unknown `type`, missing or extra field) is refused when the recording loads.

A fault turn you add carries no `request_hash`: under strict replay, it fires when the cursor reaches it in a recorded list, so a recorded run can be made to fail at a chosen call without re-recording it. A retried step starts its loop again: a fault before a step's first turn fires once and the retry gets the answer, while a fault between two turns of a tool loop fires on every attempt. A fault in a step that a cut replays fires too. A fault added to a recorded turn keeps that turn's `request_hash`, so strict replay fires it only for the request it was recorded for and refuses any other. A faulted call is still a replay lookup in `agentloom_mock_replays_total`; each fault also counts in `agentloom_mock_faults_total{workflow, fault_type}` and marks the step span with `agentloom.mock.fault_type`. A response with `finish_reason: "length"` and the usage you want needs no fault: write it as an ordinary entry.

## Inter-agent events

`agentloom run --record` also keeps the run's inter-agent event log, under `_inter_agent_events` in the recording, and so does `agentloom replay --record` for a cut replay. The log says which agent handed what to which, in causal order. An agent is a step's `speaker`, the agent an `agent` step runs, or the step itself; an agent called as a tool is one tool call of its caller and logs nothing of its own. A message is a dependency edge: when a step starts, the log gets a `task_sent` from each step it depends on that finished, then the step's `task_received`; then its tool calls and returns (`tool_call`, `tool_return`), from a `tools:` loop or a step of `type: tool`; and, when the step succeeds, its `response_sent` and a `state_transition` for each state key it wrote.

```json
{
  "type": "task_sent",
  "event_id": "ee69945c-c818-5c09-af1a-a294fdac075d",
  "agent_id": "researcher",
  "step_id": "research",
  "recipient_agent_id": "writer",
  "parent_event_ids": ["2643aefc-2236-5582-92c4-8bf12d3b0c4f"],
  "payload_hash": "b3efdde5240daba9",
  "timestamp_ns": 14419290988958
}
```

| Field | Meaning |
|---|---|
| `event_id` | UUID v5 of the step's path, the event type and its position in the step, so two runs that do the same thing write the same ids |
| `agent_id`, `step_id` | Who acted, and the step's path: a subworkflow's steps read `parent.child`. A `task_sent` carries the sending step |
| `parent_event_ids` | The events this one causally follows, sorted; empty for a top-level step that got nothing from another step |
| `payload_hash` | SHA-256 of canonical JSON, 16-char prefix: a tool call's arguments, a return's result and `success`, the step's output, the value written. A `task_sent` hashes the sender's output and state writes, a `task_received` what it received |
| `timestamp_ns` | Monotonic, per process; informational, never compared |
| `recipient_agent_id`, `tool_name`, `key` | On `task_sent`, on the tool events and on `state_transition` |

The log is sorted by the dependency graph, not by time: by causal depth (one more than the deepest parent's), then step, event type and `event_id`, so the order in which parallel steps finish does not change it. When the engine retries a step, the events of the failed attempt are dropped, those of a subworkflow that ran inside it included, as the recorder drops its turns. A subworkflow's first steps follow the subworkflow step's `task_received`, and its `response_sent` follows theirs.

The event models and the JSON Schema (draft 2020-12) of one event are in `agentloom.contracts.experimental`: `InterAgentEvent`, a union of the six models discriminated by `type`, `SCHEMA_PATH` and `SCHEMA_URI`. Recordings without `_inter_agent_events` load and replay unchanged.

### Verifying a recording

```bash
agentloom verify-determinism workflow.yaml --recording recordings/run1.json --runs 3 --tools replay
```

`verify-determinism` replays the recording strictly `--runs` times (default 2), each with a fresh engine, and compares what each replay gave: the run's status, each step's status, output, error and error classification, the final state and the event log, leaving out run ids, durations and timestamps. It then compares the event log with the recorded one, when the recording has it. On any difference it exits with code 1 and prints a unified diff to stderr; otherwise it prints `Deterministic: …` and exits 0. `--tools` works as in `replay`: with `live`, the default, a tool whose answer changes shows up as a difference, and `--tools replay` serves the results recorded with `--record-tools`. `--events-out <file>` writes the log of the first replay as JSONL, one event per line, and `--state key=value` sets state as in `run`.

## Latency models

`MockProvider` supports three modes:

| Model | Behavior | Use case |
|---|---|---|
| `constant` (default) | Sleeps `latency_ms` on every call | Fast tests |
| `normal` | Gaussian around `latency_ms` with σ = 10%, seedable via `seed=` | Jitter simulation |
| `replay` | Uses the recorded `latency_ms` from the fixture | Faithful reproduction for perf eval |

```python
from agentloom.providers.mock import MockProvider

mock = MockProvider(
    responses_file="recordings/run1.json",
    latency_model="replay",
)
```

## Key resolution

`MockProvider` resolves each call in this order:

1. **`step_id` match** — if the caller passes `step_id=` and that key exists in the fixture. A list entry is served turn by turn: lenient replay walks a cursor and repeats the last turn; strict replay (`agentloom replay`) serves only a turn whose `request_hash` matches the request, trying the next turn first. A retried step therefore replays from its own turns, and a changed prompt or tool output fails instead of replaying stale data. Hand-written lists without `request_hash` keep the cursor.
2. **Prompt hash match** — SHA-256 of the serialized messages list
3. **Default response** — returns `default_response` (defaults to `"Mock response"`) with zero cost/usage

Call metadata is recorded on `provider.calls` and exposed via observer hooks (see [Observability](#observability)).

## Programmatic use

```python
from agentloom.providers.mock import MockProvider
from agentloom.providers.recorder import RecordingProvider
from agentloom.providers.anthropic import AnthropicProvider

# Record
real = AnthropicProvider(api_key=...)
recorder = RecordingProvider(real, output_path="fixture.json")
# ... use recorder like any provider ...
await recorder.close()  # flushes

# Replay
mock = MockProvider(responses_file="fixture.json", latency_model="replay")
```

## Testing patterns

### Pattern 1 — committed fixtures

Commit a JSON fixture under `tests/fixtures/` and use `MockProvider` directly:

```python
async def test_summarization_workflow():
    provider = MockProvider(responses_file="tests/fixtures/summary.json")
    gateway = ProviderGateway()
    gateway.register(provider)
    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway)
    result = await engine.run()
    assert result.state["summary"] == "expected output"
```

### Pattern 2 — record once, replay forever

Run the workflow against a real provider once with `--record`, then commit the JSON and switch CI to `--mock-responses`. Re-record when prompts change.

### Pattern 3 — statistical evaluation

Record N variations of a prompt against a real provider, then run your evaluator against the fixture in a tight loop — no rate limits, no cost, deterministic scoring.

## Observability

Both providers emit observer events that bridge to Prometheus and OTel when the `[observability]` extra is installed:

| Metric | Labels | Meaning |
|---|---|---|
| `agentloom_mock_replays_total` | `workflow`, `matched_by` (`step_id` / `prompt_hash` / `default`) | Replay hit counter |
| `agentloom_recording_captures_total` | `provider`, `model` | Captured call counter |
| `agentloom_recording_latency_seconds` | `provider`, `model` | Histogram of real-provider latency while recording |

OTel span attributes: `mock.matched_by`, `mock.step_id`, `recording.provider`, `recording.model`, `recording.latency_s`.

The stock Grafana dashboard includes a **Mock & Replay** row with:

- Total replays (stat)
- Hit ratio (`step_id` + `prompt_hash`) vs defaults
- Captures by provider
- Captured real-provider latency p50 / p95

## Concurrency & merge semantics

If a workflow registers multiple providers each wrapped with `RecordingProvider` pointing to the same file (e.g. primary + fallback via the gateway), each recorder holds only what it captured and `_flush()` merges that over the on-disk content, so recorders never revert each other's keys.

Within one process, every writer of the file (each `RecordingProvider` and the tool recorder of `--record-tools`) holds a shared lock around its read-merge-write, so parallel steps never drop each other's entries. Writers in separate processes are not coordinated: they can still lose updates, so give each process its own file.

## Limitations & gotchas

- **A run is the recorder's lifetime.** Reusing one gateway for several `engine.run()` calls appends every run's calls to the same list of turns. If the gateway falls back to another provider in the middle of a loop, each provider's recorder keeps only its own turns and the last one to flush wins: re-record without the fallback.
- **Tools run for real on replay** unless they were recorded with `--record-tools` and replayed with `--tools replay`. Strict replay compares each turn's request, which includes the earlier tool results, so a live tool whose output changes (network, clock, files) fails strict replay from that turn on. Steps of `type: tool` always run live.
- **The event log covers one process.** A run paused at an approval gate keeps the log of the part before the pause; `agentloom resume` records nothing, so the resumed part has no log.
- **Prompt hashes are sensitive to message formatting.** A trailing space or reordered tool-result block changes the hash. If replay misses, inspect `provider.calls` to see what was tried.
- **Recordings are not encrypted.** Treat them as you would any captured LLM output — scrub PII before committing.
- **No built-in TTL.** Recordings live on disk until deleted. For hot-path caching across distributed workers, a pluggable store backend is on the roadmap.

## See also

- [Providers](providers.md) — gateway, circuit breaker, fallback
- [Observability](observability.md) — metrics, dashboards, OTel setup
- [`scripts/validate_mock_provider.py`](https://github.com/cchinchilla-dev/agentloom) — end-to-end validation script
