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

A call matches the next recorded call of its step with the same tool name and arguments, so a tool called twice gets its two results in order. When the engine retries a step, the failed attempt leaves no trace, nor do the steps of a subworkflow that ran inside it: recording keeps the calls of the attempt that finished, and replay serves again the calls that the failed attempt took. With `--allow-default-fallback`, a missing call runs live and a different result logs a warning. Steps of `type: tool` always run live.

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

- **Streaming through the gateway.** `RecordingProvider.stream()` records one entry with the joined content and the tool calls, but the `StreamResponse` a step receives from the gateway carries only usage, cost, finish reason and model.
- **A run is the recorder's lifetime.** Reusing one gateway for several `engine.run()` calls appends every run's calls to the same list of turns. If the gateway falls back to another provider in the middle of a loop, each provider's recorder keeps only its own turns and the last one to flush wins: re-record without the fallback.
- **Tools run for real on replay** unless they were recorded with `--record-tools` and replayed with `--tools replay`. Strict replay compares each turn's request, which includes the earlier tool results, so a live tool whose output changes (network, clock, files) fails strict replay from that turn on. Steps of `type: tool` always run live.
- **Prompt hashes are sensitive to message formatting.** A trailing space or reordered tool-result block changes the hash. If replay misses, inspect `provider.calls` to see what was tried.
- **Recordings are not encrypted.** Treat them as you would any captured LLM output — scrub PII before committing.
- **No built-in TTL.** Recordings live on disk until deleted. For hot-path caching across distributed workers, a pluggable store backend is on the roadmap.

## See also

- [Providers](providers.md) — gateway, circuit breaker, fallback
- [Observability](observability.md) — metrics, dashboards, OTel setup
- [`scripts/validate_mock_provider.py`](https://github.com/cchinchilla-dev/agentloom) — end-to-end validation script
