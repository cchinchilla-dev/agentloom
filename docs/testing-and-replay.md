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

The file is flushed **per call**, so a crashed workflow still leaves a partial recording. Re-running against the same path accumulates entries — it never clobbers.

The captured format is directly loadable by `MockProvider`:

```json
{
  "_version": 2,
  "summarize": {
    "content": "The article argues that...",
    "model": "claude-sonnet-4-20250514",
    "usage": {"prompt_tokens": 412, "completion_tokens": 88, "total_tokens": 500},
    "cost_usd": 0.00264,
    "latency_ms": 1843.2,
    "finish_reason": "stop"
  }
}
```

Entries sit at the top level alongside the `_version` metadata key, keyed by the step's `step_id` when available, or by the request hash (SHA-256 of model + temperature + max_tokens + extra + serialized messages) otherwise. Streaming responses are keyed under the same hash as the equivalent `complete()` call and persist the joined chunk content in the same entry shape, so a recording captures both modes uniformly.

!!! info "v1 ↔ v2 compatibility"
    The reader treats any top-level key starting with `_` as metadata, so v1 recordings (0.4.x, no underscore-prefixed keys) load against 0.5.0+ without errors. Recordings keyed by `step_id` continue to replay unchanged. v1 recordings keyed only by the **legacy messages-only request hash** will not match under 0.5.0+, because the fallback hash inputs now include `model`, `temperature`, `max_tokens`, and `extra` — regenerate them against 0.5.0+ to replay. New recordings are always written with `_version: 2`.

## Replaying a run

Two equivalent ways:

```bash
# Dedicated subcommand — observability off by default, no stream/checkpoint flags
agentloom replay workflow.yaml --recording recordings/run1.json

# Or via run --mock-responses (same effect)
agentloom run workflow.yaml --mock-responses recordings/run1.json
```

Every `llm_call` step resolves from the JSON. No network, no API key, no cost. Latency is simulated according to `latency_model`.

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

1. **`step_id` match** — if the caller passes `step_id=` and that key exists in the fixture
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

If a workflow registers multiple providers each wrapped with `RecordingProvider` pointing to the same file (e.g. primary + fallback via the gateway), `_flush()` reads existing on-disk content and merges it with the in-memory buffer before writing.

This is **best-effort**, not concurrency-safe: the implementation is a read-merge-write cycle without locking, so true concurrent writers (multiple processes, or parallel flushes across tasks) can still lose updates. In practice it covers the common case — recorders inside the same run flushing sequentially per call — but if you need strict guarantees, use a single writer or serialize access externally.

## Limitations & gotchas

- **Streaming is passthrough only.** `RecordingProvider.stream()` delegates to the wrapped provider and does **not** capture tokens. Replay of streamed runs is not supported yet.
- **`step_id` must be unique within a workflow** for the step-id matching to be useful. Steps run inside a loop share the same `step_id` and will collide — use prompt-hash matching or give each iteration a distinct id.
- **Prompt hashes are sensitive to message formatting.** A trailing space or reordered tool-result block changes the hash. If replay misses, inspect `provider.calls` to see what was tried.
- **Recordings are not encrypted.** Treat them as you would any captured LLM output — scrub PII before committing.
- **No built-in TTL.** Recordings live on disk until deleted. For hot-path caching across distributed workers, a pluggable store backend is on the roadmap.

## See also

- [Providers](providers.md) — gateway, circuit breaker, fallback
- [Observability](observability.md) — metrics, dashboards, OTel setup
- [`scripts/validate_mock_provider.py`](https://github.com/cchinchilla-dev/agentloom) — end-to-end validation script
