---
title: "Observability"
description: "The OpenTelemetry traces and metrics AgentLoom sends, the bundled Grafana dashboard, the run history records and the logs."
order: 16
section: "Operations"
---

# Observability

With the `observability` extra installed, a run sends OpenTelemetry traces and metrics to a collector: a span per step and per provider call, and counters and histograms of runs, steps, tokens and cost. Every run, with or without the extra, also writes a history record.

## Quick start

```bash
cd deploy
docker compose up -d
docker compose run --rm agentloom run /workflows/40_fault_replay.yaml
```

The stack runs an OTel Collector, Jaeger, Prometheus and Grafana, and the run replays a recording, with no key needed. Then open:

- **Grafana** on `http://localhost:3000`, dashboard `AgentLoom`;
- **Jaeger** on `http://localhost:16686`, service `agentloom`;
- **Prometheus** on `http://localhost:9090`.

A run on your machine reports to the same stack: with `pip install "agentloom[observability]"`, `agentloom run examples/40_fault_replay.yaml`, from a clone of the repository, sends to `http://localhost:4317`, the collector's port. [Deployment](deployment.md#docker-compose) describes each service of the stack.

## Turning it on

The extra installs the OpenTelemetry API, SDK and OTLP exporters, and `prometheus-client`. The default Docker image includes it; `production-lite` doesn't.

| Command | Sends traces and metrics |
|---------|--------------------------|
| `run`, `resume`, `callback-server` | When the extra is installed, unless `--lite` is passed. |
| `replay` | Only with `--observability`. |
| `verify-determinism` | Never. |
| Python | Only with an `observer`; see [From Python](#from-python). |

Without the extra nothing is sent, and nothing says so: `agentloom info` shows whether OpenTelemetry is installed.

- **The endpoint** is `OTEL_EXPORTER_OTLP_ENDPOINT`, `http://localhost:4317` by default, for traces and metrics alike; AgentLoom reads no other OpenTelemetry variable. The Docker image sets it to `http://otel-collector:4317`.
- **The protocol** is OTLP over gRPC only: an HTTP endpoint, such as port 4318, doesn't work.
- **Traces and metrics** carry `service.name` `agentloom`. **Metrics** are pushed every 5 seconds and when the run ends. No AgentLoom process serves `/metrics`: a short-lived command can't be scraped, so the collector exposes the metrics to Prometheus.

A run whose export fails still succeeds.

## Traces

A run produces this tree of spans:

```text
workflow:<workflow name>
├── step:<step id>
│   ├── chat <model>              one per provider attempt
│   ├── embeddings <model>        the same, for an embed step
│   ├── execute_tool <tool>       one per tool call, a model's or a tool step's
│   └── workflow:<child>          a subworkflow, with its own steps
└── step:<step id>
quality:<target>                  a trace of its own, from result.annotate()
```

- **Retries and fallbacks** are sibling `chat` spans under their step. `agentloom.provider.attempt` numbers the fallback candidates, so every retry with a single provider carries `0`. An attempt that an open circuit skips has no span.
- **`chat` names a model call and `embeddings` an `embed` step's call,** with the model the step asked for.
- **Tool spans** last no time: they are written once the tool returned, with its duration in `tool.duration_ms`, beside the `chat` spans. An agent used as a tool has its `chat` spans under the step, not under its tool span.
- **Skipped steps** have no span.
- **A resumed run** starts a new trace, with the same `workflow.run_id`.
- **Errors are span statuses:** a failed step or attempt has status `ERROR`, so Jaeger's error filter finds it, and `step.error`, `error.type` and `agentloom.provider.attempt_outcome` say why. A step cancelled with its layer isn't an error.
- **A call cut short** by a timeout or a cancel still ends its span, with status `ERROR` and the cancellation's class, such as `CancelledError`, in `error.type`.

### Span attributes

**`workflow:<name>`**: `workflow.name` and `workflow.run_id`, then at the end `workflow.status`, `workflow.duration_ms`, `workflow.total_tokens` and `workflow.total_cost_usd`. A run id is 12 hex characters; a subworkflow's is `<parent run id>.<step id>`.

**`step:<id>`**:

| Attribute | What it holds |
|-----------|---------------|
| `step.id`, `step.type`, `step.stream`, `workflow.run_id` | Set when the step starts. |
| `step.status`, `step.duration_ms`, `step.cost_usd` | Set when it ends. The duration is the last attempt's. |
| `step.error` | Why it failed, timed out or broke the budget. |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.reasoning.output_tokens` | The step's tokens, when it succeeded. |
| `gen_ai.request.model`, `gen_ai.provider.name` | The model the step asked for and the provider that answered, when it succeeded. On a replay, the recorded provider. |
| `gen_ai.operation.name` | `chat` for an `llm_call`, `invoke_agent` for an `agent`. |
| `gen_ai.response.finish_reasons` | The provider's finish reason, as it sent it. |
| `gen_ai.response.time_to_first_chunk` | For a streamed step, in seconds. |
| `agentloom.prompt.hash`, `.length_chars`, `.template_id`, `.template_vars` | The rendered prompt's hash and length, `<workflow>:<step>`, and the state keys it used. |
| `agentloom.conversation.turn_count`, `.token_count`, `.trimmed_messages` | For a step with a conversation. |
| `gen_ai.agent.name`, `agent.iterations`, `agent.tool_calls`, `agent.cost_usd`, `agent.output_valid` | For an `agent` step. |
| `agentloom.approval_gate.decision` | `pending`, `approved`, `rejected` or `timeout`. |
| `agentloom.webhook.status`, `agentloom.webhook.latency_s` | A gate's notification: `success`, `failed`, `timeout` or `sandbox_blocked`. |
| `agentloom.mock.matched_by`, `agentloom.mock.step_id`, `agentloom.mock.fault_type` | How a recording answered. See [Testing & replay](testing-and-replay.md#metrics). |
| `agentloom.recording.provider`, `.model`, `.latency_s` | A call recorded with `--record`. |

**`chat <model>`, `embeddings <model>`**: `gen_ai.operation.name` (`chat` or `embeddings`), `gen_ai.provider.name` (the provider called, `mock` on a replay), `gen_ai.request.model`, `gen_ai.request.temperature` and `gen_ai.request.max_tokens` when set, `gen_ai.request.stream` and `agentloom.provider.attempt`; at the end, `gen_ai.response.model`, the token counts, `gen_ai.response.finish_reasons` and `agentloom.provider.attempt_outcome` (`ok` or `error`), with `error.type`, the error's class, on a failure.

**`execute_tool <name>`**: `tool.name`, `tool.call_id`, `tool.duration_ms`, `tool.success`, and hashes of the arguments and the result, `tool.args_hash` and `tool.result_hash`. The arguments and the result themselves never reach the trace.

**`quality:<target>`**: `agentloom.quality.score`, `.source`, `.target` and `agentloom.quality.metadata.<key>` for each metadata value, from `result.annotate(target, quality_score=…, source=…, **metadata)` on the result of a run that had an observer.

Provider names follow the OpenTelemetry registry: `google` is `gcp.gemini`.

### Span events

- **`agentloom.prompt.captured`**, with `config.capture_prompts: true`: the rendered `prompt` and `system_prompt` of each attempt of an `llm_call`, on its step span. With a [redaction policy](state-and-templates.md#redaction), they are rendered again with the redacted values; without one, they are the text the provider got. Attachments, earlier turns and the reply aren't captured, and `agentloom.prompt.hash` always hashes the text the provider got.
- **`agentloom.conversation.turn`**, always, after each turn of a step with a conversation: its turn and token counts, and the messages trimmed.

## Metrics reference

The collector turns metric names into Prometheus names: dots become underscores, a metric in seconds gains `_seconds`, and a histogram becomes `_bucket`, `_sum` and `_count` series.

### Runs and steps

| Metric | Type | Labels | Counts |
|--------|------|--------|--------|
| `agentloom_workflow_runs_total` | Counter | `workflow`, `status` | Runs, each resume included. A subworkflow is a step of its parent's run and isn't counted. |
| `agentloom_workflow_duration_seconds` | Histogram | `workflow`, `status` | Run durations. |
| `agentloom_cost_usd_total` | Counter | `workflow`, `provider` (`total`) | Cost of each run, its subworkflows' included. |
| `agentloom_step_executions_total` | Counter | `step_type`, `status`, `stream` | Steps that ended, whatever their status; skipped steps aren't counted. |
| `agentloom_step_duration_seconds` | Histogram | `step_type`, `stream` | The last attempt of each step: retries and their waits aren't included. |
| `agentloom_attachments_total` | Counter | `step_type` | Attachments sent. |

### Providers and tokens

| Metric | Type | Labels | Counts |
|--------|------|--------|--------|
| `agentloom_provider_calls_total` | Counter | `provider`, `model`, `stream` | Provider attempts, failed ones included. `model` is the model that answered. |
| `agentloom_provider_errors_total` | Counter | `provider`, `error_type` | Failed attempts, by error class, such as `RateLimitError`. |
| `gen_ai.client.operation.duration` | Histogram, s | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `stream` | Provider call latency. |
| `gen_ai.client.token.usage` | Histogram | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.token.type` | Tokens per step attempt, by type: `input`, `output`, and `reasoning` when there are any. On a replay, the recorded provider. |
| `agentloom_stream_responses_total` | Counter | `provider`, `model` | Streamed steps that succeeded. |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram, s | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` | Time to a streamed step's first chunk. |
| `agentloom_embedding_calls_total` | Counter | `provider`, `model` | `embed` steps that succeeded. |
| `agentloom_embedding_dimensions` | Histogram | `provider`, `model` | The length of the vectors returned. |

`gen_ai.operation.name` is `embeddings` for the calls and tokens of an `embed` step, and `chat` for the others.

### Tools, agents and conversations

| Metric | Type | Labels | Counts |
|--------|------|--------|--------|
| `agentloom_tool_calls_total` | Counter | `tool_name`, `status` | Tool calls by a model, replayed ones included, and by `tool` steps. |
| `agentloom_tool_call_duration_seconds` | Histogram | `tool_name` | Their durations. |
| `agentloom_agent_invocations_total` | Counter | `name`, `status` | Agent turns, agents used as tools included. |
| `agentloom_agent_iterations` | Histogram | `name` | Model calls per turn. |
| `agentloom_conversation_message_count` | Histogram | `conversation_key` | Messages in the conversation after each turn. |
| `agentloom_conversation_trims_total` | Counter | `policy`, `conversation_key` | Turns that trimmed messages. |

### Gates, webhooks and replay

| Metric | Type | Labels | Counts |
|--------|------|--------|--------|
| `agentloom_approval_gates_total` | Counter | `decision`, `workflow` | `pending` when a gate pauses, then the decision when the run resumes, or `timeout` when it expires with no `on_timeout`. |
| `agentloom_webhook_deliveries_total` | Counter | `status`, `workflow` | Gate notifications, once each, however many attempts they took. |
| `agentloom_webhook_latency_seconds` | Histogram | `status` | Their durations. |
| `agentloom_mock_replays_total` | Counter | `workflow`, `matched_by` | Model calls a recording answered; embeddings aren't counted. |
| `agentloom_mock_faults_total` | Counter | `workflow`, `fault_type` | Faults a recording fired. |
| `agentloom_recording_captures_total` | Counter | `provider`, `model` | Calls recorded with `--record`. |
| `agentloom_recording_latency_seconds` | Histogram | `provider`, `model` | Their durations. |

### Gauges

| Metric | Labels | Value |
|--------|--------|-------|
| `agentloom_circuit_breaker_state` | `provider` | `0` closed, `1` open, `2` half open. |
| `agentloom_budget_remaining_usd` | `workflow` | What the budget has left, for runs with `budget_usd`. |

- **The circuit breaker gauge** is reported on each provider call and when a circuit changes state. A circuit left open by a process that has ended stays on the dashboard until the bundled collector drops its series, after 24 hours.
- **The budget gauge** updates each time the run is charged, so a run that breaks its budget reports `0`.
- `AGENTLOOM_METRICS_MAX_KEYS`, 1024 by default, caps how many providers and workflows each gauge tracks.

### Reading the numbers

- **Histograms have explicit buckets,** the same over OTLP and with `prometheus-client`: `agentloom_step_duration_seconds` ends at 60 seconds, for instance, and `gen_ai.client.operation.duration` at 30. `agentloom_workflow_duration_seconds` and `gen_ai.client.token.usage` keep the SDK's default buckets, whose first one ends at 5.
- **Each `agentloom` process exports series of its own,** told apart by a `service_instance_id` label, so totals are sums over series, as the dashboard's are. The bundled collector stops exporting a series 24 hours after its process last reported: those totals cover about the last day of runs, while [run history](#run-history) counts every run.
- There are no metrics for rate-limit waits, routers, checkpoints or the sandbox.

## The Grafana dashboard

`deploy/grafana/dashboards/agentloom.json` reads the metrics above from Prometheus, and recent traces from Jaeger. Its variables filter by `workflow`, `provider`, `step_type` and `stream`.

| Row | Panels |
|-----|--------|
| Overview | Runs, success rate, failures, tokens, reasoning tokens, estimated cost, providers, step types. |
| Workflow Performance | Runs by status; duration percentiles by workflow. |
| Step Analysis | Step executions; duration percentiles by step type. |
| Token Economics | Tokens by provider and model; input, output and reasoning. |
| Provider Performance | Latency percentiles; calls. |
| Provider Reliability | Errors; availability, as one minus errors over calls since the counters started; circuit breaker state. |
| Detailed Breakdown | Tables of runs and provider calls. |
| Cost Analysis | Cost over time and per workflow; cost, tokens and duration per run; budget remaining. |
| Multi-modal | Attachments by step type. |
| Human-in-the-Loop | Approval decisions; webhook deliveries, success rate and latency. |
| Mock & Replay | Replays by how they matched, the share answered from the recording, recorded calls and their latency, injected faults. |
| Streaming | Streamed responses, time to first token, streamed against whole responses. |
| Agents, Tools & Embeddings | Tool calls and latency; agent turns and model calls per turn; conversation length and trims; embedding calls and dimensions. |
| Traces | The latest traces, linked to Jaeger. |

The Human-in-the-Loop, Mock & Replay, Streaming and Agents, Tools & Embeddings rows start collapsed.

- **Two selectors filter by provider.** `Provider` takes its values from the token metrics, which name providers as the registry does (`gcp.gemini`), and as recorded on a replay; it filters the token, latency and time-to-first-token panels. `Provider (calls)` uses AgentLoom's names (`google`, `mock`) and filters the call, error, circuit, streaming and embedding panels.

## Run history

Every run that isn't interrupted writes `<run id>.json` to `AGENTLOOM_RUNS_DIR`, or `./agentloom_runs`, with or without the extra:

```json
{
  "_schema_version": 1,
  "run_id": "3f9c2a7b1e04",
  "timestamp": "2026-11-08T20:28:01.556153+00:00",
  "agentloom_version": "0.5.0",
  "python_version": "3.12.7",
  "workflow_name": "fault-replay",
  "workflow_hash": "9b1d…",
  "status": "success",
  "providers_used": ["mock/mock-model"],
  "total_cost_usd": 0.0,
  "total_tokens": 56,
  "steps_executed": 2,
  "duration_ms": 421.3,
  "error": null
}
```

The record holds no state and no prompts. [`agentloom history`](cli.md#history) lists and filters the records. A record that can't be written is skipped: with the default directory, silently; with `AGENTLOOM_RUNS_DIR` set, with a warning.

## Logs

By default, warnings and errors print on stderr as plain lines, such as `Step 'draft' failed (attempt 1/3), retrying in 0.2s: …`, and lower levels don't show.

`AGENTLOOM_LOG_LEVEL` sets the level, `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, and `agentloom --log-level LEVEL COMMAND` overrides it. `AGENTLOOM_LOG_FORMAT` sets the format, `json` or `text`. With any of them set, the `agentloom` logs go to stderr, at `INFO` and as JSON unless told otherwise. Any other value is a usage error, exit `2`. Stdout never gets a log line, so `--json` stays parseable. A config file can't set them: `log_level` or `log_format` in it stops the load.

From Python, `setup_logging(level="INFO", format="json")`, from `agentloom.observability`, sends the `agentloom` logs to stderr, one JSON object per line, with `timestamp`, `level`, `logger` and `message`.

## From Python

A Python program sends nothing unless it passes an observer:

```python
from agentloom.core.engine import WorkflowEngine
from agentloom.observability import WorkflowObserver
from agentloom.observability.metrics import MetricsManager
from agentloom.observability.tracing import TracingManager

observer = WorkflowObserver(
    tracing=TracingManager(endpoint="http://localhost:4317"),
    metrics=MetricsManager(endpoint="http://localhost:4317"),
)
engine = WorkflowEngine(workflow, provider_gateway=gateway, observer=observer)
try:
    result = await engine.run()
finally:
    observer.shutdown()
```

The managers set the process's global OpenTelemetry tracer and meter providers, and `shutdown()` flushes and shuts them down. Pass the observer to `WorkflowEngine` rather than setting `engine.observer` later: only the constructor connects it to the gateway, which reports the provider spans and metrics.

**An observer of your own** implements the hooks it needs, and one it lacks is skipped. `ObserverProtocol`, in `agentloom.core.protocols`, declares the six a run always calls: `on_workflow_start`, `on_workflow_end`, `on_step_start`, `on_step_end`, `on_tokens` and `on_stream_response`. The others are for provider calls, tools, agents, conversations, gates and replays. Subclass `NoopObserver`, from `agentloom.observability.noop`, which has them all, and accept `**kwargs`, as hooks may gain arguments.

**A hook that raises** doesn't affect the run: the step goes on, and its provider call isn't repeated. The engine and the gateway log a warning.

## What's next

<!-- cards -->

- **[Deployment](deployment.md)** — The Compose stack, and exporting from Kubernetes.
- **[Reliability](reliability.md)** — Budgets, circuit breakers and what the metrics count.
- **[Testing & replay](testing-and-replay.md)** — Replays, and the metrics they report.

<!-- /cards -->
