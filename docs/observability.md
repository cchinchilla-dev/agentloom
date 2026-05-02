# Observability

Every workflow step emits OpenTelemetry traces and Prometheus metrics out of the box. No external SaaS required — the full stack runs alongside your workloads.

## Quick start

```bash
# Start the observability stack
cd deploy && docker compose up -d

# Run a workflow (metrics are emitted automatically)
agentloom run examples/01_simple_qa.yaml

# Access dashboards
open http://localhost:3000    # Grafana (admin/admin)
open http://localhost:9090    # Prometheus
open http://localhost:16686   # Jaeger
```

## Stack architecture

```
CLI (agentloom run)
  |
  +-- OTel SDK --> OTel Collector (:4317 gRPC)
  |                    |
  |                    +-- Prometheus exporter (:8889) --> Prometheus (:9090)
  |                    +-- OTLP exporter --> Jaeger (:16686)
  |
  +-- Observer --> MetricsManager --> OTel gauges / counters / histograms
```

Each CLI invocation is an ephemeral process that creates its own `MeterProvider`. The OTel Collector aggregates metrics and exposes them to Prometheus with a 30-minute expiration window (`metric_expiration: 30m`), so data remains visible after the CLI exits.

| Component | Port | Purpose |
|-----------|------|---------|
| OTel Collector | 4317 (gRPC), 8889 (Prometheus) | Receives OTel data, exports to Prometheus + Jaeger |
| Prometheus | 9090 | Time-series storage, PromQL queries |
| Jaeger | 16686 | Distributed tracing UI |
| Grafana | 3000 | Dashboard visualization |

---

## Grafana dashboard

The dashboard is auto-provisioned at startup. Navigate to **Dashboards > AgentLoom** or go directly to `http://localhost:3000/d/agentloom-main`.

### Dashboard variables

Use the dropdown selectors at the top of the dashboard to filter:

| Variable | Label | Default | Description |
|----------|-------|---------|-------------|
| `$workflow` | Workflow | `.*` | Filter panels by workflow name |
| `$provider` | Provider | `.*` | Filter panels by provider |

### Row 1 — Overview

Seven stat panels showing high-level aggregates:

| Panel | Type | What it shows |
|-------|------|---------------|
| **Total Runs** | stat | Total workflow executions across all workflows |
| **Success Rate** | gauge | Percentage of successful runs (green >= 95%, orange >= 80%, red < 80%) |
| **Failed** | stat | Count of failed workflow runs |
| **Total Tokens** | stat | Sum of all tokens consumed (input + output) |
| **Est. Cost** | stat | Estimated total cost in USD |
| **Providers** | stat | Number of distinct providers that have handled requests |
| **Step Types** | stat | Number of distinct step types executed |

### Row 2 — Workflow Performance

| Panel | Type | What it shows |
|-------|------|---------------|
| **Workflow Runs** | timeseries | Cumulative success vs failed runs over time |
| **Workflow Duration (p50/p95/p99)** | timeseries | Latency percentiles per workflow |

### Row 3 — Step Analysis

| Panel | Type | What it shows |
|-------|------|---------------|
| **Step Executions** | timeseries | Cumulative step success vs failure count over time |
| **Step Duration (p50/p95/p99)** | timeseries | Per-step-type latency percentiles |

### Row 4 — Token Economics

| Panel | Type | What it shows |
|-------|------|---------------|
| **Tokens** | timeseries | Cumulative input vs output token count per provider/model |
| **Tokens by Provider** | bar gauge | Horizontal bars showing total tokens per provider/model |
| **Token Split** | pie chart | Ratio of prompt tokens to completion tokens |

### Row 5 — Provider Performance

| Panel | Type | What it shows |
|-------|------|---------------|
| **Provider Latency (p50/p95/p99)** | timeseries | Per-provider response time percentiles |
| **Provider Calls** | timeseries | Cumulative call count per provider |

### Row 6 — Provider Reliability

| Panel | Type | What it shows |
|-------|------|---------------|
| **Provider Errors** | timeseries | Cumulative error count per provider |
| **Provider Availability** | gauge | Uptime percentage over last hour (green >= 99%, orange >= 95%, red < 95%) |
| **Circuit Breaker** | stat | Current circuit breaker state per provider |

??? info "Circuit breaker state values"

    | Value | Label | Color | Meaning |
    |-------|-------|-------|---------|
    | 0 | CLOSED | Green | Normal — requests pass through |
    | 1 | OPEN | Red | Tripped — requests rejected immediately |
    | 2 | HALF-OPEN | Yellow | Recovery probe — one test request allowed |

    State transitions: CLOSED -> OPEN after 5 consecutive failures. OPEN -> HALF-OPEN after 60s timeout. HALF-OPEN -> CLOSED on success, back to OPEN on failure.

### Row 7 — Detailed Breakdown

| Panel | Type | What it shows |
|-------|------|---------------|
| **Workflow Runs Summary** | table | Per-workflow, per-status run counts |
| **Provider Call Details** | table | Per-provider, per-model call counts |

### Row 8 — Cost Analysis

| Panel | Type | What it shows |
|-------|------|---------------|
| **Cumulative Cost** | timeseries | Running total cost in USD |
| **Token Cost** | timeseries | Estimated cost based on cumulative token volume per provider/model |
| **Budget Remaining** | timeseries | Remaining budget per workflow (green line) |
| **Avg Cost/Run** | stat | Mean cost per workflow execution |
| **Avg Tokens/Run** | stat | Mean token consumption per workflow execution |
| **Avg Duration/Run** | stat | Mean wall-clock time per workflow execution |
| **Token Input/Output Ratio** | stat | Ratio of output tokens to input tokens |

### Row 9 — Multi-modal

| Panel | Type | What it shows |
|-------|------|---------------|
| **Attachments** | timeseries | Attachment count by type (image, pdf, audio) |

### Row 10 — Streaming

| Panel | Type | What it shows |
|-------|------|---------------|
| **Stream Responses** | timeseries | Cumulative streaming response count |
| **Time to First Token (p50/p95/p99)** | timeseries | TTFT latency percentiles |

---

## Metrics reference

All metrics are prefixed with `agentloom_`.

### Workflow metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `workflow_runs_total` | counter | workflow, status | Workflow execution count |
| `workflow_duration_seconds` | histogram | workflow | End-to-end workflow latency |
| `cost_usd_total` | counter | — | Estimated USD cost |
| `budget_remaining_usd` | gauge | workflow | Remaining budget per workflow |

### Step metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `step_executions_total` | counter | step_type, status, stream | Step execution count |
| `step_duration_seconds` | histogram | step_type, stream | Per-step latency |

### Provider metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `provider_calls_total` | counter | provider, model, stream | Provider API call count |
| `provider_latency_seconds` | histogram | provider, model, stream | Provider response latency |
| `provider_errors_total` | counter | provider, error_type | Provider error count |
| `circuit_breaker_state` | gauge | provider | Circuit breaker state (0/1/2) |

### Token metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tokens_total` | counter | provider, model, direction | Token usage (`input` / `output` / `reasoning`) |
| `attachments_total` | counter | step_type | Attachment count by step type |

### Streaming metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `stream_responses_total` | counter | provider, model | Streaming response count |
| `time_to_first_token_seconds` | histogram | provider, model | Time to first token latency |

---

## Span schema

AgentLoom emits a three-level span hierarchy: **workflow → step → provider call** (and **tool call** when a step invokes a registered tool). Every span / attribute / metric name is centralised in `agentloom.observability.schema` so downstream consumers (Grafana dashboards, AgentTest, Jaeger plugins) parse a stable contract.

### Hierarchy

```
workflow:<workflow_name>                   # AgentLoom orchestration
└── step:<step_id>                         # AgentLoom orchestration
    └── chat <model>                       # OTel GenAI inference span — one per fallback attempt
```

A failed primary provider followed by a successful fallback shows up as two sibling `chat <model>` spans under the same `step:*` parent — useful for debugging fallback latency.

### Attribute conventions

Inference spans follow the **canonical OTel GenAI registry** (May 2026 spec). Workflow / step orchestration spans use AgentLoom-specific names.

| Namespace | Source | Example |
|-----------|--------|---------|
| `gen_ai.*` | OpenTelemetry GenAI registry (canonical names) | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, `gen_ai.request.stream`, `gen_ai.response.model`, `gen_ai.response.finish_reasons` (array), `gen_ai.response.time_to_first_chunk`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.reasoning.output_tokens` |
| `error.*` | OTel general semantic conventions | `error.type` (set on errored inference spans alongside `step.error`) |
| `workflow.*` / `step.*` | AgentLoom orchestration metadata | `workflow.run_id`, `workflow.status`, `step.id`, `step.type`, `step.duration_ms`, `step.cost_usd` |
| `tool.*` | Tool-call details | `tool.name`, `tool.args_hash`, `tool.success` |
| `agentloom.*` | AgentLoom-specific (no OTel equivalent) | `agentloom.prompt.hash`, `agentloom.approval_gate.decision`, `agentloom.webhook.status`, `agentloom.recording.provider` |

### Workflow-level attributes

| Attribute | Description |
|-----------|-------------|
| `workflow.name` | Workflow identifier |
| `workflow.run_id` | Per-execution UUID — correlate Jaeger traces with checkpoints / external systems |
| `workflow.status` | Final status (`success` / `failed` / `paused` / `budget_exceeded` / `timeout`) |
| `workflow.duration_ms` | End-to-end execution time |
| `workflow.total_tokens`, `workflow.total_cost_usd` | Aggregates across all steps |

### Step-level attributes

| Attribute | Description |
|-----------|-------------|
| `step.id`, `step.type`, `step.status` | Step identification |
| `step.duration_ms`, `step.cost_usd` | Per-step latency / spend |
| `step.stream`, `step.attachments` | Streaming flag, attachment count |
| `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` | Operation type (`chat` for `llm_call`), provider (e.g. `openai`, `gcp.gemini`), model |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | Visible token counts |
| `gen_ai.usage.reasoning.output_tokens` | Chain-of-thought tokens (o-series, Gemini 2.5+ thinking) — emitted only when non-zero |
| `gen_ai.response.finish_reasons` | Array of provider-supplied stop reasons (e.g. `["stop"]`) |
| `gen_ai.response.time_to_first_chunk` | Streaming-only, in seconds |
| `agentloom.prompt.hash`, `agentloom.prompt.length_chars` | Prompt fingerprint for correlating failures with the prompt that caused them |
| `agentloom.prompt.template_id`, `agentloom.prompt.template_vars` | Template provenance |

### Inference-level attributes (provider span)

The `chat <model>` span — emitted once per fallback attempt by the gateway — carries the full set of GenAI inference attributes. A single `step:*` may have multiple sibling provider spans when fallback fires.

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | Always `chat` for `llm_call`; future operation types follow the OTel registry (`embeddings`, `execute_tool`, `invoke_agent`, …) |
| `gen_ai.provider.name` | Canonical OTel value (e.g. `openai`, `anthropic`, `gcp.gemini`) translated from AgentLoom's internal provider name |
| `gen_ai.request.model`, `gen_ai.response.model` | Requested model and the model the provider actually responded with (may differ when the provider auto-resolves a version, e.g. `gpt-4o-mini` → `gpt-4o-mini-2024-07-18`) |
| `gen_ai.request.temperature`, `gen_ai.request.max_tokens` | Sampling controls passed through to the provider |
| `gen_ai.request.stream` | `true` for streaming calls — distinguishes streaming vs non-streaming inference at the inference span level |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.reasoning.output_tokens` | Token counts |
| `gen_ai.response.finish_reasons` | Array of stop reasons |
| `gen_ai.response.time_to_first_chunk` | Streaming-only |
| `error.type` | Set on errored attempts (OTel general convention, alongside the AgentLoom-specific `step.error`) |
| `agentloom.provider.attempt`, `agentloom.provider.attempt_outcome` | Fallback attempt index (0-indexed) and outcome (`ok` / `error`) — debugging fallback behaviour |

### Capture flags

Full prompt content is **not** captured by default — size and secrets concerns. Set `config.capture_prompts: true` in the workflow YAML to opt in: each `llm_call` span then carries an `agentloom.prompt.captured` OTel event with the rendered `prompt` and `system_prompt`. Event payloads avoid the attribute-size cap and stay easy to filter at the OTel collector. Off by default; opt-in for debugging or trusted environments.

### Provider name translation

AgentLoom's internal provider names map to the canonical OTel registry values: `google` → `gcp.gemini`, others (`openai`, `anthropic`, `ollama`) match the registry as-is. Custom values (`ollama`, `mock`) ride the spec's "vendor extension" allowance. The bundled Grafana dashboard queries Prometheus metrics (not span attributes), so dashboard panels are unaffected by attribute renames.

---

## Troubleshooting

??? question "Panels show 'No data'"
    Metrics are ephemeral — each CLI run exports data, then the process exits. The OTel Collector retains metrics for 30 minutes (`metric_expiration`).

    ```bash
    # Verify metrics reach Prometheus
    curl -s 'http://localhost:9090/api/v1/query?query=agentloom_workflow_runs_total'

    # Run a workflow to generate fresh data
    agentloom run examples/01_simple_qa.yaml

    # Run the circuit breaker demo for reliability panels
    agentloom run examples/16_circuit_breaker_demo.yaml
    ```

??? question "Timeseries panels show flat lines"
    Each CLI invocation is a short-lived process that creates ephemeral counters. Timeseries panels show cumulative totals — run several workflows to see the values increase over time. Metrics expire from Prometheus after 30 minutes, so run workflows periodically.

??? question "Circuit Breaker shows OPEN unexpectedly"
    If you ran `16_circuit_breaker_demo.yaml`, it intentionally trips the circuit breaker with a non-existent model. The OPEN state persists in Prometheus for 30 minutes. Run a successful workflow against the same provider to reset it, or wait for the metric to expire.

??? question "Traces not appearing in Jaeger"
    Check that the OTel Collector is running and accepting gRPC on port 4317:
    ```bash
    docker compose ps
    docker compose logs otel-collector
    ```
