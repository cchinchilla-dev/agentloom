# AgentLoom Grafana Dashboard

Observability dashboard for monitoring LLM workflow execution, provider
performance, token usage, costs, and resilience patterns.

## Quick Start

```bash
# Start the observability stack
cd deploy && docker compose up -d

# Run a workflow (metrics are emitted by default)
uv run agentloom run examples/01_simple_qa.yaml

# Open Grafana
open http://localhost:3000   # admin / admin
```

The dashboard is auto-provisioned at startup. Navigate to
**Dashboards → AgentLoom** or go directly to
`http://localhost:3000/d/agentloom-main`.

## Architecture

```
CLI (agentloom run)
  │
  ├── OTel SDK ──► OTel Collector (:4317 gRPC)
  │                    │
  │                    ├── Prometheus exporter (:8889) ──► Prometheus (:9090)
  │                    └── OTLP exporter ──► Jaeger (:16686)
  │
  └── Observer ──► MetricsManager ──► OTel gauges / counters / histograms
```

Each CLI invocation is an ephemeral process that creates its own
`MeterProvider`. The OTel Collector aggregates metrics and exposes them
to Prometheus with a 30-minute expiration window (`metric_expiration: 30m`),
so data remains visible after the CLI exits.

## Dashboard Variables

| Variable     | Label    | Default | Description                       |
|-------------|----------|---------|-----------------------------------|
| `$workflow` | Workflow | `.*`    | Filter panels by workflow name    |
| `$provider` | Provider | `.*`    | Filter panels by provider (ollama, openai, etc.) |

Use the dropdown selectors at the top of the dashboard to filter.

## Rows & Panels

### Row 1 — Overview

Seven stat panels showing high-level aggregates.

| Panel | Type | What it shows |
|-------|------|---------------|
| **Total Runs** | stat | Total workflow executions across all workflows |
| **Success Rate** | gauge | Percentage of successful runs (green ≥95%, orange ≥80%, red <80%) |
| **Failed** | stat | Count of failed workflow runs |
| **Total Tokens** | stat | Sum of all tokens consumed (input + output) |
| **Est. Cost** | stat | Estimated total cost in USD based on provider pricing |
| **Providers** | stat | Number of distinct providers that have handled requests |
| **Step Types** | stat | Number of distinct step types executed (llm_call, tool, router) |

### Row 2 — Workflow Performance

| Panel | Type | What it shows |
|-------|------|---------------|
| **Workflow Runs** | timeseries | Cumulative success vs failed runs over time |
| **Workflow Duration (p50 / p95 / p99)** | timeseries | Latency percentiles per workflow. Computed from `agentloom_workflow_duration_seconds` histogram |

### Row 3 — Step Analysis

| Panel | Type | What it shows |
|-------|------|---------------|
| **Step Executions** | timeseries | Cumulative step success vs failure count over time |
| **Step Duration (p50 / p95 / p99)** | timeseries | Per-step-type latency percentiles from `agentloom_step_duration_seconds` histogram |

### Row 4 — Token Economics

| Panel | Type | What it shows |
|-------|------|---------------|
| **Tokens** | timeseries | Cumulative input vs output token count per provider/model |
| **Tokens by Provider** | bar gauge | Horizontal bars showing total tokens per provider/model combination |
| **Token Split (input / output)** | pie chart | Ratio of prompt tokens to completion tokens |

### Row 5 — Provider Performance

| Panel | Type | What it shows |
|-------|------|---------------|
| **Provider Latency (p50 / p95 / p99)** | timeseries | Per-provider response time percentiles |
| **Provider Calls** | timeseries | Cumulative call count per provider |

### Row 6 — Provider Reliability

| Panel | Type | What it shows |
|-------|------|---------------|
| **Provider Errors** | timeseries | Cumulative error count per provider |
| **Provider Availability** | gauge | Uptime percentage over last hour (green ≥99%, orange ≥95%, red <95%) |
| **Circuit Breaker** | stat | Current circuit breaker state per provider |

Circuit breaker value mappings:

| Value | Label | Color | Meaning |
|-------|-------|-------|---------|
| 0 | CLOSED | Green | Normal — requests pass through |
| 1 | OPEN | Red | Tripped — requests rejected immediately |
| 2 | HALF-OPEN | Yellow | Recovery probe — one test request allowed |

State transitions: CLOSED → OPEN after 5 consecutive failures.
OPEN → HALF-OPEN after 60s timeout. HALF-OPEN → CLOSED on success,
back to OPEN on failure.

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
| **Avg Cost / Run** | stat | Mean cost per workflow execution |
| **Avg Tokens / Run** | stat | Mean token consumption per workflow execution |
| **Avg Duration / Run** | stat | Mean wall-clock time per workflow execution |
| **Token Input/Output Ratio** | stat | Ratio of output tokens to input tokens |

## Metrics Reference

All metrics are prefixed with `agentloom_`.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `workflow_runs_total` | counter | workflow, status | Workflow execution count |
| `workflow_duration_seconds` | histogram | workflow | End-to-end workflow latency |
| `step_executions_total` | counter | step_type, status | Step execution count |
| `step_duration_seconds` | histogram | step_type | Per-step latency |
| `tokens_total` | counter | provider, model, direction | Token usage (input/output) |
| `cost_usd_total` | counter | — | Estimated USD cost |
| `provider_calls_total` | counter | provider, model | Provider API call count |
| `provider_latency_seconds` | histogram | provider, model | Provider response latency |
| `provider_errors_total` | counter | provider, error_type | Provider error count |
| `circuit_breaker_state` | gauge | provider | Circuit breaker state (0/1/2) |

## Troubleshooting

### Panels show "No data"

Metrics are ephemeral — each CLI run exports data, then the process exits.
The OTel Collector retains metrics for 30 minutes (`metric_expiration`).

```bash
# Verify metrics reach Prometheus
curl -s 'http://localhost:9090/api/v1/query?query=agentloom_workflow_runs_total'

# Run a workflow to generate fresh data
uv run agentloom run examples/01_simple_qa.yaml

# Run the circuit breaker demo for reliability panels
uv run agentloom run examples/16_circuit_breaker_demo.yaml
```

### Timeseries panels show flat lines

Each CLI invocation is a short-lived process that creates ephemeral
counters. Timeseries panels show cumulative totals — run several
workflows to see the values increase over time. Metrics expire from
Prometheus after 30 minutes (`metric_expiration`), so run workflows
periodically to keep data visible.

### Circuit Breaker shows OPEN unexpectedly

If you ran `16_circuit_breaker_demo.yaml`, it intentionally trips the
circuit breaker with a non-existent model. The OPEN state persists in
Prometheus for 30 minutes. Run a successful workflow against the same
provider to reset it, or wait for the metric to expire.

### Traces not appearing in Jaeger

Check that the OTel Collector is running and accepting gRPC on port 4317:
```bash
docker compose ps
docker compose logs otel-collector
```

## Stack Components

| Service | Port | Purpose |
|---------|------|---------|
| OTel Collector | 4317 (gRPC), 8889 (Prometheus) | Receives OTel data, exports to Prometheus + Jaeger |
| Prometheus | 9090 | Time-series storage, PromQL queries |
| Jaeger | 16686 | Distributed tracing UI |
| Grafana | 3000 | Dashboard visualization |
