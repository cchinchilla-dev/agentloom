<p align="center">
  <a href="https://github.com/cchinchilla-dev/agentloom">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cchinchilla-dev/agentloom/main/docs/images/header_dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/cchinchilla-dev/agentloom/main/docs/images/header_white.png">
      <img src="https://raw.githubusercontent.com/cchinchilla-dev/agentloom/main/docs/images/header_white.png" alt="AgentLoom">
    </picture>
  </a>
</p>
<p align="center">
  <strong>Deterministic LLM workflow orchestration with native observability, resilience, and cost control.</strong>
</p>

<p align="center">
  <a href="https://github.com/cchinchilla-dev/agentloom/actions/workflows/ci.yml"><img src="https://github.com/cchinchilla-dev/agentloom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/github/cchinchilla-dev/agentloom"><img src="https://codecov.io/github/cchinchilla-dev/agentloom/graph/badge.svg?token=BRJ4XY6AG7" alt="Coverage"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

---

## Table of Contents

- [Why AgentLoom?](#why-agentloom)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Workflow Definition (YAML)](#workflow-definition-yaml)
- [Python DSL](#python-dsl)
- [Observability](#observability)
- [Deploy](#deploy)
- [Why not autonomous agents?](#why-not-autonomous-agents)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Why AgentLoom?

Existing frameworks (LangGraph, CrewAI, AutoGen) treat observability and resilience as afterthoughts. AgentLoom is built from the ground up for production: circuit breakers, rate limiting, cost tracking, and OpenTelemetry traces are part of the core design — not plugins.

| Feature | LangGraph | CrewAI | AutoGen | AgentLoom |
|---|---|---|---|---|
| Workflow definition | Python API | Decorators | Agent chat | **YAML + Python DSL** |
| Streaming | Via API | No | Via API | **YAML config + CLI flag** |
| Multi-modal input | Via messages | No | Via messages | **YAML attachments** |
| Observability | LangSmith ($) | Minimal | Minimal | **OTel + Prometheus + Grafana** |
| Circuit breaker | No | No | No | **Built-in** |
| Cost tracking | No | No | No | **Native with budgets** |
| Multi-provider fallback | Manual | No | No | **Automatic** |
| Dependencies | Heavy | Medium | Medium | **Minimal** |

## Quick Start

```bash
# Install
pip install agentloom

# Install with observability (OTel + Prometheus)
pip install agentloom[all]

# Run a workflow
export OPENAI_API_KEY=sk-...
agentloom run examples/01_simple_qa.yaml

# Or with Ollama (free, local)
agentloom run examples/01_simple_qa.yaml --provider ollama --model phi4

# Validate a workflow
agentloom validate examples/03_router_workflow.yaml

# Visualize the DAG
agentloom visualize examples/03_router_workflow.yaml
```

## Architecture

```
+-----------------------------------------------------+
|                   CLI / Python API                  |
+-----------------------------------------------------+
|                   Workflow Engine                   |
|  +-----------+  +-----------+  +---------------+    |
|  |DAG Parser |  | Scheduler |  | State Manager |    |
|  |& Validator|  |  (anyio)  |  |  (Pydantic)   |    |
|  +-----------+  +-----------+  +---------------+    |
+-----------------------------------------------------+
|                   Step Executors                    |
|  +--------+ +---------+ +------+ +------------+     |
|  |LLM Call| |Tool Exec| |Router| | Subworkflow|     |
|  +--------+ +---------+ +------+ +------------+     |
+-----------------------------------------------------+
|                  Provider Gateway                   |
|  +-----------------------------------------------+  |
|  | OpenAI | Anthropic | Google | Ollama           | |
|  | + Fallback | Circuit Breaker | Rate Limiter    | |
|  +-----------------------------------------------+  |
+-----------------------------------------------------+
|              Observability (optional)               |
|  +------------+  +----------+  +----------+         |
|  | OTel Traces|  |Prometheus|  | JSON Logs|         |
|  +------------+  +----------+  +----------+         |
+-----------------------------------------------------+
```

## Workflow Definition (YAML)

```yaml
name: classify-and-respond
config:
  provider: openai
  model: gpt-4o-mini
  budget_usd: 0.50

state:
  user_input: ""

steps:
  - id: classify
    type: llm_call
    system_prompt: "Classify as: question, complaint, or request."
    prompt: "Classify: {state.user_input}"
    output: classification

  - id: route
    type: router
    depends_on: [classify]
    conditions:
      - expression: "state.classification == 'question'"
        target: answer
    default: general_response

  - id: answer
    type: llm_call
    depends_on: [route]
    prompt: "Answer: {state.user_input}"
    output: response

  - id: general_response
    type: llm_call
    depends_on: [route]
    prompt: "Help with: {state.user_input}"
    output: response
```

### Multi-modal Input

LLM call steps support image, PDF, and audio attachments. Sources can be URLs,
local file paths, or inline base64. URL fetching includes SSRF protection and
respects sandbox domain restrictions.

```yaml
steps:
  - id: analyze
    type: llm_call
    prompt: "Describe what you see in this image."
    attachments:
      - type: image
        source: "{state.image_url}"
    output: description
```

| Type | OpenAI | Anthropic | Google | Ollama |
|---|---|---|---|---|
| `image` | yes | yes | yes | yes |
| `pdf` | -- | yes | yes | -- |
| `audio` | yes | -- | yes | -- |

### Streaming

Stream LLM responses token-by-token for real-time output. Enable at the
workflow level, per-step, or via CLI flag.

```yaml
config:
  stream: true    # workflow-level default

steps:
  - id: answer
    type: llm_call
    stream: true  # per-step override
    prompt: "Answer: {question}"
```

```bash
# CLI flag (overrides workflow config)
agentloom run workflow.yaml --stream
```

All providers support streaming: OpenAI (SSE), Anthropic (SSE), Google (SSE),
Ollama (NDJSON). Token usage, cost, and time-to-first-token are tracked
normally.

## Python DSL

```python
from agentloom.core.dsl import workflow

wf = (
    workflow("my-workflow", provider="ollama", model="phi4")
    .set_state(question="What is Python?")
    .add_llm_step("answer", prompt="Answer: {question}", output="answer")
    .build()
)
```

## Observability

Every workflow step emits OpenTelemetry traces and Prometheus metrics out of the box. No external SaaS required — the full stack runs alongside your workloads.

```bash
# Start Prometheus + Grafana + Jaeger
cd deploy && docker compose up -d

# Access:
#   Grafana:    http://localhost:3000
#   Prometheus: http://localhost:9090
#   Jaeger:     http://localhost:16686
```

See [Dashboard Documentation](deploy/DASHBOARD.md) for panel descriptions, metrics reference, and troubleshooting.

## Deploy

AgentLoom is designed to run anywhere — from a single Docker container on your laptop to a fully orchestrated Kubernetes cluster with GitOps and observability. Every deployment method is production-hardened with non-root containers, read-only filesystems, Pod Security Standards enforcement, and network policies.

The CLI processes a workflow and exits. There is no long-running server, no HTTP API, and no persistent connections. This makes Kubernetes **Jobs** (not Deployments) the correct primitive: finite execution, automatic retries, scheduled runs via CronJobs, and clean resource isolation per workflow.

### Docker

The fastest way to run a workflow. The multi-stage Dockerfile produces a minimal image (~120MB) with a non-root user and read-only filesystem.

```bash
docker build -t agentloom .
docker run --rm -e OPENAI_API_KEY=sk-... \
  -v ./examples:/workflows:ro \
  agentloom run /workflows/01_simple_qa.yaml
```

### Kubernetes (Kustomize)

Plain YAML manifests organized with Kustomize overlays. Three environments are provided, each with progressively stricter security and resource controls:

- **dev**: minimal resources, no NetworkPolicy, `latest` tag for fast iteration.
- **staging**: moderate resources, NetworkPolicy enabled, CI image tag.
- **production**: strict NetworkPolicy (no Ollama egress), `activeDeadlineSeconds` hard timeout, pinned image version.

```bash
kubectl apply -k deploy/k8s/overlays/dev
kubectl logs job/agentloom-workflow -n agentloom
```

### Helm

The recommended method for teams that need parameterized deployments. The chart packages all Kubernetes resources with built-in input validation — deploying without a workflow definition fails at render time, not at runtime.

```bash
helm install agentloom deploy/helm/agentloom \
  -n agentloom --create-namespace \
  --set workflow.definition="$(cat examples/01_simple_qa.yaml)" \
  --set provider.existingSecret=my-secret
```

Supports Job and CronJob modes, configurable NetworkPolicies, ResourceQuotas, and optional namespace creation with PSS labels.

### Terraform

Provisions a complete local development environment in one command: a kind cluster with agentloom, plus the full observability stack (OTel Collector, Prometheus, Grafana, Jaeger). Set `enable_observability = false` for a lightweight setup without metrics and traces.

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
```

After `apply`, Grafana is available at `localhost:3000`, Prometheus at `localhost:9090`, and Jaeger at `localhost:16686` — all pre-configured with agentloom dashboards and datasources.

### ArgoCD

GitOps deployment with automated sync, self-heal, and retry policies. ArgoCD watches the Helm chart in the repository and syncs changes automatically. The Application CRD handles Kubernetes Job immutability via `Replace=true` and `ignoreDifferences` on selectors.

```bash
kubectl apply -f deploy/argocd/application.yaml
```

See [deploy/INFRASTRUCTURE.md](deploy/INFRASTRUCTURE.md) for the full deployment guide, security hardening details, Helm chart reference, and CI/CD pipeline documentation.

## Why not autonomous agents?

Most LLM frameworks focus on autonomous agents: self-directed reasoning, multi-agent delegation, unbounded tool loops. This works for demos and open-ended research, but breaks down in production where you need predictable costs, debuggable failures, and SLA compliance.

AgentLoom is **not** an autonomous agent framework. There are no self-directed agents, no unbounded loops, no emergent behavior. It is a **deterministic workflow orchestrator** that uses LLMs as execution steps within a declared DAG.

The difference matters:

- **You define the DAG, not the LLM.** Steps, dependencies, and routing logic are declared upfront in YAML. The model generates text within a step — it does not decide what runs next. Routers use explicit boolean conditions, not LLM judgement.
- **Observability is not optional.** Every step emits OpenTelemetry traces and Prometheus metrics. You can see exactly what ran, how long it took, and how much it cost. Autonomous agents are notoriously hard to debug; a static DAG with full tracing is not.
- **Cost is bounded.** Budget limits, circuit breakers, and rate limiters are first-class. A runaway autonomous agent can burn through an API budget in minutes. A workflow with `budget_usd: 0.50` cannot.
- **Fallback is structural.** If OpenAI is down, the gateway falls back to Anthropic or Ollama automatically. This is a routing decision at the infrastructure level, not an agent "choosing" a provider.

Autonomous agent frameworks solve a real problem — open-ended tasks where the execution path cannot be known in advance. But most LLM workloads in production are not open-ended. They are pipelines: classify, enrich, route, generate, validate. For those, you want predictability and control, not autonomy. That is what AgentLoom is for.

## Development

```bash
uv sync --group dev --all-extras   # install with all extras
uv run pytest                       # ~5s
uv run ruff check src/ tests/      # lint (ruff replaces flake8+isort)
uv run ruff format src/ tests/     # autoformat
uv run mypy src/                   # strict type checking
```

Pre-commit hooks run ruff automatically on staged files — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT
