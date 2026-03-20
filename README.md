# AgentLoom

**Production-ready agentic workflow orchestrator** with native observability, resilience, and cost control.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Why AgentLoom?](#why-agentloom)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Workflow Definition (YAML)](#workflow-definition-yaml)
- [Python DSL](#python-dsl)
- [Observability Stack](#observability-stack)
- [Why not autonomous agents?](#why-not-autonomous-agents)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Why AgentLoom?

Existing frameworks (LangGraph, CrewAI, AutoGen) treat observability and resilience as afterthoughts. AgentLoom is built from the ground up for production: circuit breakers, rate limiting, cost tracking, and OpenTelemetry traces are part of the core design — not plugins.

| Feature | LangGraph | CrewAI | AutoGen | AgentLoom |
|---|---|---|---|---|
| Workflow definition | Python API | Decorators | Agent chat | **YAML + Python DSL** |
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
|                   CLI / Python API                   |
+-----------------------------------------------------+
|                   Workflow Engine                    |
|  +-----------+  +-----------+  +---------------+    |
|  |DAG Parser |  | Scheduler |  | State Manager |    |
|  |& Validator|  |  (anyio)  |  |  (Pydantic)   |    |
|  +-----------+  +-----------+  +---------------+    |
+-----------------------------------------------------+
|                   Step Executors                     |
|  +--------+ +---------+ +------+ +------------+    |
|  |LLM Call| |Tool Exec| |Router| | Subworkflow|    |
|  +--------+ +---------+ +------+ +------------+    |
+-----------------------------------------------------+
|                  Provider Gateway                   |
|  +-----------------------------------------------+ |
|  | OpenAI | Anthropic | Google | Ollama           | |
|  | + Fallback | Circuit Breaker | Rate Limiter    | |
|  +-----------------------------------------------+ |
+-----------------------------------------------------+
|              Observability (optional)               |
|  +------------+  +----------+  +----------+        |
|  | OTel Traces|  |Prometheus|  | JSON Logs|        |
|  +------------+  +----------+  +----------+        |
+-----------------------------------------------------+
```

## Workflow Definition (YAML)

```yaml
name: classify-and-respond
config:
  provider: openai
  model: gpt-4.1-nano
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

## Observability Stack

```bash
# Start Prometheus + Grafana + Jaeger
cd deploy && docker compose up -d

# Access:
#   Grafana:    http://localhost:3000
#   Prometheus: http://localhost:9090
#   Jaeger:     http://localhost:16686
```

See [Dashboard Documentation](deploy/DASHBOARD.md) for panel descriptions, metrics reference, and troubleshooting.

## Why not autonomous agents?

The AI agent ecosystem is at peak hype. New frameworks ship weekly, each promising autonomous agents that reason, plan, and collaborate. Most of them optimize for demos: flashy multi-agent conversations, auto-generated chains of thought, agent-to-agent delegation — impressive in a notebook, fragile in production.

AgentLoom takes the opposite stance. It is **not** an autonomous agent framework. There are no self-directed agents, no unbounded loops, no emergent multi-agent negotiation. Instead, it is a **deterministic workflow orchestrator** that happens to use LLMs as execution steps.

The difference matters:

- **You define the DAG, not the LLM.** Steps, dependencies, and routing logic are declared upfront in YAML. The model generates text within a step — it does not decide what runs next. Routers use explicit boolean conditions, not LLM judgement.
- **Observability is not optional.** Every step emits OpenTelemetry traces and Prometheus metrics. You can see exactly what ran, how long it took, and how much it cost. Autonomous agents are notoriously hard to debug; a static DAG with full tracing is not.
- **Cost is bounded.** Budget limits, circuit breakers, and rate limiters are first-class. A runaway autonomous agent can burn through an API budget in minutes. A workflow with `budget_usd: 0.50` cannot.
- **Fallback is structural.** If OpenAI is down, the gateway falls back to Anthropic or Ollama automatically. This is a routing decision at the infrastructure level, not an agent "choosing" a provider.

Autonomous agent frameworks solve a real problem — open-ended tasks where the execution path cannot be known in advance. But most LLM workloads in production are not open-ended. They are pipelines: classify, enrich, route, generate, validate. For those, you want predictability and control, not autonomy. That is what AgentLoom is for.

## Development

```bash
# Install dev dependencies
uv sync --group dev --all-extras

# Run tests
uv run pytest

# Lint + type check
uv run ruff check src/
uv run mypy src/

# Format
uv run ruff format src/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT
