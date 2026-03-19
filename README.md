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
