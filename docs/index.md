---
hide:
  - navigation
---

# AgentLoom

**Deterministic LLM workflow orchestration with native observability, resilience, and cost control.**

---

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **DAG-based workflows**

    ---

    Define workflows as directed acyclic graphs in YAML or Python. Steps, dependencies, and routing are declared upfront — the LLM generates text, not control flow.

-   :material-eye-outline:{ .lg .middle } **Native observability**

    ---

    OpenTelemetry traces and Prometheus metrics on every step. Grafana dashboards included. No external SaaS required.

-   :material-shield-check-outline:{ .lg .middle } **Built-in resilience**

    ---

    Circuit breakers, rate limiters, and automatic multi-provider fallback. If OpenAI is down, the gateway falls back to Anthropic or Ollama.

-   :material-currency-usd:{ .lg .middle } **Cost control**

    ---

    Per-workflow budget limits, token tracking, and cost estimation across all providers. A workflow with `budget_usd: 0.50` cannot overspend.

</div>

---

## Installation

=== "Core"

    ```bash
    pip install agentloom
    ```

=== "With observability"

    ```bash
    pip install agentloom[all]
    ```

=== "With graph analysis"

    ```bash
    pip install agentloom[graph]
    ```

## Quick start

**1. Create a workflow** — `my_workflow.yaml`:

```yaml
name: simple-qa
config:
  provider: openai
  model: gpt-4o-mini

state:
  question: "What is Python in one sentence?"

steps:
  - id: answer
    type: llm_call
    prompt: "Answer this question concisely: {state.question}"
    output: answer
```

**2. Run it:**

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=sk-...
    agentloom run my_workflow.yaml
    ```

=== "Ollama (free, local)"

    ```bash
    agentloom run my_workflow.yaml --provider ollama --model phi4
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...
    agentloom run my_workflow.yaml --provider anthropic --model claude-sonnet-4-20250514
    ```

=== "Google"

    ```bash
    export GOOGLE_API_KEY=...
    agentloom run my_workflow.yaml --provider google --model gemini-2.5-flash
    ```

**3. Validate and visualize:**

```bash
agentloom validate my_workflow.yaml    # check for errors
agentloom visualize my_workflow.yaml   # render the DAG
```

## What's next

| Section | Description |
|---------|-------------|
| [Architecture](architecture.md) | Execution engine, DAG scheduler, state management |
| [Providers](providers.md) | Supported providers, models, and multi-modal capabilities |
| [Workflow YAML](workflow-yaml.md) | Full reference for step types, config, routing, and attachments |
| [Python DSL](python-dsl.md) | Build workflows programmatically |
| [Graph API](graph-api.md) | Analyze, visualize, and export workflow DAGs |
| [Observability](observability.md) | Traces, metrics, and Grafana dashboards |
| [Examples](examples.md) | 27 example workflows from basic to production-grade |
| [Deployment](deployment.md) | Docker, Kubernetes, Helm, Terraform, and ArgoCD |
| [Changelog](changelog.md) | Version history and release notes |
