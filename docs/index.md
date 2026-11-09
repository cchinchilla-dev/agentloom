---
title: "Getting started"
description: "Install AgentLoom, write a first YAML workflow and run it against OpenAI, Anthropic, Google, a local Ollama model or no provider at all."
order: 1
section: "Start"
---

# AgentLoom

Deterministic LLM workflow orchestration with native observability, resilience, and cost control.

<!-- cards -->

- **DAG-based workflows** — Define workflows as directed acyclic graphs in YAML or Python. Steps, dependencies and routing are declared upfront: the LLM generates text, not control flow.
- **Native observability** — OpenTelemetry traces and Prometheus metrics for every step, and a Grafana dashboard to read them. No external SaaS required.
- **Built-in resilience** — Retries with backoff, timeouts, and a circuit breaker and rate limiter per provider. Checkpoints let a failed or paused run resume where it left off.
- **Cost control** — Tokens and cost tracked for every step, and a budget per workflow: once `budget_usd` is spent, no further step starts.

<!-- /cards -->

## Requirements

- **Python 3.11 or newer.** AgentLoom supports 3.11 to 3.14.
- **A model to call.** An API key for OpenAI, Anthropic or Google, or a local [Ollama](https://ollama.com). No key at all works too: the `mock` provider answers offline.
- **No provider SDKs.** AgentLoom calls each provider's HTTP API itself, so `pip` installs nothing from OpenAI, Anthropic or Google.

## Installation

<!-- tabs -->

**Core**

```bash
pip install agentloom
```

**With observability**

```bash
pip install "agentloom[observability]"
```

OpenTelemetry and the Prometheus client, for traces and metrics.

**With graph analysis**

```bash
pip install "agentloom[graph]"
```

NetworkX, for the [Graph API](graph-api.md).

**Everything**

```bash
pip install "agentloom[all]"
```

<!-- /tabs -->

With extras, keep the quotes: zsh reads unquoted brackets as a file pattern. To check the install, `agentloom info` prints the version, the Python it runs on, which optional extras are installed and which provider keys are set (never their values).

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

One step calls the model with a prompt built from the state, and writes the reply to the `answer` key.

**2. Run it** with the provider you have:

<!-- tabs -->

**OpenAI**

```bash
export OPENAI_API_KEY=sk-...
agentloom run my_workflow.yaml
```

**Anthropic**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agentloom run my_workflow.yaml --provider anthropic --model claude-haiku-4-5-20251001
```

**Google**

```bash
export GOOGLE_API_KEY=...
agentloom run my_workflow.yaml --provider google --model gemini-2.5-flash
```

**Ollama**

```bash
ollama pull phi4
agentloom run my_workflow.yaml --provider ollama --model phi4
```

Free and local. Ollama must be running (`ollama serve`) at `http://localhost:11434`.

**No key**

```bash
agentloom run my_workflow.yaml --provider mock
```

Every call answers with the placeholder text `Mock response`: enough to try the CLI and check a workflow's wiring. [Testing & Replay](testing-and-replay.md) shows how to replay real responses instead.

<!-- /tabs -->

**3. Read the result.** The run prints a summary:

```text
Running workflow: simple-qa

============================================================
Workflow: simple-qa
Status:   [OK] success
Duration: 812.4ms
Tokens:   41
Cost:     $0.0000

Steps:
  [OK] answer (809ms)

Final State Keys: ['question', 'answer', 'steps']
============================================================
```

The summary lists what each step did, not what the model said. To see the answer, ask for the whole result as JSON, or stream the text as it arrives:

```bash
agentloom run my_workflow.yaml --json | jq -r .final_state.answer
agentloom run my_workflow.yaml --stream
```

**4. Validate and visualize:**

```bash
agentloom validate my_workflow.yaml    # check the file without running it
agentloom visualize my_workflow.yaml   # draw the DAG as text; --format mermaid for a diagram
```

## Keys and providers

- **Keys come from the environment**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`. AgentLoom reads no `.env` file, so load one with your shell or a tool like direnv. Each provider whose key is set is available to every step.
- **The model name picks the provider.** `gpt-*`, `o1`, `o3` and `o4` models go to OpenAI, `claude*` to Anthropic, `gemini*` to Google. `--provider` only says which provider is tried first. To switch provider, pass `--model` too: `--provider anthropic` alone still asks for `gpt-4o-mini`, which only OpenAI serves.
- **Ollama takes no key.** It is used when a workflow says `provider: ollama` (or `--provider ollama`), and accepts any model name.
- **Endpoints can be changed** with `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL` and `OLLAMA_BASE_URL`, for a proxy or a self-hosted gateway. Any other OpenAI-compatible server works through the [`generic_openai` provider](providers.md).

[Providers](providers.md) covers every provider, model and capability.

## What a run leaves behind

| Path | Written | What it is |
|------|---------|------------|
| `agentloom_runs/<run_id>.json` | Every run | A record of the run: status, steps, tokens and cost. `AGENTLOOM_RUNS_DIR` moves it; `agentloom history` reads it. See [run history](observability.md#run-history). |
| `.agentloom/checkpoints/<run_id>.json` | With `--checkpoint` | The state a run stopped with, which [`agentloom resume`](reliability.md#checkpoints-and-resume) continues from. |
| Traces and metrics | With the observability extra installed | Sent over OTLP to `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4317`). `--lite` turns them off. See [Observability](observability.md#quick-start). |

Both directories sit in the current directory: add `agentloom_runs/` and `.agentloom/` to your `.gitignore`.

## Exit codes

| Code | When |
|------|------|
| `0` | The workflow succeeded, or paused at an approval gate. |
| `1` | The workflow failed, timed out or ran out of budget, or the file did not load. |
| `2` | The command was misused: a missing file or options that cannot go together. |

Scripts and CI jobs can rely on them; `--json` prints the full result either way.

## Common first-run errors

| You see | Why | Fix |
|---------|-----|-----|
| `No provider registered for model 'gpt-4o-mini'` | No key is set for the provider that serves the model. | Export the key; `agentloom info` shows which keys are set. |
| `All providers failed for model …` with a 401 or 403 | The key is wrong or revoked. | Check the key in the provider's console. |
| `All providers failed for model …` with a 404 | The provider doesn't know the model, often a retired alias. | Use one of the [recommended model IDs](providers.md#recommended-model-ids). |
| A connection error with `--provider ollama` | Ollama isn't running, or the model isn't pulled. | `ollama serve`, then `ollama pull phi4`. |
| `Error loading workflow` with `Extra inputs are not permitted` | A key in the YAML is misspelled. | `agentloom validate` names the field. |
| A prompt shows an empty value, or a literal `{name}` | The template names a state key that doesn't exist: `{state.name}` renders empty, `{name}` stays as written. | Check the key against `state:` and the steps' `output:`. See [Missing keys](state-and-templates.md#missing-keys). |
| `zsh: no matches found: agentloom[all]` | The extras aren't quoted. | `pip install "agentloom[all]"`. |
| Export errors in the log after a run | The observability extra is installed, but no collector listens at the OTLP endpoint. | Pass `--lite`, or start the [observability stack](observability.md#quick-start). |

## What's next

<!-- cards -->

- **[Concepts](concepts.md)** — Workflows, steps, state, providers and runs: the ideas every other page builds on.
- **[Workflow files](workflow-files.md)** — Every key of a workflow file, every config option and what validate checks.
- **[Steps](steps.md)** — The seven step types, with routers, subworkflows, approval gates and embeddings in full.
- **[State & templates](state-and-templates.md)** — How steps write to the state, and the template syntax that reads it.
- **[Providers](providers.md)** — Providers, models, attachments, reasoning and pricing.
- **[CLI reference](cli.md)** — Every `agentloom` command and option.
- **[Examples](examples.md)** — 39 example workflows, from one LLM call to production-grade pipelines.
- **[Python DSL](python-dsl.md)** — Build and run workflows from Python.
- **[Testing & Replay](testing-and-replay.md)** — Record real runs and replay them offline, deterministically.
- **[Observability](observability.md)** — Traces, metrics, dashboards and run records.
- **[Deployment](deployment.md)** — Docker, Kubernetes, Helm, Terraform and ArgoCD.
- **[Architecture](architecture.md)** — How the engine schedules, runs and records a workflow.
- **[Graph API](graph-api.md)** — Analyze, visualize and export workflow DAGs.
- **[Changelog](../CHANGELOG.md)** — Every release and what changed in it.

<!-- /cards -->
