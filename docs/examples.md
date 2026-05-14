# Examples

28 example workflows covering basic patterns to production-grade pipelines. All examples run with Ollama (free, local) or any cloud provider.

```bash
# Validate any example
agentloom validate examples/01_simple_qa.yaml

# Run with Ollama
agentloom run examples/01_simple_qa.yaml --provider ollama --model phi4

# Run with OpenAI
export OPENAI_API_KEY=sk-...
agentloom run examples/02_chain_of_thought.yaml

# Visualize the DAG
agentloom visualize examples/03_router_workflow.yaml
agentloom visualize examples/03_router_workflow.yaml --format mermaid
```

---

## Basic

### 01 — Simple QA

Single LLM call. Sends a question, gets an answer.

**Demonstrates:** basic workflow structure, state, output mapping.

```yaml
name: simple-qa
config:
  provider: ollama
  model: phi4
state:
  question: "What is Python in one sentence?"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer this question concisely: {state.question}"
    output: answer
```

### 02 — Chain of Thought

Three sequential LLM calls: break down topic, research subtopics, synthesize summary.

**Demonstrates:** sequential dependencies, state passing between steps.

### 03 — Customer Support Router

Classifies user intent, routes to specialized handler (billing/technical/general).

**Demonstrates:** `router` step, conditional branching — only one branch executes.

### 04 — Tool Augmented

Fetches data from a URL using the `http_request` tool, then analyzes it with an LLM.

**Demonstrates:** tool integration, mixing `tool` and `llm_call` steps.

---

## Intermediate

### 05 — Content Moderation Pipeline

Parallel content moderation for UGC platforms. Runs toxicity, PII, and policy checks simultaneously, aggregates results, and routes to approve/review/reject.

**Demonstrates:** parallel execution (3 checks at once), aggregation, router with 3 branches.

### 06 — Lead Qualification

B2B lead qualification pipeline. Enriches company data via API, analyzes buying intent, scores the lead, and routes to personalized outreach, nurture, or archive.

**Demonstrates:** parallel tool+LLM execution, multi-step scoring, conditional routing.

### 07 — Incident Triage

Automated incident triage for SRE teams. Fetches deployment context, correlates with alert data, classifies severity (P1/P2/P3), and generates response actions.

**Demonstrates:** 5-layer deep pipeline, tool integration, severity-based routing.

### 08 — Contract Risk Analysis

Legal contract risk analysis. Extracts clauses, evaluates risk, compares against industry standards, and generates an executive summary with sign/negotiate/walk-away recommendation.

**Demonstrates:** deep 4-step sequential chain, state accumulation, structured analysis.

---

## Advanced

### 09 — Fraud Detection Pipeline

E-commerce order fraud detection. Fetches order + customer data in parallel, runs 4 concurrent fraud signal checks (velocity, amount, address, device), aggregates risk, and routes high-risk orders to a deep investigation subworkflow.

**Demonstrates:** 2 parallel tools, 4 parallel LLM checks, subworkflow, 11 steps / 5 layers.

### 10 — Multi-Market Content Localization

SaaS content localization across 3 markets. Fetches source content, runs 3 parallel localization subworkflows (ES/DE/JA) — each with 4 steps: translate, culturally adapt, legal compliance, and format — then cross-market review and deployment manifest.

**Demonstrates:** 3 parallel subworkflows (12 nested steps), 18 total steps across 4 layers.

### 11 — Insurance Claims Processing

End-to-end claims adjudication. Extracts structured data, runs 3 parallel validation tracks via subworkflows (coverage, fraud, medical necessity), aggregates results, routes to auto-approve, adjuster assignment, or denial.

**Demonstrates:** 4 subworkflows, 6 layers, 21 total steps. The most complex example.

---

## Built-in Tools

### 12 — Log Analysis & Alerting

Server log analysis with `shell_command` tool. Collects error logs and system metrics in parallel, correlates them with an LLM, classifies severity, and routes to PagerDuty alert, ops ticket, or silent log.

**Demonstrates:** `shell_command` tool, parallel data collection, severity routing.

### 13 — Report Generator

Data pipeline health report using `file_read` and `file_write` tools. Reads pipeline results and SLA config, analyzes compliance, generates executive summary, and writes the final report to disk.

**Demonstrates:** `file_write` + `file_read` tools, file-based data pipeline.

### 14 — Custom Tools: @tool Decorator

Sentiment monitoring pipeline with custom tools defined using the `@tool` decorator: `query_database`, `send_slack_message`, `create_ticket`.

**Demonstrates:** `@tool` decorator, custom tool registration.

```bash
uv run python examples/14_custom_tools_decorator.py
```

!!! info "Two files"
    This example uses a paired YAML workflow + Python runner:
    `14_custom_tools_decorator.yaml` + `14_custom_tools_decorator.py`

### 15 — Custom Tools: BaseTool Subclass

Customer data enrichment with custom tools defined as `BaseTool` subclasses: `GeocodingTool`, `CRMLookupTool`, `RiskScoreTool`.

**Demonstrates:** `BaseTool` subclass pattern, churn risk scoring.

```bash
uv run python examples/15_custom_tools_subclass.py
```

---

## Resilience & Sandbox

### 16 — Circuit Breaker Demo

Intentionally trips the circuit breaker with a non-existent model, then shows automatic fallback.

**Demonstrates:** circuit breaker state transitions (CLOSED -> OPEN), provider fallback.

### 17 — Sandbox: Allowed Operations

Runs commands and file I/O within sandbox limits. `echo` is allowed, files stay inside `/tmp/agentloom`, network is enabled.

**Demonstrates:** `config.sandbox`, command allowlist, path restriction.

### 18 — Sandbox: Blocked Operations

Attempts operations that violate sandbox policy: disallowed command (`curl`), path outside allowed directory, network disabled, and pipe injection (`echo | cat`).

**Demonstrates:** sandbox enforcement, command injection prevention, path and network blocking.

---

## Multi-modal

### 19 — URL Image

Fetches an image from a public URL and describes it. The engine downloads the image locally and sends base64 to the provider (`fetch: local`).

**Demonstrates:** `attachments`, URL fetch, multi-step vision pipeline.

### 20 — Base64 Inline

Analyzes an image embedded directly in workflow state as base64. No network access needed.

**Demonstrates:** inline base64 attachment, offline-capable vision.

### 21 — URL Passthrough

Sends the image URL directly to the provider API (`fetch: provider`). Only works with OpenAI and Anthropic.

**Demonstrates:** `fetch: provider` mode, provider-side image fetching.

### 22 — Sandboxed URL

Image analysis with sandbox restrictions. Only domains in `allowed_domains` are permitted.

**Demonstrates:** sandbox `allowed_domains` for attachments.

### 23 — PDF Document

Extracts key points from a PDF and generates an executive summary. Requires Anthropic or Google.

**Demonstrates:** `type: pdf` attachment, document analysis.

### 24 — Audio Transcription

Transcribes an audio clip and analyzes for topic, sentiment, and action items. Requires OpenAI or Google.

**Demonstrates:** `type: audio` attachment, transcription + analysis pipeline.

---

## Streaming

### 25 — Streaming QA

Streams LLM output token-by-token in real-time.

**Demonstrates:** `stream` config, `--stream` CLI flag, time-to-first-token tracking.

```bash
agentloom run examples/25_streaming_qa.yaml --stream
```

### 26 — Streaming + Multi-modal

Combines streaming with image input. The image is fetched locally, and the LLM description is streamed back in real-time.

**Demonstrates:** streaming + attachments composability.

---

## State Features

### 27 — Array Index

Array index support in state paths (`state.items[0]`, `items[0].name`, `results[-1]`).

**Demonstrates:** bracket-based array indexing in state and templates.

---

## Checkpointing

### 28 — Checkpoint & Resume

Two-step workflow with checkpointing enabled. Persists execution state so failed
or interrupted runs can be resumed without re-executing completed steps.

**Demonstrates:** `--checkpoint` flag, `agentloom runs`, `agentloom resume`.

```bash
# Run with checkpointing
agentloom run examples/28_checkpoint_resume.yaml --checkpoint --lite

# List checkpointed runs
agentloom runs

# Resume a failed or interrupted run
agentloom resume <run_id> --lite
```

## Testing & Replay

### 31 — Record and Replay

Two-step workflow used to capture real LLM responses and replay them offline.
Run once with `--record` against a real provider, then replay any number of
times without network or API keys.

**Demonstrates:** `--record`, `--mock-responses`, `agentloom replay`.

```bash
# Capture real Anthropic responses (per-call flush; partial recordings survive crashes)
agentloom run examples/31_record_and_replay.yaml --record recordings/byzantine.json

# Replay offline — pick either form
agentloom replay examples/31_record_and_replay.yaml --recording recordings/byzantine.json
agentloom run examples/31_record_and_replay.yaml --mock-responses recordings/byzantine.json
```

### 32 — YAML-configured MockProvider

Same workflow as 31, but with `provider: mock` and `responses_file` declared in
YAML. Runs via plain `agentloom run` with no CLI flags — useful for committed
fixtures and CI.

**Demonstrates:** `provider: mock`, `responses_file`, `latency_model: replay`.

```bash
# Depends on the recording captured from example 31
agentloom run examples/32_yaml_mock.yaml --lite
```

## Tool calling

### 35 — Native tool/function calling

ReAct-style agent: the model decides to invoke `http_request` against `httpbin.org/get`, receives the JSON, and emits a final natural-language answer. Sandbox is on with `allowed_domains: ["httpbin.org"]`, so the model-dispatched call goes through the same security policy as static `tool` steps (#105).

**Demonstrates:** `tools` declaration on `llm_call`, `tool_choice: auto`, `max_tool_iterations`, model-driven dispatch via `ToolRegistry`, sandboxed tool execution, replay support for tool-iteration loops.

```bash
# Mock-replay against the committed recording (no API calls)
agentloom run examples/35_tool_calling.yaml --lite

# Real call: pass --provider + --model to drive a live model
agentloom run examples/35_tool_calling.yaml --provider openai --model gpt-4o-mini
```
