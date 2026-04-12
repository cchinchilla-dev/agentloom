# AgentLoom Examples

## Quick Start

```bash
# Validate a workflow
agentloom validate examples/01_simple_qa.yaml

# Run with Ollama (local, free)
agentloom run examples/01_simple_qa.yaml --provider ollama --model phi4

# Run with OpenAI
export OPENAI_API_KEY=sk-...
agentloom run examples/02_chain_of_thought.yaml

# Visualize the DAG
agentloom visualize examples/03_router_workflow.yaml
agentloom visualize examples/03_router_workflow.yaml --format mermaid
```

## Examples

### 01 — Simple QA
Single LLM call. Sends a question, gets an answer.
Demonstrates: basic workflow structure, state, output mapping.

### 02 — Chain of Thought
Three sequential LLM calls: break down topic, research subtopics, synthesize summary.
Demonstrates: sequential dependencies, state passing between steps.

### 03 — Customer Support Router
Classifies user intent, routes to specialized handler (billing/technical/general).
Demonstrates: router step, conditional branching, only one branch executes.

### 04 — Tool Augmented
Fetches data from a URL using the http_request tool, then analyzes it with an LLM.
Demonstrates: tool integration, mixing tool and LLM steps.

---

### 05 — Content Moderation Pipeline
Parallel content moderation for UGC platforms. Runs toxicity, PII, and policy
checks simultaneously, aggregates results, and routes to approve/review/reject.
Demonstrates: parallel execution (3 checks at once), aggregation, router with 3 branches.

### 06 — Lead Qualification
B2B lead qualification pipeline. Enriches company data via API, analyzes buying
intent, scores the lead, and routes to personalized outreach, nurture, or archive.
Demonstrates: parallel tool+LLM execution, multi-step scoring, conditional routing.

### 07 — Incident Triage
Automated incident triage for SRE teams. Fetches deployment context, correlates
with alert data, classifies severity (P1/P2/P3), and generates response actions.
Demonstrates: 5-layer deep pipeline, tool integration, severity-based routing.

### 08 — Contract Risk Analysis
Legal contract risk analysis pipeline. Extracts clauses, evaluates risk from the
client's perspective, compares against industry standards, and generates an executive
summary with sign/negotiate/walk-away recommendation.
Demonstrates: deep 4-step sequential chain, state accumulation, structured analysis.

---

### 09 — Fraud Detection Pipeline
E-commerce order fraud detection. Fetches order + customer data in parallel, runs 4
concurrent fraud signal checks (velocity, amount, address, device), aggregates risk,
and routes high-risk orders to a deep investigation subworkflow that classifies the
fraud type and recommends specific actions.
Demonstrates: 2 parallel tools, 4 parallel LLM checks, subworkflow, 11 steps / 5 layers.

### 10 — Multi-Market Content Localization
SaaS content localization across 3 markets. Fetches source content, runs 3 parallel
localization subworkflows (ES/DE/JA) — each with 4 steps: translate, culturally adapt,
legal compliance check, and format — then cross-market quality review and deployment manifest.
Demonstrates: 3 parallel subworkflows (12 nested steps), 18 total steps across 4 layers.

### 11 — Insurance Claims Processing
End-to-end claims adjudication. Extracts structured data, runs 3 parallel validation
tracks via subworkflows (coverage verification with payout calculation, fraud detection
with pattern analysis, medical necessity review), aggregates results, routes to
auto-approve (with payment authorization subworkflow), adjuster assignment, or denial.
Demonstrates: 4 subworkflows, 6 layers, 21 total steps. The most complex example.

---

### 12 — Log Analysis & Alerting
Server log analysis with `shell_command` tool. Collects error logs and system metrics
in parallel via shell commands, correlates them with an LLM, classifies severity, and
routes to PagerDuty alert, ops ticket, or silent log.
Demonstrates: `shell_command` tool, parallel data collection, severity routing.

### 13 — Report Generator
Data pipeline health report using `file_read` and `file_write` tools. Reads pipeline
results and SLA config from JSON files, analyzes compliance, generates executive
summary, and writes the final report to disk.
Demonstrates: `file_write` + `file_read` tools, file-based data pipeline, structured analysis.

### 14 — Custom Tools: @tool Decorator (`*.yaml` + `*.py`)
Sentiment monitoring pipeline with custom tools defined using the `@tool` decorator:
`query_database`, `send_slack_message`, `create_ticket`. Queries customer feedback,
analyzes sentiment, routes alerts to Slack or ticketing.
Demonstrates: `@tool` decorator, custom tool registration, separate YAML workflow + Python runner.

- `14_custom_tools_decorator.yaml` — workflow definition
- `14_custom_tools_decorator.py` — custom tools + runner

```bash
uv run python examples/14_custom_tools_decorator.py
uv run python examples/14_custom_tools_decorator.py --provider openai --model gpt-4o-mini
```

### 15 — Custom Tools: BaseTool Subclass (`*.yaml` + `*.py`)
Customer data enrichment pipeline with custom tools defined as `BaseTool` subclasses:
`GeocodingTool`, `CRMLookupTool`, `RiskScoreTool`. Enriches customer profiles with
geodata and CRM history, computes churn risk, and generates retention strategies.
Demonstrates: `BaseTool` subclass, separate YAML workflow + Python runner.

- `15_custom_tools_subclass.yaml` — workflow definition
- `15_custom_tools_subclass.py` — custom tool classes + runner

```bash
uv run python examples/15_custom_tools_subclass.py
uv run python examples/15_custom_tools_subclass.py --provider openai --model gpt-4o-mini
```

---

### 16 — Circuit Breaker Demo
Demonstrates the resilience layer. Intentionally sends requests to a non-existent
model to trip the circuit breaker, then shows automatic fallback to a working provider.
Demonstrates: circuit breaker state transitions (CLOSED → OPEN), provider fallback, resilience metrics.

### 17 — Sandbox: Allowed Operations
Runs commands and file I/O within sandbox limits. `echo` is allowed, files stay
inside `/tmp/agentloom`, network is enabled. All steps complete successfully.
Demonstrates: `config.sandbox`, command allowlist, path restriction, sandboxed workflow.

### 18 — Sandbox: Blocked Operations
Attempts operations that violate the sandbox policy: disallowed command (`curl`),
path outside allowed directory (`/etc/hostname`), network disabled, and pipe injection
(`echo | cat`). Only the first step (`echo`) succeeds — the rest are blocked with
`SandboxViolationError`.
Demonstrates: sandbox enforcement, command injection prevention, path and network blocking.

---

### 19 — Multi-modal: URL Image
Fetches an image from a public URL and asks the LLM to describe it. The pod
downloads the image locally and sends it to the provider as base64 (`fetch: local`,
the default). A second step summarizes the description.
Demonstrates: `attachments`, URL fetch with local download, multi-step vision pipeline.

### 20 — Multi-modal: Base64 Inline
Analyzes an image embedded directly in the workflow state as a base64 string. No
network access is needed — the image data is self-contained in the YAML.
Demonstrates: inline base64 attachment, offline-capable vision, explicit `media_type`.

### 21 — Multi-modal: URL Passthrough
Sends the image URL directly to the LLM provider API (`fetch: provider`). The
provider fetches the image itself. Only works with providers that support URL-based
vision input (OpenAI, Anthropic). Google and Ollama will reject this with a clear error.
Demonstrates: `fetch: provider` mode, provider-side image fetching.

### 22 — Multi-modal: Sandboxed URL
Image analysis with sandbox restrictions on URL fetching. Only domains listed in
`config.sandbox.allowed_domains` are permitted. Requests to other domains are
blocked with a `PermissionError` before any network call is made.
Demonstrates: sandbox `allowed_domains` for attachments, security controls for vision input.

### 23 — Multi-modal: PDF Document
Extracts key points from a PDF document and generates an executive summary.
Requires a provider that supports PDF attachments (Anthropic or Google).
OpenAI and Ollama will reject PDF attachments with a clear error.
Demonstrates: `type: pdf` attachment, document analysis pipeline.

### 24 — Multi-modal: Audio Transcription
Transcribes an audio clip and analyzes the transcript for topic, sentiment,
and action items. Requires a provider that supports audio (OpenAI or Google).
OpenAI only accepts WAV and MP3 formats. Anthropic and Ollama will reject audio.
Demonstrates: `type: audio` attachment, transcription + analysis pipeline.

---

### 25 — Streaming: Simple QA
Streams LLM output token-by-token in real-time. Uses `config.stream: true`
at the workflow level. Run with `--stream` to see tokens appear live.
Demonstrates: `stream` config, `--stream` CLI flag, time-to-first-token tracking.

```bash
agentloom run examples/25_streaming_qa.yaml --stream
```

### 26 — Streaming + Multi-modal
Combines streaming with image input. The image is fetched locally, and the
LLM description is streamed back in real-time.
Demonstrates: streaming + attachments composability, real-time vision output.

---

### 27 — Array Index Access
Accesses array elements in state using bracket notation (`{state.items[0]}`).
Demonstrates: array index state paths, bracket notation in templates.

### 28 — Checkpoint & Resume
Two-step workflow with checkpointing enabled. Run with `--checkpoint` to
persist execution state. If the workflow fails mid-execution, resume from the
last checkpoint instead of re-running completed steps.
Demonstrates: `--checkpoint` flag, `agentloom runs`, `agentloom resume`.

```bash
# Run with checkpointing
agentloom run examples/28_checkpoint_resume.yaml --checkpoint --lite

# List checkpointed runs
agentloom runs

# Resume a failed or interrupted run
agentloom resume <run_id> --lite
```
