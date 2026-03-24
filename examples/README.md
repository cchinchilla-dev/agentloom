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
