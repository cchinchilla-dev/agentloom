---
title: "Examples"
description: "39 example workflows in the AgentLoom repository, from one LLM call to production pipelines, and how to run each one, offline or with a key."
order: 3
section: "Start"
---

# Examples

Thirty-nine workflows in the repository's [`examples/`](../examples/) folder, from a single LLM call to pipelines with parallel checks, routing, subworkflows and human approval. Seven of them run offline, against recordings committed next to them.

## Running the examples

Clone the repository, install it, and run the examples from its root: recordings, agent definitions and data files are found from there.

```bash
git clone https://github.com/cchinchilla-dev/agentloom.git
cd agentloom
pip install -e .    # the package that matches these examples
```

<!-- tabs -->

**Offline**

```bash
agentloom run examples/41_agent.yaml --lite
```

Examples 35 to 41 name the `mock` provider and replay a recording from [`recordings/`](../recordings/): no key, no network, the same answers every time. `--lite` skips the observability exporter.

**With a key**

```bash
export OPENAI_API_KEY=sk-...
agentloom run examples/02_chain_of_thought.yaml
```

Most examples name OpenAI's `gpt-4o-mini` in their `config`; the list below says which need another provider or files of their own.

**Another provider**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agentloom run examples/02_chain_of_thought.yaml --provider anthropic --model claude-haiku-4-5-20251001
```

Pass the model too: the model name picks the provider.

**Ollama**

```bash
ollama pull phi4
agentloom run examples/02_chain_of_thought.yaml --provider ollama --model phi4
```

Free and local; small models follow long prompts less closely than the ones the examples were written for.

<!-- /tabs -->

Before a run, `agentloom validate examples/<file>` checks a file and `agentloom visualize examples/<file>` draws its DAG.

## Basics

One idea each: the shape of a workflow, a chain, a branch, a tool.

<!-- cards -->

- **[01 · Simple QA](../examples/01_simple_qa.yaml)** — One LLM call that answers a question from the state: config, state and an output, and nothing else. Runs on Ollama with `phi4`.
- **[02 · Chain of thought](../examples/02_chain_of_thought.yaml)** — Three calls in sequence: break a question down, research each part, synthesize an answer. Each step reads what the previous one wrote. OpenAI.
- **[03 · Support router](../examples/03_router_workflow.yaml)** — Classifies a customer's message and routes it to a billing, technical or general handler; only the chosen branch runs. OpenAI.
- **[04 · Tool augmented](../examples/04_tool_augmented.yaml)** — Fetches JSON with the built-in `http_request` tool, then has a model analyze it. OpenAI.

<!-- /cards -->

## Pipelines

Realistic workflows: parallel checks that join, a router at the end, subworkflows for repeated tracks. They use placeholder APIs, so the HTTP steps show the shape rather than real data.

<!-- cards -->

- **[05 · Content moderation](../examples/05_content_moderation.yaml)** — Toxicity, PII and policy checks run in parallel, then a router approves, escalates to a person or rejects. Eight steps in four layers, at most three at once. OpenAI.
- **[06 · Lead qualification](../examples/06_lead_qualification.yaml)** — Enriches a lead through an API, scores its buying intent and routes it to outreach, nurture or archive. OpenAI.
- **[07 · Incident triage](../examples/07_incident_triage.yaml)** — Correlates a deployment with an alert, classifies the severity as P1, P2 or P3 and drafts a page, a ticket or a backlog item. OpenAI.
- **[08 · Contract analysis](../examples/08_contract_analysis.yaml)** — A four-step chain over a SaaS agreement: extract the clauses, assess the risk, compare with industry standards, summarize. OpenAI.
- **[09 · Fraud detection](../examples/09_fraud_detection.yaml)** — Two parallel fetches, four parallel fraud checks and a risk score; high-risk orders go to a nested investigation subworkflow. Eleven steps in five layers. OpenAI.
- **[10 · Content localization](../examples/10_content_localization.yaml)** — Three localization subworkflows, Spanish, German and Japanese, run in parallel, four steps each, then a review and a deployment manifest. Eighteen steps in all. OpenAI.
- **[11 · Insurance claims](../examples/11_insurance_claims.yaml)** — Coverage and fraud checks run as subworkflows beside a medical review; the result routes to automatic approval, with its own payment subworkflow, an adjuster or a denial. Eighteen steps in six layers. OpenAI.

<!-- /cards -->

## Tools

The built-in tools, and tools of your own.

<!-- cards -->

- **[12 · Log analysis](../examples/12_log_analysis.yaml)** — Two `shell_command` steps gather error logs and system metrics in parallel (canned `echo` output, so it runs anywhere), then a model correlates them and a router acts on the severity. OpenAI.
- **[13 · Report generator](../examples/13_report_generator.yaml)** — Writes its own input files with `file_write`, reads them back with `file_read`, analyzes SLA compliance and writes a report, all under `/tmp/agentloom-demo/`. OpenAI.
- **[14 · Custom tools with a decorator](../examples/14_custom_tools_decorator.yaml)** — Tools defined as decorated Python functions, with their schema generated from the signature. Run the Python file next to it, `python examples/14_custom_tools_decorator.py`, which registers the tools and loads the YAML. Ollama with `phi4`.
- **[15 · Custom tools as classes](../examples/15_custom_tools_subclass.yaml)** — The same idea with tools that subclass `BaseTool`; run `python examples/15_custom_tools_subclass.py`. Ollama with `phi4`.

<!-- /cards -->

## Resilience and sandbox

How failures surface, and what the sandbox lets a tool do.

<!-- cards -->

- **[16 · Circuit breaker](../examples/16_circuit_breaker_demo.yaml)** — Calls a model that doesn't exist, so every call fails. When Ollama isn't running at all, the connection errors are retried until the provider's circuit breaker opens and rejects the remaining attempts. With Ollama running, the 404 is not retried and the steps fail at once.
- **[17 · Sandbox, allowed](../examples/17_sandbox_allowed.yaml)** — Enables the sandbox with only `echo` and `cat` allowed and files limited to `/tmp/agentloom`, then runs tools that stay inside it. OpenAI.
- **[18 · Sandbox, blocked](../examples/18_sandbox_blocked.yaml)** — Five tools that break the policy: a command off the list, a pipe, a path outside it. The run fails by design, and each blocked step is retried before it does. Needs no key.

<!-- /cards -->

## Attachments

Images, PDFs and audio sent with a prompt. [Providers](providers.md) lists which provider takes which.

<!-- cards -->

- **[19 · Image from a URL](../examples/19_multimodal_vision.yaml)** — AgentLoom downloads the image and sends it inline. Any vision model; the file names OpenAI's `gpt-4o`.
- **[20 · Inline image](../examples/20_multimodal_base64.yaml)** — The image is base64 data in the state: no network needed. OpenAI `gpt-4o`.
- **[21 · URL passthrough](../examples/21_multimodal_url_passthrough.yaml)** — The URL goes to the provider, which fetches it itself; for providers that accept image URLs, OpenAI and Anthropic. OpenAI `gpt-4o`.
- **[22 · Attachments in the sandbox](../examples/22_multimodal_sandbox.yaml)** — Only URLs from `allowed_domains` may be fetched. OpenAI `gpt-4o`.
- **[23 · PDF](../examples/23_multimodal_pdf.yaml)** — Analyzes a PDF that you place at `/tmp/agentloom/sample.pdf`. Anthropic or Google.
- **[24 · Audio](../examples/24_multimodal_audio.yaml)** — Transcribes and analyzes a clip that you place at `/tmp/agentloom/recording.wav`. Google, or OpenAI with `gpt-4o-audio-preview` and a WAV or MP3 file.

<!-- /cards -->

## Streaming

<!-- cards -->

- **[25 · Streaming answer](../examples/25_streaming_qa.yaml)** — A question answered token by token with `--stream`, with usage and cost tracked as usual. OpenAI.
- **[26 · Streaming with an image](../examples/26_streaming_multimodal.yaml)** — An image description streamed as it is written. OpenAI `gpt-4o`.

<!-- /cards -->

## State, checkpoints and approval

<!-- cards -->

- **[27 · Array indexes](../examples/27_array_index.yaml)** — Templates that read list items from the state, such as `{state.items[0]}`. Ollama with `phi4`.
- **[28 · Checkpoint and resume](../examples/28_checkpoint_resume.yaml)** — Run it with `--checkpoint`; if it fails, `agentloom resume <run_id>` continues from the saved state. Ollama with `phi4`.
- **[29 · Approval gate](../examples/29_approval_gate.yaml)** — Drafts an email and pauses for a person. Run it with `--checkpoint`, then `agentloom resume <run_id> --approve` or `--reject`. OpenAI.
- **[30 · Approval with a webhook](../examples/30_approval_gate_webhook.yaml)** — The same gate, which also posts to a URL when it pauses so a chat or CI system can act on it. The URL points at `localhost`, which the sandbox blocks for webhooks unless `allow_internal_webhook_targets` is set. OpenAI.

<!-- /cards -->

## Testing and replay

<!-- cards -->

- **[31 · Record and replay](../examples/31_record_and_replay.yaml)** — Run it once with `--record` against a real provider, then replay the recording offline with `agentloom replay` or `--mock-responses`. Anthropic.
- **[32 · Mock provider from YAML](../examples/32_yaml_mock.yaml)** — The same replay configured in the file, with no flags: `provider: mock` and a `responses_file`. Needs the recording that 31 makes, at `recordings/byzantine.json`.
- **[40 · Fault replay](../examples/40_fault_replay.yaml)** — The recording declares a 429 and a 503 before the answer: every run fails the same two calls and recovers on the third attempt, through the real retry path. Offline.

<!-- /cards -->

## Model features

<!-- cards -->

- **[35 · Tool calling](../examples/35_tool_calling.yaml)** — The model decides to call `http_request`, reads the result and answers. Offline; with `--provider openai --model gpt-4o-mini` the model picks the tool itself.
- **[36 · Structured output](../examples/36_structured_output.yaml)** — Classifies a message into one of four intents and returns a typed object with a confidence score, from a schema in [`structured_output_schemas.py`](../examples/structured_output_schemas.py). Offline, or live on any provider.
- **[37 · Embeddings](../examples/37_embeddings.yaml)** — Embeds a small set of documents with the `embed` step. Offline, or live with OpenAI, Google or Ollama embedding models.

<!-- /cards -->

## Conversations and agents

<!-- cards -->

- **[38 · Chatbot](../examples/38_chatbot_conversation.yaml)** — Three turns of one conversation kept in the state: each step adds a user turn and the reply. Offline.
- **[39 · Two agents talking](../examples/39_multi_agent_conversation.yaml)** — A planner and a critic take turns in one conversation, each with its own system prompt and name. Offline.
- **[41 · Agent](../examples/41_agent.yaml)** — A support agent defined once in [`agents/support.yaml`](../examples/agents/support.yaml), with its tools, a refunds agent it calls as a tool, memory and a typed reply, run for two turns. Offline.

<!-- /cards -->
