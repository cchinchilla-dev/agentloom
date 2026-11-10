---
title: "LLM calls"
description: "The llm_call step: prompts, attachments, reasoning, tool calling, structured output, conversations and streaming, and what each one writes."
order: 6
section: "Workflows"
---

# LLM calls

An `llm_call` step sends one request to a model and writes the reply to the state. Beyond a prompt, it can send images, PDFs and audio, ask for reasoning, let the model call tools, hold the reply to a JSON shape and keep a conversation going across steps.

## A call

```yaml
- id: summarize
  type: llm_call
  depends_on: [fetch]
  system_prompt: "You write short, factual summaries."
  prompt: "Summarize in three sentences: {state.report.body}"
  model: gpt-4o
  temperature: 0.2
  max_tokens: 400
  output: summary
```

| Field | Default | What it does |
|-------|---------|--------------|
| `prompt` | required | The user turn, a [template](state-and-templates.md#templates). Optional only with a `conversation`. |
| `system_prompt` | none | The system message, a template too. |
| `model` | `config.model` | The model of this step. It wins over `--model`, which only changes `config.model`. |
| `temperature` | the provider's | Sent as given, without a range check. |
| `max_tokens` | the provider's | The reply's token limit. Anthropic requires one, so AgentLoom sends 4096 when it is unset, plus the reasoning budget if there is one. |
| `stream` | `config.stream` | Stream the reply as it is written. See [Streaming](#streaming). |
| `attachments` | none | Images, PDFs and audio sent with the prompt. See [Attachments](#attachments). |
| `thinking` | none | Reasoning settings. See [Reasoning](#reasoning). |
| `tools`, `tool_choice`, `max_tool_iterations` | none, `auto`, `5` | Tools the model may call. See [Tool calling](#tool-calling). |
| `response_schema` | none | A JSON shape for the reply. See [Structured output](#structured-output). |
| `conversation`, `speaker` | none | A conversation kept in the state. See [Conversations](#conversations). |

The step also takes the [fields every step has](workflow-files.md#fields-every-step-has): `output`, `timeout`, `retry` and `depends_on`.

### What the step writes

- **`output`** receives the reply's text, or the parsed value with a `response_schema`. With tools, it is the text of the model's last reply.
- **`steps.<id>.output`** always holds the raw reply text: with a `response_schema`, that is the reply as the model wrote it, any words around the JSON included.
- **The step's result** carries the tokens, reasoning tokens included, the cost, the model and provider that answered, and the provider's own `finish_reason`, such as `stop`, `end_turn` or `STOP`. `agentloom run --json` prints it.

Tokens and cost cover every call the step made: each turn of a tool loop, each re-ask for valid JSON and each conversation summary.

### When a call fails

| Failure | What happens |
|---------|--------------|
| Neither `prompt` nor `conversation` | Fails at once, without calling the provider. |
| Broken braces in `prompt` or `system_prompt` | Fails at once with `Prompt template error`. Literal braces are written twice; see [Templates](state-and-templates.md#templates). |
| A provider status in `retry.retryable_status_codes` (429, 500, 502, 503 and 504 by default), or an error with no status, such as a dropped connection | Retried, with backoff. |
| Any other status, such as 400, 401, 403 or 404 | Fails at once. |

[Reliability](reliability.md) covers retries, timeouts and what a failed step does to the rest of the run.

## Attachments

```yaml
- id: describe
  type: llm_call
  model: gpt-4o
  prompt: "What is in this picture, and is the receipt total legible?"
  attachments:
    - source: "{state.receipt_url}"          # downloaded by AgentLoom
    - type: pdf
      source: invoices/2026-05.pdf           # a local file
  output: description
```

| Field | Default | What it is |
|-------|---------|------------|
| `type` | `image` | `image`, `pdf` or `audio`. |
| `source` | required | A URL, a local file path, a `data:` URL or a base64 string. It is a template. |
| `media_type` | guessed | The MIME type, such as `image/jpeg`. Without it, AgentLoom guesses from the file extension, then from a download's `Content-Type`, then falls back to `image/png`, `application/pdf` or `audio/wav`. A `data:` URL's own type wins over this field. One that contradicts `type`, such as `application/pdf` on an `image`, fails the step. |
| `fetch` | `local` | For URLs: `local` downloads the file and sends its bytes; `provider` sends the URL itself, for the provider to fetch. |

How the `source` is read:

| Source | What happens |
|--------|--------------|
| `https://…` with `fetch: local` | Downloaded with a 30-second timeout, following up to 5 redirects. |
| `https://…` with `fetch: provider` | Sent as a URL. Images only, and only OpenAI and Anthropic take it. |
| `data:image/png;base64,…` | Decoded; nothing is fetched. |
| A long base64 string | Sent as it is. Set `media_type`. |
| Anything else | A file path, relative to the directory you run from. |

- A file, a download, a `data:` URL or a base64 string may be up to 20 MB.
- URLs at private, loopback or link-local addresses are refused, with either `fetch`.
- The [sandbox](tools-and-sandbox.md) applies. `sandbox.allow_network: false` and `sandbox.allowed_domains` restrict downloads even when the sandbox is disabled. With the sandbox enabled, a local file must sit under `readable_paths` or `allowed_paths`.

| Provider | Image | Image by URL | PDF | Audio |
|----------|-------|--------------|-----|-------|
| OpenAI, `generic_openai` | Yes | Yes | No | WAV and MP3 |
| Anthropic | Yes | Yes | Yes | No |
| Google | Yes | No | Yes | Yes |
| Ollama | Yes | No | No | No |

An attachment the provider can't take, such as a PDF for OpenAI, fails before any request is sent. The call moves on to the next provider for the model, and the step fails without a retry when none can take it.

In a [conversation](#conversations), attachments go with this step's user turn only and aren't stored in the conversation. A conversation step without a `prompt` has no user turn, and sends no attachments.

## Reasoning

```yaml
- id: plan
  type: llm_call
  model: claude-sonnet-4-5
  max_tokens: 8000                 # above the budget
  prompt: "Plan the migration of {state.service} to the new queue."
  thinking:
    enabled: true
    budget_tokens: 4000
  output: plan
```

| Field | Default | What it is |
|-------|---------|------------|
| `enabled` | `false` | Turns reasoning on. Without it, nothing else in the block is sent. |
| `budget_tokens` | none | The most tokens the model may spend reasoning, where the provider takes a budget. |
| `level` | none | `low`, `medium` or `high`, where the provider takes a level. |
| `capture_reasoning` | `true` | Whether the provider returns its reasoning text, and the step's result keeps it. |

| Provider | What AgentLoom sends |
|----------|----------------------|
| OpenAI | `reasoning_effort`, set to the `level`, on the o-series and GPT-5 models, which reason on their own, except `o1-mini` and `gpt-5-chat`. Nothing without a `level`, or on other models. `budget_tokens` is ignored. |
| Anthropic | `thinking` with `budget_tokens`, which a step on a Claude model must set, or it fails at load. `level` is ignored. Without `max_tokens`, AgentLoom sends the budget plus 4096; one that isn't above the budget fails at load. |
| Google | `thinkingConfig` with the budget, the level and whether to include thoughts. |
| Ollama | `think`, set to the `level`, or to `true` without one. `budget_tokens` is ignored. |

- Reasoning tokens count in the step's tokens and are billed at the output price. OpenAI and Google report them apart; Anthropic and Ollama count them in the reply's tokens.
- With `capture_reasoning` on, the reasoning text reaches the step's result as `reasoning_content`, where `agentloom run --json` prints it, and is redacted when the step's `output` is. It doesn't reach the state or a recording. OpenAI doesn't return it.
- OpenAI's reasoning models get `max_tokens` as `max_completion_tokens`, and no `temperature`. `gpt-5-chat` isn't one of them: it gets both as given.
- [Agents](agents.md) don't take a `thinking` block.

## Tool calling

The model decides which tools to call, and with what arguments. AgentLoom runs them and sends back their results until the model answers.

```yaml
- id: ask
  type: llm_call
  prompt: "{state.question}"
  tools:
    - name: http_request
      description: "Make an HTTP request to a given URL and return the response body."
      parameters:
        type: object
        properties:
          url: { type: string, description: "Full URL to GET." }
          method: { type: string, enum: ["GET", "POST"] }
        required: [url]
  tool_choice: auto
  max_tool_iterations: 3
  output: answer
```

| Field | Default | What it is |
|-------|---------|------------|
| `tools` | none | Each tool's `name`, `description` and `parameters`, a JSON Schema. This is what the model sees. A tool without `parameters` is sent with the registered tool's own. |
| `tool_choice` | `auto` | `auto` lets the model decide, `none` turns tools off, `required` makes its first call use a tool, and `{name: http_request}` makes its first call use that one. |
| `max_tool_iterations` | `5` | The most model calls the step makes, the one that answers included. |

The `name` must match a registered tool. `agentloom run` registers the four [built-in tools](tools-and-sandbox.md): `http_request`, `shell_command`, `file_read` and `file_write`. Tools of your own are registered from Python. A registered tool that the step doesn't declare stays out of the model's reach.

### The loop

1. AgentLoom calls the model with the tools.
2. A reply without tool calls is the answer: the step succeeds with it.
3. Otherwise, AgentLoom runs every call in the reply at once, sends the results back in the order of the calls, and calls the model again.
4. At `max_tool_iterations`, the step still succeeds, with the last reply's text and `finish_reason: max_tool_iterations`. The tools that last reply asked for don't run, as no call would read their results: with `max_tool_iterations: 1`, no tool ever runs.

- **A failing tool doesn't fail the step.** An unknown tool, arguments that aren't valid JSON, an exception or a sandbox refusal go back to the model as the tool's result, and the loop goes on. Arguments aren't checked against `parameters`.
- **`required` and a named tool apply to the first call only.** The later calls of the loop use `auto`, so the model can read the results and answer. A model that keeps calling tools up to `max_tool_iterations` and leaves an empty reply ends with `finish_reason: max_tool_iterations_no_answer`.
- **A retry runs the whole loop again,** the tools included. A tool with side effects, such as a write, a message or a payment, should be safe to repeat, or the step should set `retry: {max_retries: 0}`.
- **Tools and a `response_schema` can't share a step.** An [agent](agents.md) with an `output_schema` combines them.
- **A step with tools doesn't stream:** its answer reaches the stream in one piece at the end.

OpenAI, Anthropic, Google, Ollama and `generic_openai` all take tools; Ollama ignores `tool_choice`, and whether a local model calls tools depends on the model. A model calling a tool is the dynamic counterpart of a [`tool` step](tools-and-sandbox.md), where the workflow picks the tool.

## Structured output

`response_schema` holds the reply to a JSON shape, and the step writes the parsed value to the state.

<!-- tabs -->

**Pydantic model**

```yaml
- id: classify
  type: llm_call
  prompt: "Classify the intent of: {state.message}"
  response_schema:
    type: pydantic
    model: examples.structured_output_schemas.Classification
  output: classification          # a Classification instance
```

**JSON Schema**

```yaml
- id: extract
  type: llm_call
  prompt: "Extract the name and date from: {state.text}"
  response_schema:
    type: json_schema
    schema:
      type: object
      properties:
        name: { type: string }
        date: { type: string, format: date }
      required: [name, date]
  output: extracted               # a dictionary
```

**Any JSON object**

```yaml
- id: ideas
  type: llm_call
  prompt: "Suggest five names for {state.product}, as a JSON object."
  response_schema:
    type: json_object
  output: ideas
```

<!-- /tabs -->

| Field | Default | What it is |
|-------|---------|------------|
| `type` | `json_object` | `pydantic`, `json_schema` or `json_object`. |
| `model` | required for `pydantic` | The model class, as a dotted path: `package.module.Class` or `package.module:Class`. `agentloom run` imports it from the directory you run from. |
| `schema` | required for `json_schema` | An inline JSON Schema. `additionalProperties: true` anywhere in it is refused. |
| `name` | the class name, else the step id | The schema's name, which OpenAI requires. |
| `strict` | `true` | OpenAI's strict mode, which also makes every property required. |
| `description` | none | A description of the schema, for OpenAI. |

| Provider | How the shape is enforced |
|----------|---------------------------|
| OpenAI | Native: `response_format` with the schema, or a JSON object. |
| Google | Native: `responseSchema`, adapted to what Gemini's schema dialect accepts, with the reply validated against the full schema. |
| Ollama | `format`, with the schema or as plain JSON; how well it holds depends on the model. |
| Anthropic | No native mode: the schema goes into the system prompt and the reply is prefilled with `{`. |

**Validation.** AgentLoom reads the JSON from the reply, from a fenced JSON block in it, or from its first `{…}`, and validates it. An invalid reply goes back to the model with the validation error, and the model tries again, up to `retry.max_retries` more calls within the same attempt. If none is valid, the step fails without a retry. A streamed reply is validated once it ends, and a failure there fails the step without a retry.

**Storage.** `output` receives a Pydantic instance or a dictionary, which templates read like any other value: `{state.classification.label}`. Checkpoints and `--json` write it as a plain dictionary, so a resumed run reads a dictionary. Redaction patterns reach the fields of a Pydantic instance, as those of a dictionary.

## Conversations

A conversation is a list of messages kept in the state, which several steps continue. Each step adds its user turn and the model's reply.

```yaml
state:
  chat:
    messages:
      - { role: system, content: "You are a concise, cheerful assistant." }
    token_budget: 800
    trim_policy: drop_pairs

steps:
  - id: turn_1
    type: llm_call
    conversation: state.chat
    prompt: "Recommend one Python library for CLI apps."
    output: reply_1

  - id: turn_2
    type: llm_call
    depends_on: [turn_1]
    conversation: state.chat
    prompt: "Any second choice for larger apps?"
    output: reply_2
```

`conversation` names a state key, as `state.chat` or `chat`. A key that doesn't exist yet starts an empty conversation, and a plain list of messages is accepted as one. The key holds:

| Field | What it is |
|-------|------------|
| `messages` | The turns: `role` (`system`, `user`, `assistant` or `tool`), `content`, and `name` for a speaker's turns. |
| `token_budget` | The most tokens the messages and the summary may take; no limit without one. |
| `trim_policy` | How to make room: `drop_oldest` (the default), `drop_pairs` or `summarize_oldest`. |
| `summary` | What `summarize_oldest` has folded in so far. |

### A turn

1. A `system_prompt` is stored as the first message if the conversation has none. If it already has one, a later step's different `system_prompt` is ignored.
2. The `prompt` is added as a user turn. Without a `prompt`, the last message must already be the user's or a tool's.
3. The conversation is trimmed to its budget.
4. The model answers from the whole conversation, and its reply is added as an assistant turn.

- Only the final text is stored. The calls and results of a tool loop, and the re-asks for valid JSON, aren't. A structured reply is stored as the model wrote it.
- The conversation and `output` are written only when the step succeeds.
- Two steps of one layer on one conversation are refused when the file loads: each would write its own copy, and one of the turns would be lost. Chain them with `depends_on`. Branches of one router are exempt, because only one of them runs.

### Trimming

The budget is counted roughly: about one token per four characters, plus a little for each message's role and name. Leading system messages always stay.

- **`drop_oldest`** removes the oldest messages until the rest fits. If the system messages and the summary alone reach the budget, nothing is removed, and the request goes over.
- **`drop_pairs`** removes the oldest user and assistant pairs. If the system messages and the summary alone reach the budget, nothing is removed, and the request goes over.
- **`summarize_oldest`** folds the oldest messages into `summary`, with one extra call to the step's model, and sends the summary as a system message. If that call fails, the messages are joined as plain text instead. The summary counts in the budget and isn't trimmed: once it and the system messages reach the budget, nothing more is folded, and the request goes over.

Trimmed messages leave the stored conversation too. A budget too small to hold the new user turn fails the step at once.

### Several speakers

`speaker` names the assistant turns of a step, so several voices can share one conversation. It takes letters, digits, `_` and `-`, up to 64 characters:

```yaml
- id: planner
  type: llm_call
  conversation: state.debate
  speaker: alice
  system_prompt: "You propose plans. Be concrete."
  prompt: "Propose a rollout plan for {state.feature}."
  output: plan

- id: critic
  type: llm_call
  depends_on: [planner]
  conversation: state.debate
  speaker: bob
  system_prompt: "You find the weakest point of a plan."
  output: critique
```

With a `speaker`, the step's `system_prompt` applies to its own turn only and isn't stored, so each speaker keeps its own instructions. OpenAI receives the speaker's name as the message's `name`; the other providers see it as a `[alice]` prefix on the turn. [Agents](agents.md) are speakers by their name.

## Streaming

`stream: true` on a step, or `config.stream: true` for every step, asks the provider for the reply as it is written. `agentloom run --stream` sets `config.stream` for one run; a step's own `stream: false` still wins.

- `agentloom run --stream` prints the text as it arrives, without step names, so parallel steps interleave. With `--json`, nothing is printed until the end. Without `--stream`, steps that stream print nothing as they go.
- A streaming step records its time to first token in its result and metrics.
- Steps with tools, [agent](agents.md) turns and conversation summaries don't stream: their text arrives in one piece. A structured reply streams as raw JSON and is validated at the end.
- A streamed call falls back to another provider only before its first chunk. After that, a provider error fails the attempt, and the step's `retry` decides what comes next. A 429 doesn't count against the provider's circuit breaker.

## What's next

<!-- cards -->

- **[Agents](agents.md)** — A model, its tools and its rules, defined once and run as a step.
- **[Providers](providers.md)** — Each provider's models, capabilities and pricing.
- **[Tools & sandbox](tools-and-sandbox.md)** — The built-in tools, tools of your own, and what the sandbox allows.

<!-- /cards -->
