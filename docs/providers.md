---
title: "Providers"
description: "Which provider answers each step, how to configure OpenAI, Anthropic, Google, Ollama and OpenAI-compatible servers, and what each one supports."
order: 11
section: "Running"
---

# Providers

AgentLoom calls OpenAI, Anthropic, Google and Ollama over their HTTP APIs, any OpenAI-compatible server through `generic_openai`, and a `mock` provider for offline runs. This page covers which provider answers a step, how to set each one up, and what each supports.

## Which provider answers

Without a [config file](#config-file), `agentloom run` sets up a provider for each key it finds in the environment:

| Provider | Set up when | Key | Endpoint |
|----------|-------------|-----|----------|
| `openai` | `OPENAI_API_KEY` is set | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `anthropic` | `ANTHROPIC_API_KEY` is set | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| `google` | `GOOGLE_API_KEY` is set | `GOOGLE_API_KEY` | `GOOGLE_BASE_URL` |
| `ollama` | The workflow says `provider: ollama`, or `AGENTLOOM_OLLAMA_FALLBACK` is on | none | `OLLAMA_BASE_URL`, else `http://localhost:11434` |
| `generic_openai` | The workflow says `provider: generic_openai` | The variable `api_key_env` names | `base_url` |
| `mock` | The workflow says `provider: mock`, or the run passes `--mock-responses` | none | none |

**The model name picks the provider,** not `provider`:

| Provider | Serves |
|----------|--------|
| OpenAI | Names that start with `gpt-`, `o1`, `o3`, `o4`, `text-embedding-3-` or `text-embedding-ada-`. |
| Anthropic | Names that contain `claude`. |
| Google | Names that contain `gemini`, and names that start with `embedding-`. |
| Ollama, `generic_openai`, `mock` | Any name. |

`config.provider`, or `--provider`, only puts that provider first among those that serve the model. So `provider: anthropic` with the default model, `gpt-4o-mini`, still needs OpenAI: set `model` along with `provider`. An unknown `provider` name is accepted without an error, and changes nothing.

Each step runs `config.model` unless it names its own `model`. `--model` changes `config.model`, so a step's own `model` wins over it.

A model that no provider serves fails with `No provider registered for model 'llama3.2'`, after the step's retries.

### Recommended model IDs

Pin dated model IDs rather than aliases: a dated ID behaves as it did when you tested it, while an alias such as `-latest` can change or stop answering without notice.

| Provider | Recommended ID | Notes |
|----------|----------------|-------|
| Anthropic | `claude-haiku-4-5-20251001` | Not `claude-3-5-haiku-latest`: current keys get a 404 for it. |
| Google | `gemini-2.5-flash` | Not `gemini-2.0-flash`, which returns a 404 for new users. |
| OpenAI | `gpt-4o-mini`, `gpt-4o` | |

Six names belong to their provider: `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `o4-mini`, `claude-haiku-4-5-20251001` and `gemini-2.5-flash` go only to it, and never to a fallback.

### Fallback

When a provider fails, the attempt moves on to the next one that can serve the model: first the providers that serve its name, by priority, then Ollama when it is a fallback. A few limits:

- **Every provider receives the same model name.** Anthropic can't answer for `gpt-4o`, so a fallback is useful only towards a provider that takes any name, Ollama or `generic_openai`, and only if it serves a model of that exact name.
- **The six names above never fall back.**
- **A [streamed](llm-calls.md#streaming) call doesn't fall back:** a provider error fails the attempt.

When every provider fails, the error starts with `All providers failed for model`. [Reliability](reliability.md#provider-failures) covers how fallback, circuit breakers and rate limits combine with retries.

### Using Ollama

- `provider: ollama` sets up Ollama and tries it first for every model but the six names above: with an OpenAI key set, a name like `o3` goes to Ollama before OpenAI.
- `AGENTLOOM_OLLAMA_FALLBACK=1`, or `true`, `yes` or `on`, sets up Ollama as a fallback, tried after the others. Any other value leaves it off, without a warning.
- AgentLoom doesn't look for a running Ollama: without one of the two, it isn't used.
- Ollama costs 0, and its requests time out after 120 seconds.

## Config file

A `providers:` block in a config file replaces the discovery above: the CLI sets up the providers it lists, and only those.

```yaml
# agentloom.yaml
providers:
  - name: openai
    models: [gpt-4o, gpt-4o-mini]
  - name: anthropic
    priority: 10
  - name: ollama
    base_url: http://gpu-box:11434
    priority: 100
```

| Key | What it does |
|-----|--------------|
| `name` | `openai`, `anthropic`, `google` or `ollama`. Any other name, `mock` and `generic_openai` included, stops the command. |
| `models` | Names that go only to the providers that list them, by priority, and never to a fallback. |
| `priority` | The order among the providers that serve a model, lowest first. `0` by default; a tie keeps the file's order. |
| `base_url` | The endpoint. Without it, the provider's `*_BASE_URL` variable, then its default. See [Endpoints](#endpoints). |
| `is_fallback` | Also tries the provider for names it doesn't serve, after the others. |

- **Keys stay in the environment.** Each provider reads its usual variable, and an `api_key` in the file stops the command, so the file can be committed. A provider listed without its key is set up all the same, and its calls fail with the provider's `401`.
- **Ollama serves any name,** so it is a candidate for every model that `models` doesn't pin: give it a higher `priority` than the others, or it is tried first. It needs no `AGENTLOOM_OLLAMA_FALLBACK`, and the file must list it, even for a workflow that says `provider: ollama`.
- **Only `models` pins a name:** the six names above go to fallbacks like any other.
- **`config.provider` and `--provider` don't change the order;** `priority` does.
- **`is_fallback` adds little:** every provider receives the same name, and Ollama, which serves any name, is a candidate already.
- **Besides `providers`,** the file accepts `log_level`, `log_format`, `observability_enabled`, `default_provider`, `max_concurrent_steps`, `budget_limit_usd`, `checkpoint_enabled` and `checkpoint_dir`, which change nothing. Any other key, or a file that doesn't parse, stops the command with a one-line error and exit `1`.

`run`, `resume`, `replay` (for the steps a cut runs live) and `callback-server` read the file that `--config` names, else the one `AGENTLOOM_CONFIG` names, else `agentloom.yaml` in the current directory, when there is one. A `--config` or `AGENTLOOM_CONFIG` that names a missing file stops the command. `callback-server` reads the file when it starts, and stops if it doesn't load. A workflow with `provider: mock` or `provider: generic_openai` doesn't use it.

## Endpoints

`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL` and `OLLAMA_BASE_URL` point a provider at a proxy or a self-hosted gateway. Each provider treats the URL its own way:

| Provider | The URL | Example |
|----------|---------|---------|
| OpenAI, `generic_openai` | A bare host gets `/v1`; any other path is kept as it is. A trailing `/` is dropped. | `http://vllm:8000` becomes `http://vllm:8000/v1`; `https://gw.example.com/openai` stays. |
| Anthropic | Gets `/v1`, unless it already ends in `/v1`. | `https://gw.example.com/anthropic` becomes `…/anthropic/v1`; `…/v2` becomes `…/v2/v1`. |
| Google | Used as it is: include `/v1beta`. | `https://generativelanguage.googleapis.com/v1beta` |
| Ollama | Used as it is. | `http://gpu-box:11434` |

The `*_BASE_URL` variables apply to the CLI. From Python, pass `base_url` to the provider, except for Ollama, which reads `OLLAMA_BASE_URL` itself.

Requests honour `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY` and `SSL_CERT_FILE`. Keys go in the provider's usual header, `Authorization: Bearer`, `x-api-key` or `x-goog-api-key`; custom headers can't be added.

## OpenAI-compatible servers

`generic_openai` reaches any server with an OpenAI-style API, such as vLLM, LM Studio or a hosted gateway:

```yaml
config:
  provider: generic_openai
  base_url: http://localhost:8000
  api_key_env: VLLM_API_KEY      # optional
  model: meta-llama/Llama-3.1-8B-Instruct
```

- `base_url` is required with `generic_openai`, and refused with any other provider.
- `api_key_env` names the environment variable that holds the key. If that variable isn't set, the run stops before it starts. Without `api_key_env`, requests carry no key; `OPENAI_API_KEY` is never sent.
- It is the only provider of the run: every model goes to it, and other keys and `AGENTLOOM_OLLAMA_FALLBACK` are ignored.
- It supports what OpenAI does: streaming, tools, structured output, attachments in OpenAI's format and embeddings, as far as the server does. Streaming sends `stream_options`, which some servers refuse.
- Its cost comes from the price table, by model name: a model named like a cloud model is priced like it, and other names cost 0.
- `--provider generic_openai` works only on a file that sets `generic_openai` already, and `--provider` with any other provider drops the file's `base_url`.

## Capabilities

| | OpenAI, `generic_openai` | Anthropic | Google | Ollama |
|---|---|---|---|---|
| [Streaming](llm-calls.md#streaming) | Yes | Yes | Yes | Yes |
| [Tool calling](llm-calls.md#tool-calling) | Yes | Yes | Yes | Yes, without `tool_choice` |
| [Structured output](llm-calls.md#structured-output) | Native | Through the prompt | Native | Native, as the model allows |
| [Images](llm-calls.md#attachments) | Yes, by URL too | Yes, by URL too | Yes | Yes |
| PDF | No | Yes | Yes | No |
| Audio | WAV and MP3 | No | Yes | No |
| [Reasoning](llm-calls.md#reasoning) | Built into the model | Yes | Yes | Yes |
| Reasoning tokens reported | Yes | In the reply's tokens | Yes | In the reply's tokens |
| [Embeddings](steps.md#embed) | Yes | No | Yes | Yes, without `dimensions` |
| Cost | Price table | Price table | Price table | 0 |

[Prices](reliability.md#prices) covers the price table and how to add models to it.

### OpenAI

- Chat Completions, at `{base_url}/chat/completions`. Requests time out after 60 seconds.
- `temperature` and `max_tokens` are sent as given, so leave them unset for reasoning models; `max_completion_tokens` and `reasoning_effort` can't be sent.
- Reasoning tokens are reported apart from the reply's.
- Embeddings go in batches of 100, with `dimensions`.

### Anthropic

- The Messages API, at `{base_url}/messages`, with `anthropic-version: 2023-06-01`. Requests time out after 60 seconds.
- `max_tokens` is required by the API: AgentLoom sends 4096 when it is unset, so raise it above a reasoning budget.
- All system messages are joined into one.
- No embeddings: an `embed` step skips Anthropic and goes to the next provider.

### Google

- The Gemini API, at `{base_url}/models/{model}:generateContent`. Requests time out after 60 seconds.
- Reasoning tokens are reported whenever Gemini returns them.
- Embeddings go in one batch, with `dimensions`. Gemini returns no token count for them: AgentLoom estimates about 1.3 tokens a word, and prices that.

### Ollama

- `{base_url}/api/chat`. Requests time out after 120 seconds.
- `<think>` blocks are removed from replies.
- Embeddings ignore `dimensions`.

## Errors

| The provider returns | What AgentLoom raises | Retried |
|----------------------|-----------------------|---------|
| 429 | `RateLimitError`, with the status | Yes, with the step's own waits: `Retry-After` isn't followed. |
| Any other error status | `ProviderError`, with the status and the provider's message | When the status is in [`retryable_status_codes`](reliability.md#retries). |
| No answer: a refused connection, a timeout | An error without a status | Yes. |
| An attachment it can't take | `ProviderError`, before any request | Yes. See [Attachments](llm-calls.md#attachments). |

When every provider fails, the gateway's error keeps a status only if they all returned the same one. [What is retried](reliability.md#what-is-retried) covers the rest.

## Response metadata

Each step's result records what answered:

- `model`, as the provider reports it: OpenAI's dated name, such as `gpt-4o-2024-08-06`, Ollama's tag, such as `llama3.2:latest`, and Google's requested name;
- `provider`, the one that answered, which shows a fallback;
- `token_usage`, with `reasoning_tokens`, and `cost_usd`;
- `prompt_metadata.finish_reason`, as the provider wrote it: `stop`, `end_turn`, `STOP`.

`agentloom run --json` prints them. A failed step has no `model` or `provider`.

From Python, a provider's `ProviderResponse.meta` adds what the call sent and got back besides the answer: the `temperature`, `top_p` and `seed` sent, the provider's identifiers, its rate-limit headers and `received_at`. It reaches [recordings](testing-and-replay.md#recording-a-run) too, but not results or `--json`.

| Provider | `system_fingerprint` | `model_version` | `request_id` |
|----------|----------------------|-----------------|--------------|
| OpenAI, `generic_openai` | From the body | | The `x-request-id` header |
| Anthropic | | | The `request-id` header |
| Google | | `modelVersion` | `responseId` |
| Ollama | | | |

## Environment variables

| Variable | What it does |
|----------|--------------|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | Set up the provider, with that key. AgentLoom reads no `.env` file. |
| `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`, `OLLAMA_BASE_URL` | The provider's endpoint. See [Endpoints](#endpoints). |
| `AGENTLOOM_OLLAMA_FALLBACK` | `1`, `true`, `yes` or `on` sets up Ollama as a fallback. |
| `AGENTLOOM_CONFIG` | The [config file](#config-file), when the command has no `--config`. |
| The variable `api_key_env` names | The key of a `generic_openai` server. |
| `AGENTLOOM_PRICING_FILE` | A price table that replaces the bundled one. See [Prices](reliability.md#prices). |
| `AGENTLOOM_CANDIDATE_CACHE_MAX` | How many model names the gateway remembers the providers of. 1024 by default. |

> [!WARNING]
> `AGENTLOOM_LOG_LEVEL`, `AGENTLOOM_LOG_FORMAT`, `AGENTLOOM_OBSERVABILITY`, `AGENTLOOM_DEFAULT_PROVIDER`, `AGENTLOOM_MAX_CONCURRENT_STEPS`, `AGENTLOOM_BUDGET_LIMIT`, `AGENTLOOM_CHECKPOINT` and `AGENTLOOM_CHECKPOINT_DIR` are read but change nothing. An invalid value in one of them, such as `AGENTLOOM_CHECKPOINT=maybe`, stops every run with a traceback: unset them.

## From Python

The CLI's setup doesn't apply to a Python program, which registers its providers on a `ProviderGateway` itself, with their priorities, fallbacks, rate limits and circuit breakers. See [The provider gateway](python-dsl.md#the-provider-gateway).

## What's next

<!-- cards -->

- **[LLM calls](llm-calls.md)** — Attachments, reasoning, tool calling and structured output, provider by provider.
- **[Reliability](reliability.md)** — Retries, fallback, circuit breakers, rate limits and prices.
- **[Testing & replay](testing-and-replay.md)** — The `mock` provider and recordings.

<!-- /cards -->
