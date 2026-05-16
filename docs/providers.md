# Providers

AgentLoom ships with four providers. The gateway routes requests based on model name and falls back automatically when a provider is unavailable.

## Capability matrix

| Capability | OpenAI | Anthropic | Google | Ollama |
|---|---|---|---|---|
| Models | `gpt-*`, `o3*`, `o4*` | `claude*` | `gemini*` | Any local model |
| Streaming | SSE | SSE | SSE | NDJSON |
| Image input | :material-check: | :material-check: | :material-check: | :material-check: |
| PDF input | :material-close: | :material-check: | :material-check: | :material-close: |
| Audio input | :material-check: | :material-close: | :material-check: | :material-close: |
| Reasoning token count | :material-check: (o-series, implicit) | :material-close: (rolled into `output_tokens`) | :material-check: (Gemini 2.5+, opt-in) | :material-close: (no `eval_count` split) |
| Reasoning content (trace) | :material-close: (server-side only) | :material-check: (`type="thinking"` blocks) | :material-check: (`includeThoughts` opt-in) | :material-check: (Ollama 0.9+ `message.thinking`) |
| Cost tracking | :material-check: | :material-check: | :material-check: | Free (local) |

## Configuration

Switch provider in any workflow:

```yaml
config:
  provider: google
  model: gemini-2.5-flash
```

Or override at runtime via CLI:

```bash
agentloom run workflow.yaml --provider anthropic --model claude-sonnet-4-20250514
```

## Environment variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google |
| `OLLAMA_BASE_URL` | Ollama (default: `http://localhost:11434`) |

## Circuit breaker

The gateway wraps each provider with a circuit breaker:

| State | Behavior | Transition |
|-------|----------|------------|
| **Closed** | Requests pass through normally | :material-arrow-right: Open after 5 consecutive failures |
| **Open** | Requests rejected immediately, fallback provider used | :material-arrow-right: Half-open after 60s |
| **Half-open** | One test request allowed | :material-arrow-right: Closed on success, Open on failure |

`RateLimitError` (HTTP 429) and stream cancellations (`GeneratorExit` / `anyio.CancelledError`) are excluded from the failure count — being throttled or aborted is not a provider outage. Only genuine errors count toward the 5-failure threshold.

## Rate limiter

Dual token-bucket rate limiting per provider:

- **Requests per minute** — default 60 RPM
- **Tokens per minute** — default 100,000 TPM

```python
gateway.register(
    provider,
    max_rpm=120,          # requests/minute
    max_tpm=200_000,      # tokens/minute
)
```

`max_rpm` and `max_tpm` must be `>= 1`; the limiter rejects zero/negative bounds at registration with `ValueError`. A request whose estimated `token_count` exceeds `max_tpm` also raises `ValueError` instead of blocking forever on a bucket that can never refill that high — this is a local precondition violation, not a `RateLimitError` (which is reserved for HTTP 429 responses from the provider).

## HTTP errors

All provider adapters normalize remote errors to a common taxonomy:

| HTTP status | Exception | Notes |
|-------------|-----------|-------|
| `429 Too Many Requests` | `RateLimitError` | Numeric `Retry-After` (seconds) is parsed and exposed on the exception. HTTP-date form is not supported — providers we talk to use integer seconds. |
| `5xx` | `ProviderError` | Counts toward the circuit breaker |
| network / timeout | `ProviderError` | Counts toward the circuit breaker |

Provider adapters declare an explicit kwargs allowlist for `extra` parameters; unknown kwargs raise a `TypeError` at call time rather than silently reaching the vendor's API. Each adapter exposes its allowlist via a constant (`_OPENAI_EXTRA_PAYLOAD_KEYS`, `_ANTHROPIC_EXTRA_PAYLOAD_KEYS`, `_GOOGLE_GEN_CONFIG_KEYS` + `_GOOGLE_TOPLEVEL_KEYS`, `_OLLAMA_OPTION_KEYS` + `_OLLAMA_TOPLEVEL_KEYS`).

## Fallback chain

Providers are tried in priority order. Register multiple providers for automatic fallback:

```python
gateway.register(openai_provider, priority=0)
gateway.register(anthropic_provider, priority=1, is_fallback=True)
gateway.register(ollama_provider, priority=2, is_fallback=True)
```

If OpenAI fails or its circuit breaker trips, the gateway automatically routes to Anthropic. If Anthropic also fails, it falls back to Ollama.

## Multi-modal attachments

LLM steps support image, PDF, and audio attachments:

```yaml
steps:
  - id: analyze
    type: llm_call
    prompt: "Describe what you see in this image."
    attachments:
      - type: image
        source: "{state.image_url}"
        fetch: local
    output: description
```

| Field | Description |
|-------|-------------|
| `type` | `image`, `pdf`, or `audio` |
| `source` | URL, local file path, or base64 data |
| `media_type` | Optional; inferred from type if omitted |
| `fetch` | `local` (engine downloads) or `provider` (provider fetches URL directly) |

!!! warning "Provider support varies"
    Check the [capability matrix](#capability-matrix) above. Sending a PDF to OpenAI or audio to Anthropic will raise a `ProviderError`.

## Reasoning models

OpenAI o-series (`o1`, `o3`, `o4-mini`) and Anthropic Claude with extended thinking produce internal *reasoning tokens* before the final answer. Providers bill these at the output rate, so cost accounting must include them.

`TokenUsage` exposes the count alongside the usual fields:

```python
usage.prompt_tokens          # input
usage.completion_tokens      # visible output
usage.reasoning_tokens       # provider-side chain-of-thought
usage.billable_completion_tokens  # completion + reasoning
```

`calculate_cost()` charges `(prompt × input_rate) + ((completion + reasoning) × output_rate)` automatically, so workflow budgets and Prometheus cost metrics reflect the true spend.

**OpenAI** — reasoning is implicit when an o-series model is selected. The adapter parses `completion_tokens_details.reasoning_tokens` from the response. The chain-of-thought trace is kept server-side and is never returned, so `ProviderResponse.reasoning_content` stays `None`.

**Anthropic** — extended thinking is opt-in via the step-level `thinking` block (see [workflow YAML](workflow-yaml.md#llm_call)). `ThinkingConfig` translates to the `thinking: {type: "enabled", budget_tokens}` request payload, and `type="thinking"` content blocks are concatenated into `ProviderResponse.reasoning_content`. The Anthropic API does **not** surface a separate thinking-token count — extended-thinking volume is rolled into `usage.output_tokens` per the [Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) — so `reasoning_tokens` stays `0` for this provider. Cost is automatically correct because the output rate is applied to `output_tokens` which already includes the thinking volume.

**Google Gemini 2.5+** — opt-in via the same `thinking` block. `ThinkingConfig` translates to `generationConfig.thinkingConfig` with `thinkingBudget` (from `budget_tokens`), `thinkingLevel` (from `level`), and `includeThoughts` (from `capture_reasoning`). The adapter parses `usageMetadata.thoughtsTokenCount` (defaulting to `0` when the field is absent — Gemini omits it for non-thinking models and intermittently on `gemini-3-flash-preview`). When `includeThoughts=true`, parts marked `thought=true` are split into `reasoning_content` so the visible `content` stays clean.

**Ollama 0.9+** — opt-in via `thinking`. `ThinkingConfig` translates to the top-level `think` request parameter (`<level>` when `level` is set, else `true`). The adapter surfaces `message.thinking` on `reasoning_content`. As a fallback for older models or calls without `think=true`, the adapter strips inline `<think>...</think>` tags from `content` and surfaces the captured trace the same way.

!!! warning "Ollama caveat — no token split"
    Ollama exposes a single `eval_count` for all output tokens regardless of whether thinking is active, so `reasoning_tokens` always reports `0` for this provider. Cost is unaffected (local models are free), but `billable_completion_tokens` will not reflect the true thinking volume.

## Security

### SSRF protection

URL-based attachments (`fetch: local`) are protected against Server-Side Request Forgery. The engine blocks requests to private and reserved IP ranges (RFC 1918, loopback, link-local) before any network call is made.

### Webhook destination gate

Approval-gate webhooks (`approval_gate.notify.url`) pass through two independent gates:

1. **Scheme gate** — always on. Non-`http`/`https` schemes (`file://`, `data:`, `javascript:`, `gopher://`, ...) are refused regardless of opt-in flags.
2. **Internal-host gate** — always on unless the workflow explicitly opts out. Blocks loopback (`127.0.0.0/8`, `::1`, `localhost`, `*.localhost`), link-local (`169.254.0.0/16` — covers AWS / GCP / Azure metadata service at `169.254.169.254`; `fe80::/10`), RFC 1918 (`10/8`, `172.16/12`, `192.168/16`), CGNAT (`100.64/10`), ULA (`fc00::/7`), the unspecified addresses (`0.0.0.0`, `::`), multicast, and reserved ranges. IPv4-mapped IPv6 forms (`::ffff:127.0.0.1`) are normalised first so they hit the same flags.

Hostname classification uses `getaddrinfo` so both A and AAAA records are inspected — an attacker can't smuggle a loopback target through an AAAA-only DNS response. Percent-encoded and IDN hostnames are decoded before the literal-string check.

When the sandbox is enabled, the URL must additionally satisfy `allow_network`, `allowed_schemes`, and `allowed_domains`. Workflows that genuinely need to notify an in-cluster service can waive **only** the internal-host gate via:

```yaml
config:
  sandbox:
    allow_internal_webhook_targets: true
```

The opt-in does NOT widen the scheme gate — `file://` and friends stay refused. A blocked webhook is logged and emitted as a `status="sandbox_blocked"` observer breadcrumb; the approval gate itself still pauses normally because pause and notify are independent.

### Router expression boundary

Router conditions are AST-validated against an allowlist (`==`, `and`/`or`, safe builtins like `len`). Dunder and underscored attributes are rejected on both `state.foo` and `state['foo']` so a workflow author who seeds state with `_secret` cannot accidentally surface it through a router predicate. Subscript slices must be literal int/str — variables (`state[lookup]`), arithmetic (`state['_' + 'secret']`), conditionals, and calls are refused because the validator cannot determine the resulting key at parse time. Numeric slicing with optional unary `±` bounds works (`state['items'][::-1]`, `state['items'][-2:]`). Attribute calls (`state.label.strip().lower()`) keep working but `eval`, `__import__`, comprehensions, lambdas, and starred unpacking remain blocked.

State is reachable from a router predicate **only** through the `state.X` (or `state['X']`) surface — top-level state keys are not exposed as bare names, so a state key called `len` cannot shadow the safe builtin.

### Allowed paths

`sandbox.allowed_paths` grants both read and write access to a directory tree; `readable_paths` and `writable_paths` narrow it down per direction. Resolved paths must live inside an allowed prefix, and the resolution itself is wrapped — null bytes, oversized components, symlink loops, OS-level rejections, and non-string callers all surface as `SandboxViolationError` (not the raw `ValueError` / `OSError` / `RuntimeError` / `TypeError`).

Command argument validation also covers flag-embedded paths: `tee --output=/etc/passwd`, `dd of=/dev/sda`, and similar `--key=value` / `key=value` forms have their value side validated against the allowlist, not just bare positional tokens.

!!! warning "Avoid mounting `/dev`"
    `allowed_paths: ["/dev"]` grants access to every device node — `/dev/null`, `/dev/console`, `/dev/mem` on Linux — and a tool that opens a file descriptor against an unexpected device can hang the workflow or leak data. Pick the tightest sub-directory you actually need (`/dev/null` if you only want to discard output) instead of the whole tree.

### State redaction

Sensitive state values (API keys, passwords, tokens) can be flagged so they never land in a persisted artefact:

```yaml
state:
  api_key: "..."
  password: "..."
  user_id: 42
state_schema:
  api_key: { redact: true }
  password: { redact: true }
  "*token*": { redact: true }
```

Or, for a deployment-wide baseline, set `AGENTLOOM_REDACT_STATE_KEYS=api_key,password,*token*` — the env-var policy is merged with the YAML one.

Redaction is applied at every persistence boundary:

- **Checkpoint files**: the runtime state snapshot, the literal `state:` block in `workflow_definition`, every `step_results[id].output` value (an LLM call that returns a structured payload), and any step-level config field whose key matches the policy (`notify.headers.api_key`, `tool_args.api_key`, ...).
- **`WorkflowResult.final_state`**: what `agentloom run --json` echoes to stdout and what `result.model_dump_json()` returns to Python callers.
- **Webhook `body_template` rendering**.
- **Opt-in `capture_prompts` span event**: re-rendered against the redacted state so the trace backend sees the sentinel.
- **Subworkflows**: the parent's redaction patterns are merged into the child's `state_schema` at dispatch, so a parent's `redact: true` survives the parent/child boundary. The parent's sandbox config is also forwarded (a child cannot loosen what the parent locked).

The in-memory state stays plaintext so a step that legitimately interpolates `{state.api_key}` against the provider keeps working — only the persistence layer sees the stable `<REDACTED:sha256=...>` sentinel. Redaction is idempotent: a second pass over an already-redacted value preserves it byte-for-byte (so diffing across resume cycles is stable).

`WorkflowDefinition` uses `extra="forbid"` so a typo in `state_schema:` (e.g. `stat_schema:`) fails at parse time instead of silently shipping the secret to disk.

!!! note "Resume contract"
    A redacted checkpoint cannot be resumed with the original secret value. If a workflow pauses on `approval_gate` before consuming the secret, plan to re-inject it on resume (CLI `--state api_key=...`) or do not flag the key as `redact: true`. `WorkflowEngine.from_checkpoint` logs a warning that lists every state key whose loaded value is a sentinel.

### Attachment size limit

All attachments are limited to **20 MB** per file. Larger files are rejected before being sent to the provider.
