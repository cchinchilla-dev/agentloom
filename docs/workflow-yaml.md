# Workflow YAML Reference

Complete reference for workflow definition files.

## Top-level structure

```yaml
name: my-workflow                           # required
version: "1.0"                              # optional, default "1.0"
description: "What this workflow does"      # optional

config:
  provider: openai                          # default provider
  model: gpt-4o-mini                        # default model
  max_retries: 3                            # retry attempts per step
  budget_usd: 0.50                          # spending limit (null = unlimited)
  timeout: 300.0                            # workflow timeout in seconds (null = unlimited)
  max_concurrent_steps: 10                  # parallel step limit
  stream: false                             # streaming default

state:
  key: "value"                              # initial state variables
  nested:
    key: "value"

steps:                                      # at least one step required
  - id: step_id
    type: llm_call                          # llm_call | tool | router | subworkflow
    # ... step-specific fields
```

## Config options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | `string` | `openai` | Default LLM provider |
| `model` | `string` | `gpt-4o-mini` | Default model for all LLM steps |
| `max_retries` | `int` | `3` | Retry attempts on failure |
| `budget_usd` | `float` | `null` | Maximum spend in USD |
| `timeout` | `float` | `null` | Workflow timeout in seconds |
| `max_concurrent_steps` | `int` | `10` | Max parallel steps per layer |
| `stream` | `bool` | `false` | Enable streaming by default |
| `sandbox` | `object` | disabled | Security sandbox config |
| `responses_file` | `string` | `null` | Mock provider recording path (when `provider: mock`) |
| `latency_model` | `string` | `constant` | Mock latency mode: `constant` / `normal` / `replay` |
| `latency_ms` | `float` | `0` | Mock provider simulated latency per call |
| `capture_prompts` | `bool` | `false` | When true, `llm_call` spans emit an `agentloom.prompt.captured` event with the rendered prompt + system prompt. Off by default — opt-in for debugging or trusted environments only |

## Checkpointing

Persist workflow execution state so failed or paused runs can be resumed
without re-executing completed steps.

### CLI usage

```bash
# Run with checkpointing enabled
agentloom run workflow.yaml --checkpoint

# Custom checkpoint directory
agentloom run workflow.yaml --checkpoint --checkpoint-dir /data/checkpoints

# List all checkpointed runs
agentloom runs
agentloom runs --json

# Resume a previous run
agentloom resume <run_id>
agentloom resume <run_id> --lite --json
```

### How it works

When `--checkpoint` is enabled, the engine:

1. Generates a unique **run ID** (printed at startup).
2. Saves a checkpoint file after the workflow completes (success or failure).
3. The checkpoint contains the full workflow definition, state, and step results.

On `agentloom resume <run_id>`:

1. Loads the checkpoint from disk.
2. Reconstructs the workflow engine with the saved state.
3. Skips already-completed steps and continues from where it left off.

Checkpoint files are stored as JSON in `.agentloom/checkpoints/` by default
(configurable via `--checkpoint-dir`).

## State

State variables are initialized in the `state` block and accessible in templates:

```yaml
state:
  question: "What is Python?"
  items:
    - id: 1
      name: "Item A"
  count: 42
```

**Template syntax:**

| Expression | Result |
|------------|--------|
| `{state.question}` | `"What is Python?"` |
| `{question}` | `"What is Python?"` (flat access) |
| `{state.items[0].name}` | `"Item A"` |
| `{state.count}` | `42` |

Steps with `output: key` update `state[key]` after execution.

Templates render `tool` step args through the same renderer as `prompt` fields, so `{state.url}` substitutions work uniformly across step types. By default a missing key (`{state.does_not_exist}`) is logged and rendered as an empty string; to raise `TemplateError` on missing keys instead, build the namespace with `build_template_vars(state, strict=True)` (so nested `{state.*}` lookups also raise) and render with `SafeFormatDict(template_vars, strict=True)`.

---

## Step types

### `llm_call`

Sends a prompt to an LLM and stores the response.

```yaml
- id: answer
  type: llm_call
  prompt: "Answer: {state.question}"        # required
  system_prompt: "You are helpful."         # optional
  model: gpt-4o                             # optional, overrides config
  temperature: 0.7                          # optional (0-2)
  max_tokens: 1000                          # optional
  stream: true                              # optional, overrides config
  output: answer                            # state key for result
  timeout: 30.0                             # per-step timeout
  depends_on: [previous_step]               # dependencies
  attachments:                              # multi-modal input
    - type: image
      source: "{state.image_url}"
      fetch: local
  retry:
    max_retries: 3
    backoff_base: 2.0
    backoff_max: 60.0
```

**LLM step fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | — | Required. Template string with `{state.*}` interpolation |
| `system_prompt` | `string` | `null` | Optional system message |
| `model` | `string` | `null` | Override workflow-level model |
| `temperature` | `float` | `null` | Sampling temperature (0-2), provider default if null |
| `max_tokens` | `int` | `null` | Output token limit |
| `stream` | `bool` | `null` | Override workflow-level streaming setting |
| `attachments` | `list[Attachment]` | `[]` | Multi-modal inputs (see [Providers](providers.md#multi-modal-attachments)) |
| `thinking` | `ThinkingConfig` | `null` | Extended-thinking / reasoning config (see [Reasoning models](providers.md#reasoning-models)) |
| `output` | `string` | `null` | State key to store result |
| `timeout` | `float` | `null` | Per-step timeout in seconds |
| `depends_on` | `list[string]` | `[]` | Step IDs that must complete first |

**Thinking config:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Activate provider-side reasoning |
| `budget_tokens` | `int` | `null` | Anthropic `budget_tokens` / Gemini `thinkingBudget` cap. OpenAI infers from model tier and ignores this field |
| `level` | `"low" \| "medium" \| "high"` | `null` | Gemini `thinkingLevel` / Ollama `think` value |
| `capture_reasoning` | `bool` | `true` | Expose the chain-of-thought trace via `ProviderResponse.reasoning_content` (Anthropic / Gemini / Ollama). OpenAI o-series keeps the trace server-side regardless |

Per-provider translation:

| Provider | Translation |
|----------|-------------|
| OpenAI o-series | Reasoning is implicit in the model name; the config is accepted for YAML uniformity but not forwarded to the wire |
| Anthropic | `thinking: {type: "enabled", budget_tokens: <budget_tokens>}` |
| Google Gemini 2.5+ | `generationConfig.thinkingConfig: {thinkingBudget, thinkingLevel, includeThoughts}` |
| Ollama 0.9+ | top-level `think: <level>` if `level` is set, else `think: true` |

```yaml
- id: complex_reasoning
  type: llm_call
  model: claude-opus-4
  prompt: "Solve: {state.problem}"
  thinking:
    enabled: true
    budget_tokens: 5000
    level: high
    capture_reasoning: true
  output: answer
```

Reasoning tokens are billed at the output rate. `TokenUsage.reasoning_tokens` and `billable_completion_tokens` track the spend; `calculate_cost()` includes them automatically. See [Reasoning models](providers.md#reasoning-models) for per-provider details, including the Ollama caveat that `eval_count` is not split.

**Tool calling:**

The model can pick tools at runtime. Declare them on the step; the engine dispatches via the workflow's `ToolRegistry`, feeds results back, and re-prompts until the model stops asking for tools.

```yaml
- id: ask
  type: llm_call
  prompt: "What is the user's account balance?"
  tools:
    - name: lookup_account
      description: "Retrieve account info by ID."
      parameters:
        type: object
        properties:
          account_id: { type: string }
        required: [account_id]
  tool_choice: auto              # auto | required | none | {name: lookup_account}
  max_tool_iterations: 5         # bound the loop; default 5
  output: answer
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tools` | `list[ToolDefinition]` | `[]` | Tool declarations the model can pick. `parameters` is JSON Schema. Names resolve against the registered `ToolRegistry`; an unknown name is reported back as a tool failure rather than aborting the loop. |
| `tool_choice` | `string \| dict` | `"auto"` | `"auto"` lets the model decide; `"required"` forces a call; `"none"` disables tools for this turn; `{"name": "..."}` pins to a specific tool. Anthropic has no native `"none"` mode, so when `"none"` is set the adapter drops `tools` from the wire entirely — same observable behavior as the other providers. Ollama ignores `tool_choice` at the wire level (model-side support decides whether a call fires). |
| `max_tool_iterations` | `int` | `5` | Cap on call→result→re-prompt loops. When hit, `finish_reason` becomes `"max_tool_iterations"` so callers can detect runaway behavior. |

The dispatched tool runs through the existing sandbox (#105), so `http_request`, `shell_command`, `file_read`, `file_write` honor the workflow's `sandbox:` config. Multiple tool calls in one response are dispatched concurrently (anyio task group); results preserve order in the conversation. Cost and tokens accumulate across iterations on the surfaced `StepResult`.

The legacy `tool` step (static DAG node, author chooses the tool) keeps working unchanged — `tools=` on `llm_call` is the new dynamic, model-driven path.

**Retry config:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Number of retry attempts |
| `backoff_base` | `float` | `2.0` | Exponential backoff base (wait = base^attempt) |
| `backoff_max` | `float` | `60.0` | Maximum wait between retries in seconds |
| `jitter` | `bool` | `true` | Apply ±25% jitter to each backoff so concurrent retries don't cluster |
| `retryable_status_codes` | `list[int]` | `[429, 500, 502, 503, 504]` | Provider status codes that trigger a retry. Other 4xx (e.g. 400/401/403/404) bail out immediately so the retry budget isn't burned on permanent failures. Status-less exceptions (network errors, generic provider failures) are always treated as transient and retried. |

### `router`

Evaluates conditions against state and activates a target step. Steps not activated are skipped.

```yaml
- id: route
  type: router
  depends_on: [classify]
  conditions:
    - expression: "state.classification == 'question'"
      target: answer_question
    - expression: "state.score > 80"
      target: handle_high
  default: handle_general                   # fallback if no condition matches
```

**Allowed in expressions:** comparisons, boolean operators (`and`, `or`, `not`), builtins (`len`, `str`, `int`, `float`, `bool`, `abs`, `min`, `max`).

!!! warning "Safety"
    Router expressions are validated via AST and run inside a strict sandbox. The validator rejects:

    - imports, `exec`, attribute assignment;
    - any `_`-prefixed name (`__class__`, `_private`) — blocks dunder traversal and access to private attributes;
    - `kwargs` and starred arguments in calls — closes `format_map` / `**vars()` exfiltration;
    - the `type` builtin — was usable as `type(x).__mro__[1].__subclasses__()`.

    Violations raise `SecurityError`. Only a small audited subset of Python is allowed.

### `tool`

Executes a registered tool with author-chosen arguments — the workflow author decides which tool to call, not the model. For model-driven tool selection, use the `tools=` field on an `llm_call` step (see [tool calling](#llm_call) above).

```yaml
- id: fetch
  type: tool
  tool_name: http_request                   # registered tool name
  tool_args:
    url: "state.api_url"                    # "state." prefix resolves from state
    method: "GET"
    headers:
      Authorization: "Bearer token"
  output: response
  depends_on: [previous_step]
```

!!! info "Argument resolution"
    String values starting with `state.` are resolved from workflow state. Other values are passed as literals.

### `subworkflow`

Nests a workflow inside another. The child inherits the parent's state.

=== "External file"

    ```yaml
    - id: nested
      type: subworkflow
      workflow_path: "./child_workflow.yaml"
      output: child_result
      depends_on: [prepare]
    ```

=== "Inline definition"

    ```yaml
    - id: nested
      type: subworkflow
      workflow_inline:
        name: child
        steps:
          - id: inner
            type: llm_call
            prompt: "Process: {state.data}"
            output: processed
      output: child_result
    ```

---

## Streaming

Enable streaming at the workflow level, per-step, or via CLI:

=== "Workflow config"

    ```yaml
    config:
      stream: true
    ```

=== "Per-step"

    ```yaml
    steps:
      - id: answer
        type: llm_call
        stream: true
        prompt: "Answer: {question}"
    ```

=== "CLI flag"

    ```bash
    agentloom run workflow.yaml --stream
    ```

Token usage, cost, and time-to-first-token are tracked during streaming.

**Streaming + tools.** `stream: true` is compatible with `tools: [...]` — the request wire carries the tool spec and the final `ProviderResponse` returned by `StreamResponse.to_provider_response()` exposes any `tool_calls` the model emitted. Per-chunk `ToolCallDelta` / `ToolCallComplete` events are not yet surfaced by every adapter (follow-up work); read `tool_calls` after the stream is exhausted for now.

---

## Sandbox

Restrict tool execution with an allowlist-based sandbox:

```yaml
config:
  sandbox:
    enabled: true
    allowed_commands: [echo, cat, curl]
    allowed_paths: [/tmp/work]
    readable_paths: [/data]
    writable_paths: [/tmp/output]
    allow_network: true
    allowed_domains: [api.example.com]
    allowed_schemes: [https]                # restrict URL schemes (default: http, https)
    max_write_bytes: 1000000
    danger_opt_in: [bash]                   # opt-in per meta-executable (empty by default)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable sandbox restrictions |
| `allowed_commands` | `list[str]` | `[]` | Shell command whitelist |
| `allowed_paths` | `list[str]` | `[]` | General file access paths |
| `readable_paths` | `list[str]` | `[]` | Read-only paths |
| `writable_paths` | `list[str]` | `[]` | Write-allowed paths |
| `allow_network` | `bool` | `true` | Allow HTTP/network calls |
| `allowed_domains` | `list[str]` | `[]` | Domain whitelist |
| `allowed_schemes` | `list[str]` | `["http", "https"]` | URL scheme whitelist (rejects `file://`, `gopher://`, etc.) |
| `max_write_bytes` | `int \| null` | `null` (unlimited) | Maximum file write size |
| `danger_opt_in` | `list[str]` | `[]` | Per-binary opt-in for meta-executables (`bash`, `python`, `env`, `xargs`, ...). Empty by default — meta-executables defeat the command allowlist by re-launching arbitrary binaries. Add only the names you actually need. |

!!! warning "Meta-executables"
    Even when `bash` is in `allowed_commands`, the sandbox **rejects** the call unless `bash` is also listed in `danger_opt_in`. The opt-in is per-binary, not a global flag — `danger_opt_in: ["bash"]` does not also enable `python`. The same gate applies to `sh`, `python`, `python3`, `env`, `xargs`, `eval`, `exec`. Relative path arguments are validated against the configured cwd; `../` escapes are rejected.

---

## Complete example

A classify-and-respond workflow with routing:

```yaml
name: classify-and-respond
config:
  provider: openai
  model: gpt-4o-mini
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
