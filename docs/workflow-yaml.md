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
(configurable via `--checkpoint-dir` or the `AGENTLOOM_CHECKPOINT_DIR` env var).

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
| `output` | `string` | `null` | State key to store result |
| `timeout` | `float` | `null` | Per-step timeout in seconds |
| `depends_on` | `list[string]` | `[]` | Step IDs that must complete first |

**Retry config:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Number of retry attempts |
| `backoff_base` | `float` | `2.0` | Exponential backoff base (wait = base^attempt) |
| `backoff_max` | `float` | `60.0` | Maximum wait between retries in seconds |

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
    Router expressions are validated via AST. No imports, no `exec`, no attribute assignment. Only a safe subset of Python is allowed.

### `tool`

Executes a registered tool (shell command, HTTP request, etc.).

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
    max_write_bytes: 1000000
```

| Option | Type | Description |
|--------|------|-------------|
| `enabled` | `bool` | Enable sandbox restrictions |
| `allowed_commands` | `list[str]` | Shell command whitelist |
| `allowed_paths` | `list[str]` | General file access paths |
| `readable_paths` | `list[str]` | Read-only paths |
| `writable_paths` | `list[str]` | Write-allowed paths |
| `allow_network` | `bool` | Allow HTTP/network calls |
| `allowed_domains` | `list[str]` | Domain whitelist |
| `max_write_bytes` | `int` | Maximum file write size |

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
