---
name: debug-workflow
description: Debug a workflow YAML statically (validate, trace the DAG and state flow, flag risks) before running it. Use when a workflow fails, behaves unexpectedly, or is new and untested.
---

Debug the workflow at the path given as the argument (or ask for one).

## Phase 1: Static validation
1. `uv run agentloom validate <workflow.yaml>`
2. `uv run agentloom visualize <workflow.yaml>`

Unknown step keys fail at `validate` (`extra="forbid"`), so a typo shows up here with the key named.

## Phase 2: Deep analysis
Read the YAML and trace execution mentally:

- **DAG**: What's the execution order? Which steps run in parallel?
- **State flow**: Track every `output:` field. Does each step's `prompt:` reference variables that will exist at that point? Walk through the state dict step by step.
- **Router logic**: Are conditions mutually exclusive? Is there a default? Could the expression fail on unexpected values (e.g., `state.x` is a dict, not a string)?
- **Tool calls**: Is `tool_name` a registered tool? Are `tool_args` valid for that tool's schema? Read `tools/builtins.py` to check.
- **Budget**: If `budget_usd` is set, estimate cost per step (model + estimated tokens). Will it exceed?
- **Failure propagation**: `config.on_step_failure` defaults to `skip_downstream`, so a failed step or router skips every transitive dependent. Is that intended, or does a best-effort fan-out need `continue`?
- **LLM tool calling**: every entry in `tools:` must resolve in the workflow's tool registry; `max_tool_iterations` (default 5) bounds the loop. Does `tool_choice` force a tool the prompt can't use?
- **Structured output**: `response_schema` validation failures retry within `retry.max_retries`; is that budget enough for the schema's complexity?
- **Approval gates**: the run needs `agentloom run --checkpoint`; `timeout_seconds` only applies under `agentloom callback-server`; `notify.url` must pass the sandbox.
- **Embed**: `inputs` must resolve to a `list[str]` (or `str`); Anthropic has no embeddings API, so the step needs another provider in the chain.
- **Conversation**: `conversation:` must point to a state key holding `messages`; steps sharing it should be ordered by `depends_on` so turns land in sequence. Check `token_budget` / `trim_policy`.
- **Subworkflow**: with `isolated_state: true`, only `input:` flows down and only `return_keys` come back.
- **Secrets**: credentials in `state:` should be listed in `state_schema` with `redact: true`.

## Phase 3: Simulate
For each step in execution order, write out:
```
Step: <id> (<type>)
  Input state: {key: expected_value, ...}
  Expected output: <what this step produces>
  State after: {key: new_value, ...}
  Risk: <what could go wrong>
```

## Phase 4: Verdict
- READY TO RUN / NEEDS FIXES / WILL FAIL
- If fixes needed, suggest specific YAML changes
