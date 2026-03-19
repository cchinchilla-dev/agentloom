---
name: debug-workflow
description: Debug a workflow YAML — validate, trace execution, simulate state flow, and catch issues before runtime.
---

Debug the workflow at the path given in $ARGUMENTS (or ask for one).

## Phase 1: Static validation
1. `uv run agentloom validate $ARGUMENTS`
2. `uv run agentloom visualize $ARGUMENTS`

## Phase 2: Deep analysis
Read the YAML and trace execution mentally:

- **DAG**: What's the execution order? Which steps run in parallel?
- **State flow**: Track every `output:` field. Does each step's `prompt:` reference variables that will exist at that point? Walk through the state dict step by step.
- **Router logic**: Are conditions mutually exclusive? Is there a default? Could the expression fail on unexpected values (e.g., `state.x` is a dict, not a string)?
- **Tool calls**: Is `tool_name` a registered tool? Are `tool_args` valid for that tool's schema? Read `tools/builtins.py` to check.
- **Budget**: If `budget_usd` is set, estimate cost per step (model + estimated tokens). Will it exceed?

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
