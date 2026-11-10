---
title: "Steps"
description: "The seven AgentLoom step types at a glance, and the full reference for routers, subworkflows, approval gates and embedding steps."
order: 5
section: "Workflows"
---

# Steps

Seven step types cover model calls, agents, tools, branching, nested workflows, human approval and embeddings. This page lists them all and covers four in full; model calls, agents and tools have pages of their own.

## Step types at a glance

| Type | What it does | Its main fields | Reference |
|------|--------------|-----------------|-----------|
| `llm_call` | One model call, with optional attachments, tools, structured output or a conversation. | `prompt`, `system_prompt`, `model` | [LLM calls](llm-calls.md) |
| `agent` | One turn of an agent defined once and reused: its prompt, tools and loop policy. | `agent`, `prompt` | [Agents](agents.md) |
| `tool` | Runs a tool, built in or your own, with arguments from the state. | `tool_name`, `tool_args` | [Tools & sandbox](tools-and-sandbox.md) |
| `router` | Chooses which branch runs next. | `conditions`, `default` | [below](#router) |
| `subworkflow` | Runs another workflow as a single step. | `workflow_path` or `workflow_inline` | [below](#subworkflow) |
| `approval_gate` | Pauses the run until someone approves or rejects. | `notify` | [below](#approval_gate) |
| `embed` | Turns text from the state into vectors. | `inputs`, `model` | [below](#embed) |

Every type also takes the [fields all steps share](workflow-files.md#fields-every-step-has): `id`, `type`, `depends_on`, `output`, `timeout` and `retry`.

## `router`

A router reads the state and picks the step that runs next. It checks its conditions in order: the first one that is true names the target, and `default` covers the case where none is.

```yaml
- id: route
  type: router
  depends_on: [classify]
  conditions:
    - expression: "state.category == 'billing'"
      target: billing_reply
    - expression: "state.priority is not None and state.priority >= 3"
      target: escalate
  default: general_reply
  output: route_taken          # optional: the id of the chosen step
```

| Field | What it is |
|-------|------------|
| `conditions` | A list of `expression` and `target` pairs, checked in order. |
| `default` | The target when no condition is true. Without one, and with no match, the router fails. |
| `output` | Receives the chosen step's id, which is also the router's `steps.<id>.output`. |

The parser checks that every `target` and the `default` name a step of the file.

### Branches

A router gates only the steps that list it in `depends_on`:

- The step it chose runs.
- Every other step that depends on the router is skipped, and so is everything downstream of it.
- A target that doesn't depend on the router isn't gated at all: it runs whatever the router chose. The parser doesn't catch this, so give every target `depends_on: [<router id>]`.

A skipped branch doesn't fail the run. A step that depends on two branches never runs, though, because one of them was always skipped: there is no join after a router. Put a follow-up step in each branch instead.

### Expressions

An expression is a small subset of Python, evaluated against two names:

- `state`, the run's state: `state.category`, `state.user.name`, `state['user']['name']` or `state.items[0]`. A key that doesn't exist reads as `None`.
- `steps`, the result of each step that succeeded: `steps.classify.output` and `steps.classify.status`.

There are no bare names: `category == 'billing'` fails, `state.category == 'billing'` works.

| Allowed | Examples |
|---------|----------|
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `is`, `is not` |
| Logic | `and`, `or`, `not`, and `a if condition else b` |
| Arithmetic | `+` and `-`, nothing else |
| Literals | strings, numbers, `True`, `False`, `None`, lists, tuples and dictionaries |
| Functions | `len`, `str`, `int`, `float`, `bool`, `abs`, `min`, `max` and `isinstance` |
| Methods of strings and numbers | `state.category.strip().lower() == 'billing'`, `state.subject.startswith('RE:')` |
| Indexes | literal keys and integer slices: `state['user']`, `state.items[-1]`, `len(state.items[1:3])` |

Lists and dictionaries from the state, or from `steps.<id>.output`, work with `len`, `in`, indexes and `==`, and with their read-only methods: `get`, `keys`, `values` and `items` on a dictionary, `count` and `index` on a list. A key wins over a method of the same name: with a key called `items`, `state.order.items` is its value. Without one it is `None`, like any missing key: only a call, `state.order.items()`, reaches the method.

A comparison with a missing key raises an error rather than evaluating false: `state.priority > 3` fails when `priority` isn't set. Test for it first, as in `state.priority is not None and state.priority >= 3`.

Anything else, from imports and assignments to other functions and attributes that start with `_`, is refused with a `SecurityError` that names the construct.

### When a router fails

A router fails when an expression raises, or when nothing matches and there is no `default`. That failure isn't retried: evaluating the same state again gives the same result. Expressions are only evaluated when the router runs, so `agentloom validate` doesn't catch their mistakes. A router that failed chose no branch, so its branches and everything after them are skipped, under `on_step_failure: continue` too.

## `subworkflow`

Runs another workflow as one step. Its steps run in their own layers, and the step succeeds when the child workflow does.

<!-- tabs -->

**From a file**

```yaml
- id: localize_es
  type: subworkflow
  depends_on: [draft]
  workflow_path: workflows/localize.yaml    # relative to the directory you run from
  output: es
```

**Inline**

```yaml
- id: score
  type: subworkflow
  depends_on: [draft]
  workflow_inline:
    name: scorer
    steps:
      - id: rate
        type: llm_call
        prompt: "Rate this draft from 1 to 10: {state.draft}"
        output: rating
  output: score_result
```

<!-- /tabs -->

With both `workflow_path` and `workflow_inline`, the file wins.

### State in and out

| Mode | The child starts with | `output` receives |
|------|-----------------------|-------------------|
| Default | Its own `state:` block as defaults, with a copy of the parent's state over it. `input` is ignored. | The child's final state, or only the keys in `return_keys`. |
| `isolated_state: true` | Its own `state:` block, plus the keys in `input`, rendered with the parent's state. Nothing else from the parent. | The child's final state, or only the keys in `return_keys`. |

```yaml
- id: classify
  type: subworkflow
  isolated_state: true
  input:
    threshold: 0.75
    topic: "{state.topic}"      # rendered with the parent's state
  return_keys: [label, score]
  workflow_path: workflows/classifier.yaml
  output: classification        # {label: ..., score: ...}
```

- The child never writes to the parent's state. What comes back lands under the step's `output` key, and only when the child succeeds.
- In the default mode, the child's final state includes its copy of the parent's keys: `return_keys` keeps only what you need.
- A key of `return_keys` that the child never set is left out, without an error.
- `input` values render like a `tool` step's `tool_args`, with the parent's state: `"{state.topic}"` is a template, `"state.items"` hands over the value with its type and fails the step if the key is missing, lists and dictionaries render item by item, and `{value: "{literal}", template: false}` keeps a string as written.

### What the child inherits

- **Its own `config`.** A child without one runs with the defaults, OpenAI and `gpt-4o-mini`, not with the parent's model. `--provider` and `--model` on the command line change only the parent.
- **The parent's budget,** unless the child sets its own `budget_usd`. A child that goes over its own budget fails the step, which isn't retried.
- **The stricter sandbox.** A child runs under what its sandbox and the parent's both allow, or under the only one enabled, for its built-in tools too: see [Subworkflows](tools-and-sandbox.md#subworkflows).
- **The parent's redaction rules,** added to the child's `state_schema`. A key of `input` filled from a redacted key is redacted too, whatever its name: the child reads the value, its checkpoints don't hold it, and the parent gets the sentinel back in `output`.
- **The same providers and tools,** with the same keys, rate limits and circuit breakers.

### Failures and retries

A failed child fails the step, with the child's error. The step's `retry` then runs the whole child again, unless the child's failure was permanent, such as a rejected key. The child's steps have their own retries inside each of those attempts.

### Approval gates inside

An [approval gate](#approval_gate) in the child pauses the parent too, and the checkpoint records the pause as `<step id>.<gate id>`, such as `localize_es.review`. `agentloom resume <run_id> --approve` continues the parent and the child together. The child resumes from its own checkpoint, saved as `<run_id>.<step id>`, so its finished steps don't run again.

### Step ids and validation

The child's step ids are its own: a parent and its child can both have a step called `classify`. The parent's load only checks that a `workflow_path` exists; the child is parsed when the step runs. Run `agentloom validate` on a child file before you reference it.

## `approval_gate`

Pauses the run until someone approves or rejects, then writes the decision to `output`.

```yaml
- id: approve_send
  type: approval_gate
  depends_on: [draft_email]
  output: decision                # "approved" or "rejected"
  notify:                         # optional
    url: https://hooks.example.com/agentloom
    headers: { X-Source: agentloom }
```

1. **The gate runs.** It sends the `notify` webhook if there is one, and asks for a pause. The steps already running in its layer finish, and the run ends `paused`.
2. **The run waits.** `agentloom run` prints what to do next:

    ```text
    [APPROVAL REQUIRED] Step 'approve_send' is waiting for a decision.
      Approve: agentloom resume 3f2a9c1b7d4e --approve
      Reject:  agentloom resume 3f2a9c1b7d4e --reject
    ```

3. **Someone decides,** with `agentloom resume` or over HTTP through the [callback server](cli.md#callback-server). The gate succeeds, writes `approved` or `rejected` to `output`, and the run goes on from the next layer.

Run it with `--checkpoint`: without a checkpoint, a paused run leaves nothing to resume.

A rejection doesn't stop the run. The gate succeeds either way, and the steps after it run, so route on the decision to act on it:

```yaml
- id: after_review
  type: router
  depends_on: [approve_send]
  conditions:
    - expression: "state.decision == 'approved'"
      target: send_email
  default: archive_draft
```

| Field | Default | What it is |
|-------|---------|------------|
| `notify.url` | required with `notify` | Where the webhook goes, as a POST. |
| `notify.headers` | none | Headers sent with it, as given. |
| `notify.body_template` | none | The JSON body, as a template. Without it, a default body is sent. |
| `notify.callback_base_url` | none | Where a [callback server](cli.md#callback-server) answers. The default body then links to approve and reject the run. |
| `notify.timeout` | `30` | Seconds for each delivery attempt, within the overall limit below. |
| `timeout_seconds` | none | Seconds the gate waits for a decision, from its first pause. See [When nobody decides](#when-nobody-decides). |
| `on_timeout` | none | `approve` or `reject`: the decision the gate takes once its time runs out. Needs `timeout_seconds`. |

### When nobody decides

With `timeout_seconds`, the gate's first pause sets when it expires, and the checkpoint keeps that time. The first resume after it settles the gate:

- **With `on_timeout`,** the gate takes that decision, whatever `--approve` or `--reject` says, and writes it to `output`.
- **Without `on_timeout`,** the gate ends as `timeout`, without retries, and the run fails.
- **Before the expiry,** a decision applies as usual, and a resume without one pauses again with the same expiry.

[`agentloom callback-server`](cli.md#callback-server) makes that resume itself, without a request, within 30 seconds of the expiry. Without the server, the gate waits for the next `agentloom resume`. `agentloom runs` and the server's `GET /pending` show when each paused gate expires. A gate inside a [subworkflow](#approval-gates-inside) expires the same way.

A checkpoint written by an earlier version has no expiry: its gate starts counting at the next resume.

### The webhook

The default body names the run and the gate:

```json
{"run_id": "3f2a9c1b7d4e", "step_id": "approve_send", "workflow_name": "outreach", "status": "awaiting_approval"}
```

With `notify.callback_base_url`, the default body also holds `approve_url` and `reject_url`: `<base>/approve/<run_id>` and `<base>/reject/<run_id>`, which take a POST. A gate inside a subworkflow links to the run at the top, with the gate's path, `<base>/approve/<run_id>?gate=<step id>.<gate id>`, so one POST resumes the parent and the child together. A `body_template` replaces them with whatever it says.

A `body_template` is rendered like a prompt, with the state, after [redaction](state-and-templates.md#redaction), plus `run_id`, `step_id` and `workflow_name`. Literal braces in the JSON are written twice:

```yaml
notify:
  url: https://hooks.example.com/agentloom
  body_template: '{{"text": "Run {run_id} is waiting at {step_id}: {state.subject}"}}'
```

Delivery is best effort. AgentLoom makes up to three attempts, 2 and then 4 seconds apart, within 5 seconds in all, and a response outside 2xx counts as a failure. A webhook that fails never fails the gate: the run pauses all the same, and the failure is logged.

The URL passes through the workflow's [sandbox](tools-and-sandbox.md). With the sandbox enabled, it must satisfy its network rules. With the sandbox disabled, loopback, link-local and private addresses are still refused, unless `sandbox.allow_internal_webhook_targets` is `true`: a webhook to `localhost` or to an [`agentloom callback-server`](cli.md#callback-server) on your network needs that setting. A host that doesn't resolve is refused, and the delivery connects to the addresses the check resolved. A refused webhook isn't sent, and the gate pauses all the same.

## `embed`

Turns text from the state into vectors.

```yaml
- id: vectorize
  type: embed
  inputs: state.documents          # a string or a list of strings
  model: text-embedding-3-small
  dimensions: 256                  # optional
  output: vectors                  # one vector per input
```

| Field | Default | What it is |
|-------|---------|------------|
| `inputs` | required | A state path, `state.documents` or `{state.documents}`. Dotted paths work, indexes don't. It must hold a string or a list of strings. |
| `model` | `config.model` | An embedding model. The default is the workflow's model, usually a chat model, so set it. |
| `dimensions` | the model's | Shorter vectors, where the provider supports it. |
| `output` | none | Receives the vectors, a list of lists of numbers. The step's `steps.<id>.output` holds a summary instead, such as `3 vectors × 256`. |

| Provider | Models | `dimensions` |
|----------|--------|--------------|
| OpenAI | `text-embedding-3-small`, `text-embedding-3-large` | Yes |
| Google | `gemini-embedding-001` | Yes |
| Ollama | Any embedding model you pulled, such as `nomic-embed-text` | Ignored |
| Anthropic | None: the gateway skips it and tries the next provider | — |

The model name finds the provider, as for a model call. The `mock` provider returns recorded vectors, or made-up ones that are the same on every run; see [Testing & Replay](testing-and-replay.md).

## What's next

<!-- cards -->

- **[LLM calls](llm-calls.md)** — Prompts, attachments, reasoning, tool calling, structured output and conversations.
- **[State & templates](state-and-templates.md)** — What steps write to the state, and the template syntax that reads it.
- **[Examples](examples.md)** — Workflows with routers, subworkflows and approval gates you can run.

<!-- /cards -->
