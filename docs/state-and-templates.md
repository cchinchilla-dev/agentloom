---
title: "State & templates"
description: "How an AgentLoom run's state starts, how steps write to it, the template syntax that reads it, and how to keep secrets in it out of saved files."
order: 8
section: "Workflows"
---

# State & templates

The state is one dictionary that the whole run shares. Steps write to it through `output` and read it through templates; this page covers both, and how to keep the secrets in it out of everything a run saves.

## The initial state

A run starts from the file's `state:` block, which holds any YAML value:

```yaml
state:
  question: "What changed in the last release?"
  items:
    - { id: 1, name: "Item A" }
  limits: { max_words: 120 }
```

`agentloom run` adds or replaces top-level keys with `--state key=value`, once per key:

```bash
agentloom run qa.yaml --state question="Is Python 3.14 supported?" --state items='[1, 2, 3]'
```

- A value that looks like JSON, starting with `[`, `{` or `"`, a number, `true`, `false` or `null`, is parsed as JSON. Anything else, or JSON that fails to parse, stays a string.
- The key is taken as written, dots included: `--state limits.max_words=80` sets a key called `limits.max_words`, not `max_words` inside `limits`.
- Quote values with brackets or spaces, so the shell passes them whole.

## Writing state

A step's `output` names the key its result goes to, written when the step succeeds:

| Step type | What `output` receives |
|-----------|------------------------|
| `llm_call` | The reply's text, or the parsed value with a `response_schema`. |
| `agent` | The turn's reply. |
| `tool` | What the tool returned. |
| `router` | The id of the step it chose. |
| `subworkflow` | The child's final state, or only its `return_keys`. |
| `approval_gate` | `approved` or `rejected`. |
| `embed` | The vectors. |

`output` takes a path as well as a key:

- `report.summary` writes `summary` inside `report`, and creates `report` if it doesn't exist.
- `items[0]` and `items[0].name` write into a list item that already exists. Lists never grow: an index past the end fails the step.
- A path can't go through a value that isn't a dictionary: writing `user.name` when `user` is the string `"alice"` fails the step instead of replacing the string.

Each step reads a copy of the state taken when its layer starts, so steps in the same layer don't see each other's writes; the next layer sees all of them. See [Dependencies and layers](concepts.md#dependencies-and-layers).

### Keys AgentLoom writes

| Key | What it holds |
|-----|---------------|
| `steps.<id>.output`, `steps.<id>.status` | The result and status of each step that succeeded with a result. Templates read `{state.steps.classify.output}`; routers read `steps.classify.output`. |
| `_approval.<id>` | The decision for each approval gate, written when the run resumes. |

Don't use `steps` or `_approval` as keys of your own.

## Templates

Templates put state values into text. They are rendered in:

- the `prompt` and `system_prompt` of an `llm_call`, and the `prompt` of an `agent` step;
- the `source` of an attachment;
- the string values of `tool_args`, with [narrower rules](#templates-in-tool_args);
- the `body_template` of an approval gate's webhook.

Other fields that name state are paths, not templates: a router's `expression`, an embed step's `inputs`, a `conversation`. A subworkflow's `input` values are rendered like a tool step's `tool_args`.

| Template | Renders |
|----------|---------|
| `{state.question}` | The value of `question`. |
| `{state.limits.max_words}` | A key inside a dictionary. |
| `{state.items[0].name}`, `{state.items[-1]}` | A list item, counted from the end with a negative index. |
| `{state.score:.2f}` | The value with a Python format spec: `0.87`. |
| `{state.items}` | A whole list or dictionary, as Python writes it: `[1, 2, 3]`. |
| `{state.steps.classify.output}` | The result of the step `classify`. |
| `{question}` | The same value without `state.`, for top-level keys. |
| `{{` and `}}` | A literal `{` and `}`. |

Literal braces in a prompt must be written twice. `Return JSON like {"label": "x"}` fails the step with an `Invalid format specifier` error; `Return JSON like {{"label": "x"}}` sends the braces.

### Missing keys

A template that names a key the state doesn't have doesn't stop the run:

- `{state.name}` renders as an empty string, however deep the path or whatever the format spec, and logs a warning: `Template variable 'state.name' not found, rendering as empty`.
- `{name}` without `state.` stays in the text as written, braces included.
- An index past the end of a list renders empty, like a missing key.

A typo shows up as a blank, or as a literal `{name}`, in what the model receives. Prefer `{state.…}`, check the log for those warnings, and keep state keys in plain ASCII: lookups are byte for byte, so a key written in one Unicode form doesn't match the same word in another.

### Templates in `tool_args`

`tool_args` often carry JSON, HTML or code, whose braces aren't templates. A string value is rendered only when it contains a placeholder: `{state.` or `{state[`, or `{name}`, `{name!r}` or `{name:spec}` with no space after the colon. Anything else passes through unchanged:

```yaml
tool_args:
  url: "https://api.example.com/users/{state.user_id}"   # rendered
  summary: "Total: {total:.2f}"                          # rendered
  payload: '{"k": [1, 2, 3], "v": true}'                 # unchanged
  style: "<style>.x { color: red; }</style>"             # unchanged
```

- **A whole value.** A string that starts with `state.`, such as `items: state.items`, passes the value itself, a list or a dictionary, not its text. If the key doesn't exist, the step fails without a retry.
- **At any depth.** A string inside a nested mapping or list, such as `headers: {Authorization: "Bearer {state.token}"}`, follows these same rules.
- **Mixed strings.** A string with a placeholder and literal braces, such as `'{"user": "{state.user}"}'`, is rendered as a whole and fails on the literal braces. Write those braces twice: `'{{"user": "{state.user}"}}'`.
- **A literal that looks like a placeholder.** `{value: "...", template: false}`, with no other keys, passes `value` as written:

```yaml
tool_args:
  body:
    value: '{"query": "{state.q}"}'    # sent exactly like this
    template: false
```

## Redaction

`state_schema` marks keys whose values must never reach a file, a record or a trace:

```yaml
state:
  api_key: "..."
  github_token: "..."
  user_id: 42

state_schema:
  api_key: { redact: true }
  "*_token": { redact: true }      # a glob
```

- Patterns are shell-style globs, matched against each key's name and, for nested keys, against its dotted path, such as `credentials.access_token`. A matching list or dictionary has each of its values redacted.
- `AGENTLOOM_REDACT_STATE_KEYS=api_key,*_token,*password*` sets patterns for every workflow run in that environment, on top of each file's `state_schema`.
- A redacted value becomes `<REDACTED:sha256=…>`, the same for the same value, so files can still be compared.

Redaction applies wherever the state leaves the process:

- checkpoints, including the workflow definition and step results saved in them;
- run records in `agentloom_runs/`;
- the result that `agentloom run --json` prints, and that `WorkflowResult` returns in Python;
- an approval gate's webhook `body_template`;
- the prompts that `capture_prompts` adds to traces.

In the workflow definition and the step results, patterns match the workflow's own data: its `state:` block, `tool_args`, `input`, webhook headers and each step's `output`, which is redacted whole when the key it writes matches. AgentLoom's own fields, such as `max_tokens` or `token_usage`, are never matched, so a pattern like `*token*` is safe.

The state in memory stays plain, so a step that needs the secret, such as a tool call with `{state.api_key}`, still gets it. A subworkflow inherits its parent's patterns.

> [!WARNING]
> A run resumed from a checkpoint gets the redacted value, not the original: steps after the pause that read the key receive `<REDACTED:sha256=…>`, and `agentloom resume` logs a warning that names the key. `agentloom resume --state key=value` puts the value back; without it, redact only keys that no step after an approval gate needs.

## What's next

<!-- cards -->

- **[Steps](steps.md)** — What each step type reads and writes.
- **[LLM calls](llm-calls.md)** — Prompts, conversations stored in the state, and structured replies.
- **[Tools & sandbox](tools-and-sandbox.md)** — Built-in and custom tools, and the arguments they take from the state.

<!-- /cards -->
