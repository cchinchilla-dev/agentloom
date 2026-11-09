---
title: "Agents"
description: "Define an agent once, with its prompt, tools, memory, loop limits, typed reply and constraints, and run it as a step, as another agent's tool or from Python."
order: 7
section: "Workflows"
---

# Agents

An agent is a model with its instructions, tools and rules, defined once in its own file. An `agent` step runs one turn of it; the same agent can serve several steps and workflows, be offered to another agent as a tool, or run from Python without a workflow.

## Defining an agent

```yaml
# examples/agents/support.yaml
name: support_desk
description: Answers billing questions for ACME customers.
system_prompt: |
  You answer billing questions for ACME customers. Before answering, read the
  billing policy at examples/data/billing_policy.txt with file_read. When the
  customer asks for money back, ask the refunds agent whether the charge is
  eligible. Keep the answer to one or two sentences.
tools:
  - file_read                        # a registered tool
  - name: ask_refunds                # another agent, offered as a tool
    agent: examples/agents/refunds.yaml
memory:
  token_budget: 4000
  trim_policy: drop_pairs
loop_policy:
  max_iterations: 6
  on_max_iterations: error
output_schema:
  type: pydantic
  model: examples.structured_output_schemas.SupportReply
constraints:
  max_cost_usd: 0.05
  max_latency_ms: 60000
  forbidden_patterns: ["as an AI language model"]
```

| Field | Default | What it is |
|-------|---------|------------|
| `name` | required | The agent's name: letters, digits, `_` and `-`, up to 64 characters, starting with a letter or `_`. It names the agent's turns in a [conversation](llm-calls.md#several-speakers). |
| `description` | empty | What the agent does. Another agent sees it when this one is offered as a tool. |
| `model` | `config.model` | The model of every turn. |
| `system_prompt` | none | The agent's instructions, a [template](state-and-templates.md#templates) rendered against the workflow's state. |
| `temperature`, `max_tokens` | the provider's | As on an [LLM call](llm-calls.md#a-call). |
| `tools` | none | Registered tool names, and `{name, agent, description}` entries that offer [another agent as a tool](#agents-as-tools). |
| `memory` | none | `token_budget` and `trim_policy` for the conversation, on this agent's turns. See [Memory](#memory). |
| `loop_policy.max_iterations` | `5` | The most model calls of one turn. |
| `loop_policy.on_max_iterations` | `error` | At the cap, fail the step (`error`) or return the last reply with a warning (`warn_and_return`). |
| `output_schema` | none | A JSON shape for the reply, as an LLM call's [`response_schema`](llm-calls.md#structured-output). |
| `constraints.max_cost_usd` | none | The most a turn may spend, the calls of agents it uses as tools included. |
| `constraints.max_latency_ms` | none | The most time a turn may take. |
| `constraints.forbidden_patterns` | none | Text the reply must not contain, matched without regard to case. |

The file is checked when the workflow loads: unknown keys, duplicate tool names and agent files that refer to each other in a cycle are errors. An agent takes no `thinking`, `tool_choice` or `stream`.

## The `agent` step

```yaml
steps:
  - id: answer
    type: agent
    agent: examples/agents/support.yaml
    prompt: "{state.question}"
    conversation: state.chat
    output: reply

  - id: follow_up
    type: agent
    depends_on: [answer]
    agent: examples/agents/support.yaml
    prompt: "How long until the money is back on my card?"
    conversation: state.chat
    output: follow_up
```

| Field | What it is |
|-------|------------|
| `agent` | A path to the agent's file, or its definition inline. |
| `prompt` | The user turn, a template. |
| `conversation` | A [conversation](llm-calls.md#conversations) in the state, which the turn continues. |
| `attachments` | Files sent with the prompt, as on an [LLM call](llm-calls.md#attachments). |
| `output` | The key for the reply: the parsed value with an `output_schema`, the text without one. |

The step also takes `depends_on`, `retry` and `timeout`. The fields the agent defines can't be set on the step: `model`, `system_prompt`, `temperature`, `max_tokens`, `thinking`, `tools`, `tool_choice`, `max_tool_iterations`, `response_schema` and `speaker` are refused with an error that names them.

Paths are relative to the directory you run from, not to the workflow file, and so are the paths of agents named inside an agent file. `examples/41_agent.yaml` loads `examples/agents/support.yaml`, which loads `examples/agents/refunds.yaml`, and runs from the repository's root.

## A turn

A turn is an [LLM call](llm-calls.md) with the agent's settings: its model, system prompt, tools and `max_iterations` as the tool loop's cap, with the agent's `name` as the [speaker](llm-calls.md#several-speakers).

1. The agent's tools are looked up. A tool that isn't registered, the agent's own or one of an agent it calls, fails the step before the first model call.
2. The model is called with the prompt, the conversation and only the agent's tools. A call to any other tool, even one the workflow registers, goes back to the model as not registered.
3. Tool calls run as in an [LLM call's loop](llm-calls.md#the-loop), until the model replies without calling a tool.
4. The reply is checked against `output_schema` and `constraints`. If it passes, it is written to `output` and to the conversation.

The agent's system prompt applies to its own turns and isn't stored in the conversation, so several agents can share one. Nothing is written when the turn fails.

### The cap

Reaching `max_iterations` without a reply:

- under `on_max_iterations: error`, fails the step without a retry;
- under `warn_and_return`, returns the last reply's text with a warning. With tools, that is the text that came with the calls still pending, such as "I'll check the policy", stored as it is. A last reply with no text fails the step either way.

### A typed reply

With an `output_schema`, the reply is validated before it is accepted.

- **Without tools,** the provider's structured output holds the reply to the schema, as on an LLM call. An invalid reply is asked for again, within the same `max_iterations`; if none validates, the step fails without a retry.
- **With tools,** the agent replies by calling a `final_result` tool whose arguments follow the schema, and the model must call a tool on every call of the turn. Invalid arguments go back to the model as the tool's error, and a plain-text reply is sent back asking for the tool. The loop ends on the first valid `final_result`. No valid one within `max_iterations` fails the step, under either `on_max_iterations`. The conversation keeps the validated reply as JSON.

`final_result` is a reserved tool name in an agent with an `output_schema`, and the schema must describe an object.

### Constraints

| Constraint | Checked | When it's broken |
|------------|---------|------------------|
| `max_cost_usd` | Before and after each model call, and when the turn ends. | The step fails without a retry, and reports what the turn spent. |
| `forbidden_patterns` | On the reply's text, or on every text value of a typed reply. | The step fails without a retry. |
| `max_latency_ms` | Over the whole turn, as the step's `timeout`; the shorter of the two applies. | The step fails as a transient failure, so its `retry` repeats the turn. |

Models without a price in AgentLoom's pricing table cost 0, so `max_cost_usd` never stops them. See [Providers](providers.md).

### Memory

`memory` sets the conversation's budget and trimming policy for this agent's turns, without changing the settings the conversation stores; a field left out keeps the conversation's own. The conversation is stored as trimmed, so agents that share it share the trim, and the smallest `memory` wins. To give them all one budget, set it on the conversation instead.

### Retries and results

- The step's `retry` repeats the whole turn after a transient failure: a provider error such as a 429, or `max_latency_ms`. Tools run again. A broken constraint, the cap, a missing `final_result`, a missing tool and a broken template fail the step for good.
- An agent turn doesn't stream: with `stream: true`, the reply reaches the stream in one piece, once it has passed the checks.
- The step's tokens and cost include the calls of agents used as tools and conversation summaries.

## Agents as tools

A `{name, agent}` entry in `tools` offers another agent to the model as a tool, with one `message` parameter:

```yaml
tools:
  - name: ask_refunds
    agent: examples/agents/refunds.yaml
    description: "Ask whether a charge can be refunded."   # the agent's own description without it
```

Each call runs one turn of that agent:

- on a new, empty conversation, with the call's `message` as the prompt;
- with its prompts rendered against the calling workflow's state, plus `message`;
- with its own `model`, or else the workflow's `config.model`, not the calling agent's model.

Its reply goes back to the caller as the tool's result, as JSON with an `output_schema`.

| The called agent's turn… | What the caller sees |
|--------------------------|----------------------|
| fails for good: a broken constraint, the cap under `error` | The tool's error, which goes back to the caller's model. |
| fails on a transient error, such as a provider's 503 or `max_latency_ms` | The calling step fails as transient, and its `retry` repeats the whole turn. |
| can't start: a missing tool, a broken template, a schema that doesn't load | The calling step fails at once. |

- Its cost and tokens count toward the calling step's, and toward the caller's `max_cost_usd`.
- Its model calls are recorded under `<step id>::<tool name>#<n>`, such as `answer::ask_refunds#1` for the first call of the turn, so a [recording](testing-and-replay.md) replays them in order.
- There is no depth limit: agents can call agents that call agents. Only cycles are refused.

## From Python

`Agent` runs a turn without a workflow:

```python
from agentloom import Agent
from agentloom.exceptions import AgentError

agent = Agent.from_yaml(
    "examples/agents/support.yaml",
    gateway=gateway,
    tool_registry=registry,
    model="claude-haiku-4-5",   # for a file that names no model
)

try:
    result = await agent.invoke("My bill seems wrong", conversation=previous)
except AgentError as err:
    print(err)

result.reply          # a SupportReply, typed per output_schema; the text without one
result.conversation   # the conversation, with the new turns
result.cost_usd       # what the turn spent, agents used as tools included
result.iterations     # the agent's own model calls
```

- `Agent(definition, *, gateway, tool_registry=None, model=None)` takes a definition already loaded; `Agent.from_yaml` loads one from a file. The definition's `model` wins over `model=`, and an agent with neither is refused.
- `invoke` raises `AgentError` when the prompt is empty, when the turn can't start, or when it fails. The `conversation` passed in is copied, not changed.
- There is no engine around the turn: no retries, budget or observer.

## Inspecting an agent

`agentloom agent inspect` prints a definition as it loads, without running it:

```bash
agentloom agent inspect examples/agents/support.yaml
agentloom agent inspect examples/agents/support.yaml --json
```

It shows the model and its prices, the system prompt, each tool with its parameters, the agents it calls, described in turn, and the resolved JSON Schema of the `output_schema`. See [CLI](cli.md#agent-inspect).

[`examples/41_agent.yaml`](../examples/41_agent.yaml) runs the support desk above offline, two turns on one conversation.

## What's next

<!-- cards -->

- **[LLM calls](llm-calls.md)** — Tool loops, structured output and conversations, which agent turns are built on.
- **[Tools & sandbox](tools-and-sandbox.md)** — The built-in tools, tools of your own, and what the sandbox allows.
- **[Testing & replay](testing-and-replay.md)** — Recordings, replay and the inter-agent event log.

<!-- /cards -->
