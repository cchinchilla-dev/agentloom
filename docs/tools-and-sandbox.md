---
title: "Tools & sandbox"
description: "The tool step, the four built-in tools, tools of your own written in Python, and what the sandbox lets the built-in tools run, read, write and reach."
order: 9
section: "Workflows"
---

# Tools & sandbox

Tools let a workflow act: call an HTTP API, run a command, read and write files, or run Python code of your own. A `tool` step runs one with arguments the workflow chooses; an [LLM call](llm-calls.md#tool-calling) or an [agent](agents.md) lets the model choose. The sandbox limits what the built-in tools may do.

## The `tool` step

```yaml
- id: fetch
  type: tool
  tool_name: http_request
  tool_args:
    url: "https://api.example.com/orders/{state.order_id}"
    method: GET
  output: order
```

| Field | What it is |
|-------|------------|
| `tool_name` | A registered tool: one of the [built-in tools](#built-in-tools), or [one of your own](#tools-of-your-own). |
| `tool_args` | The tool's arguments. String values, at any depth, are [templates](state-and-templates.md#templates-in-tool_args), and `state.items` passes a whole value. |
| `output` | The key for what the tool returns. |

The step also takes the [fields every step has](workflow-files.md#fields-every-step-has): `depends_on`, `timeout` and `retry`.

`output` receives the tool's return value as it is: a dictionary stays a dictionary, and templates read its fields, as `{state.order.status_code}`. `steps.<id>.output` holds the same value. A tool that returns nothing writes `None` to `output` and leaves no `steps.<id>` entry.

`agentloom validate` doesn't look tools up: a `tool_name` that isn't registered passes it and fails when the run reaches the step.

### When a tool step fails

| Failure | What happens |
|---------|--------------|
| A `tool_name` that isn't registered | Fails at once. |
| A `state.` argument naming a key the state doesn't have | Fails at once. |
| The tool raises an error: a refused path, a missing argument, a dropped connection | Retried like a transient failure, so the tool runs again on each attempt. |
| No `tool_name`, or broken braces in a `tool_args` template | Retried too, though no attempt can succeed. |
| The step's `timeout` runs out | The attempt ends `timeout` and is retried. |

An error status from `http_request`, such as 404 or 500, and a non-zero exit code from `shell_command` aren't failures: they come back in the result, for a later step or a [router](steps.md#router) to check.

Every error a tool raises is retried, a [sandbox](#the-sandbox) refusal included, though the next attempt is refused the same way. On a step that may be refused, set `retry: {max_retries: 0}`. [Reliability](reliability.md) covers retries and what a failed step does to the steps after it.

### A step or the model

| | `tool` step | A model's tool call |
|---|---|---|
| Who picks the tool and its arguments | The workflow. | The model, from the tools the step declares. |
| The tool's parameters | Not needed. | An LLM call declares them in `tools`; an agent takes the tool's own. |
| A tool error | Fails the step, which is retried. | Goes back to the model as the tool's result; the step goes on. |
| `--record-tools` | Not recorded: the tool runs again on replay. | Recorded, and replayed without running the tool. See [Testing & replay](testing-and-replay.md). |
| Traces and metrics | A tool span and the `agentloom_tool_calls_total` and `agentloom_tool_call_duration_seconds` metrics. See [Observability](observability.md). | The same. |

## Built-in tools

`agentloom run`, and every other command that runs a workflow, registers four tools, which check every call against the workflow's [sandbox](#the-sandbox).

### `http_request`

| Argument | Default | What it is |
|----------|---------|------------|
| `url` | required | The URL, `http` or `https`. |
| `method` | `GET` | `GET`, `POST`, `PUT` or `DELETE`. Any other method fails the call. |
| `headers` | none | Request headers. |
| `body` | none | The request body. A string is sent as it is, so set its `Content-Type` header. A mapping is sent as JSON. |
| `timeout` | `30` | Seconds for each phase of the request: connecting, sending, reading. |

It returns `{status_code, headers, body}`, with the body as text.

- Any status is a result, not an error.
- Redirects aren't followed: a 3xx response comes back with its `location` header.
- Response headers whose names contain `authorization`, `cookie`, `token`, `secret` or `key` are left out of the result.
- The response isn't limited in size.

### `shell_command`

| Argument | Default | What it is |
|----------|---------|------------|
| `command` | required | The command line, run by `sh -c`. |
| `timeout` | `30` | Seconds before the command is stopped. |
| `cwd` | `.` | The directory to run in, relative to the directory you run from. |

It returns `{returncode, stdout, stderr}`.

- A non-zero exit code is a result, not an error.
- Past `timeout`, the command and every process it started are killed, and the step still succeeds, with `returncode: -1` and `stderr: "Command timed out"`.
- The output isn't limited in size.

> [!WARNING]
> With the sandbox disabled, commands inherit the whole environment of the AgentLoom process, provider API keys included. With it enabled, a command gets only `PATH`, `HOME`, `LANG`, `LC_ALL` and `TZ`.

### `file_read` and `file_write`

| Tool | Arguments | Returns |
|------|-----------|---------|
| `file_read` | `path`, `encoding` (`utf-8`) | The whole file, as text. |
| `file_write` | `path`, `content`, `encoding` (`utf-8`) | `{written, path}`, where `written` counts bytes in `encoding`. |

- Relative paths start from the directory you run from.
- `file_write` creates missing parent directories and replaces an existing file. `content` must be a string.
- `file_read` reads text only: a binary file fails, and so does a missing one.

### Declaring them to a model

An [LLM call](llm-calls.md#tool-calling) sends the model the `tools` it declares, with their `description` and `parameters` as written in the step. A tool declared without `parameters` is sent with the registered tool's own. An [agent](agents.md) names its tools and sends their own description and parameters.

```yaml
tools:
  - name: file_read
    description: "Read a text file."
    parameters:
      type: object
      properties:
        path: { type: string }
      required: [path]
```

## Tools of your own

Tools of your own are written in Python and registered on a `ToolRegistry`. `agentloom run` registers only the built-in tools, so a workflow that uses yours runs, and resumes, from Python.

### Defining a tool

`@tool` turns a function into a tool:

```python
from agentloom.tools import tool

@tool(name="word_count", description="Count the words in a text.")
def word_count(text: str) -> dict:
    return {"words": len(text.split())}
```

- The name defaults to the function's, and the description to its docstring.
- The parameters come from the type hints: `str`, `int`, `float` and `bool`, lists such as `list[str]`, dictionaries such as `dict[str, float]`, and `Optional[...]` for a value that may be null. A parameter without a hint is a string, and one without a default is required.
- A function that isn't `async` runs in a worker thread, so it doesn't block other steps.

For more control, subclass `BaseTool` from `agentloom.tools`: set `name`, `description` and `parameters_schema`, a JSON Schema, and implement `async def execute(self, **kwargs)`.

A tool receives its arguments and nothing else: not the state, the run or the sandbox. Its arguments aren't checked against its parameters, so a tool should check them itself. Return values that serialise to JSON, such as strings, numbers, lists and dictionaries: the value goes to the state, to checkpoints and to `--json`, and a model receives anything but a string as JSON.

### Running a workflow with them

```python
import anyio

from agentloom.core.engine import WorkflowEngine
from agentloom.core.parser import WorkflowParser
from agentloom.providers.gateway import ProviderGateway
from agentloom.providers.openai import OpenAIProvider
from agentloom.tools import ToolRegistry
from agentloom.tools.builtins import register_builtins
from agentloom.tools.sandbox import ToolSandbox

async def main() -> None:
    workflow = WorkflowParser.from_yaml("word_tools.yaml")

    registry = ToolRegistry()
    register_builtins(registry, sandbox=ToolSandbox(**workflow.config.sandbox.model_dump()))
    registry.register(word_count)

    gateway = ProviderGateway()
    gateway.register(OpenAIProvider(), priority=0)

    engine = WorkflowEngine(workflow=workflow, provider_gateway=gateway, tool_registry=registry)
    result = await engine.run()
    print(result.status, result.final_state["counted"])

anyio.run(main)
```

- `register_builtins` gives the built-in tools the sandbox you pass. Without `sandbox=`, they run with no limits, whatever the file's `config.sandbox` says.
- Registering a name that is already taken replaces that tool, with a warning; `register(tool, replace=True)` replaces it quietly.
- An engine without a `tool_registry` fails every `tool` step.
- `examples/14_custom_tools_decorator.py` and `examples/15_custom_tools_subclass.py` run complete workflows with tools of their own. [Python DSL](python-dsl.md) covers the engine and the gateway.

> [!WARNING]
> The sandbox applies to the built-in tools only. A tool of your own runs with whatever access its code has. To hold it to the same rules, give it a `ToolSandbox` and call `validate_path(path, writable=...)`, `validate_command(command, cwd=...)` or `validate_network(url)`, which raise `SandboxViolationError` on a refusal.

## The sandbox

`config.sandbox` limits what the built-in tools may run, read, write and reach. It is disabled by default, and the built-in tools then have the access of the process that runs AgentLoom: any command, any file and any host, internal addresses included.

```yaml
config:
  sandbox:
    enabled: true
    allowed_commands: [ls, wc, grep]
    readable_paths: [data]
    writable_paths: [out]
    allowed_domains: [api.example.com]
    max_write_bytes: 1000000
```

| Field | Default | What it allows, with `enabled: true` |
|-------|---------|--------------------------------------|
| `enabled` | `false` | Turns the checks on. |
| `allowed_commands` | none | The programs `shell_command` may run, by name. |
| `allowed_paths` | none | Directories the file tools may read and write. |
| `readable_paths` | none | More directories to read from. |
| `writable_paths` | none | More directories to write to. |
| `allow_network` | `true` | `false` refuses every `http_request`. |
| `allowed_domains` | any | The hosts `http_request` may reach. |
| `allowed_schemes` | `http`, `https` | The URL schemes `http_request` may use. |
| `max_write_bytes` | no limit | The largest `file_write`, in bytes. |
| `danger_opt_in` | none | Interpreters and launchers that `allowed_commands` may include. See [Commands](#commands). |
| `allow_internal_webhook_targets` | `false` | Lets an approval gate's webhook reach internal addresses. See [The webhook](steps.md#the-webhook). |
| `deny_internal_hosts` | `false` | Refuses internal addresses for `http_request`, even when `allowed_domains` lists them. See [Subworkflows](#subworkflows). |

Paths are relative to the directory you run from, resolved when the run starts.

`allow_network: false` without `enabled: true` limits [attachment](llm-calls.md#attachments) downloads and nothing else: `http_request` still reaches any URL.

> [!WARNING]
> A misspelled key under `sandbox`, such as `allowed_command`, stops the load, and `agentloom validate` names it.

### What it covers

| | `enabled: false` | `enabled: true` |
|---|---|---|
| `shell_command` | Any command. | [Commands](#commands). |
| `file_read`, `file_write` | Any path. | [Files](#files). |
| `http_request` | Any URL, internal addresses included. | [Network](#network). |
| Tools of your own | No checks. | No checks. |
| [Attachment](llm-calls.md#attachments) downloads | `allow_network`, `allowed_domains`, no internal addresses. | The same. |
| Attachment files | Any path. | `allowed_paths` and `readable_paths`. |
| [Webhooks](steps.md#the-webhook) | No internal addresses, unless allowed. | Also `allow_network`, `allowed_schemes` and `allowed_domains`. |

### Commands

With the sandbox enabled, a `shell_command` call is checked in this order:

1. **The directory.** `cwd` must sit under `allowed_paths` or `readable_paths`. Otherwise every command is refused: with `cwd` left at `.`, the directory you run from must be covered, or each call must pass a `cwd` that is.
2. **Shell operators.** `|`, `;`, `&`, a backtick, `<`, `>`, `$(`, `<(`, `>(` and line breaks are refused anywhere in the command, even inside quotes. One call runs one program.
3. **Shell expansion.** `$`, `~`, `*`, `?`, `[` and brace lists such as `{a,b}` or `{1..3}` are refused anywhere in the command, even inside quotes, so `'$5'` or `"*.log"` can't be passed to a program. A command that needs them can't run in the sandbox.
4. **The program.** The file name of the first word must be in `allowed_commands`, matching case. A path, such as `/usr/bin/grep` or `./grep`, must be the file that name resolves to on `PATH`; any other file of that name is refused.
5. **Interpreters and launchers.** `sh`, `bash`, `zsh`, `fish`, `ksh`, `dash`, `env`, `xargs`, `find`, `exec`, `eval`, `source`, `.`, `python`, `python3`, `node`, `perl`, `ruby`, `php`, `lua`, `awk`, `nc`, `ncat`, `socat` and `ssh` can run anything else, so they are refused even when allowed, unless `danger_opt_in` lists them too, one by one.
6. **Path arguments.** When any path list is set, each argument that looks like a path, such as `.`, `..`, anything with a `/`, or the value of `--opt=dir/file`, must resolve, from `cwd`, under one of the three lists. A bare name that is a symbolic link is checked too. A path attached to a short flag is checked from its first `/`, so `-finclude/list` reads as `/list` and is refused: pass it as a separate argument, `-f include/list`.

> [!CAUTION]
> The checks see the command as written, not what an allowed program does with its arguments. They guard against mistakes, not against a model that reads untrusted input: don't give `shell_command` to one.
>
> A path argument doesn't say whether the program reads or writes it, so `cp` can write under `readable_paths` and `cat` can read under `writable_paths`. A URL is taken for a relative path, so with `curl` allowed, `curl file:///etc/passwd` passes, and `allow_network: false` doesn't stop it.

### Files

- `file_read` may read under `allowed_paths` and `readable_paths`; `file_write` may write under `allowed_paths` and `writable_paths`. A directory listed only in `writable_paths` can't be read.
- Paths are resolved before the check, `..` and symbolic links included: a link inside an allowed directory that points outside it is refused.
- With the sandbox enabled and no path lists, nothing can be read or written.
- `max_write_bytes` limits the encoded size of what `file_write` writes.

### Network

With the sandbox enabled, an `http_request` call is checked against:

- `allow_network: false`, which refuses every request;
- `allowed_schemes`: the URL's scheme must be listed. Only `http` and `https` ever work;
- `allowed_domains`, when set: the host must be listed exactly, without regard to case. `api.example.com` doesn't admit `eu.api.example.com`. A listed host is reached even when it is internal, which is how to allow a service on your own network, unless `deny_internal_hosts` is on.

Without `allowed_domains`, internal addresses are refused: loopback, private and link-local ranges, the cloud metadata address `169.254.169.254`, shared address space and `localhost`, in any spelling, such as `127.1`, `0x7f000001` or `[::ffff:127.0.0.1]`. A host name is resolved and every address it returns is checked, and a host that doesn't resolve is refused.

The request then connects to the addresses the check resolved, so a DNS answer that changes in between can't reach an internal host. The host name stays in the `Host` header and the TLS server name, and the certificate is checked against it. A proxy from the environment, such as `HTTPS_PROXY`, resolves the name itself: nothing is pinned then, so restrict which hosts the proxy lets through. Webhooks and attachment downloads connect the same way.

### When the sandbox refuses

| Where | What happens |
|-------|--------------|
| A `tool` step | The step fails with `Sandbox violation (shell_command): …`, and is retried like any tool error. |
| A model's tool call | The refusal goes back to the model as the tool's result, and the step goes on. |
| An attachment | The step fails, and is retried. |
| A webhook | Nothing is sent, a warning is logged, and the approval gate still pauses the run. |

### Subworkflows

A [subworkflow](steps.md#subworkflow) runs its built-in tools, attachments and webhooks under one sandbox:

| Parent | Child | The child runs under |
|--------|-------|----------------------|
| Disabled | Disabled | Its own `sandbox` block, which leaves the built-in tools unlimited. |
| Enabled | Disabled | The parent's. |
| Disabled | Enabled | Its own. |
| Enabled | Enabled | What both allow. |

With both enabled, a child can narrow the parent's sandbox and can't widen it:

- A command or a `danger_opt_in` entry must be on both lists.
- A directory must be readable, or writable, under both. Of two nested directories, the deeper one applies.
- `allowed_domains` and `allowed_schemes` must share an entry when both are set; one left empty sets no limit. Two lists with nothing in common refuse every request. A parent with no `allowed_domains` refuses internal addresses, so a child that lists hosts runs with `deny_internal_hosts` on: its list can't open them.
- `allow_network` and `allow_internal_webhook_targets` must be true in both, and the smaller `max_write_bytes` applies.

The child's built-in tools are rebuilt from these `config.sandbox` blocks and from nothing else: from Python, a different `ToolSandbox` passed to `register_builtins` doesn't carry over to a child that has its own sandbox. Tools of your own stay unchecked.

## Running tools in parallel

- `tool` steps in the same [layer](concepts.md#dependencies-and-layers) run at the same time, up to `config.max_concurrent_steps`.
- The tool calls of one model reply run at the same time; see [the loop](llm-calls.md#the-loop).
- Tools read a copy of the state and change it only through `output`.

## What's next

<!-- cards -->

- **[Reliability](reliability.md)** — Retries, timeouts, failure policies, budgets and checkpoints.
- **[LLM calls](llm-calls.md#tool-calling)** — Letting a model call tools, and how the loop ends.
- **[Testing & replay](testing-and-replay.md)** — Recording model tool calls and replaying them offline.

<!-- /cards -->
