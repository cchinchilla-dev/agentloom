# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-11-11

### Added

- Agent primitive and `agent` step type (#120). An agent is defined once, in its own YAML file or inline: model, system prompt, tools, memory, loop policy, `output_schema` and constraints (`max_cost_usd`, `max_latency_ms`, `forbidden_patterns`). An `agent` step runs one turn of it, and an agent can offer another as a tool. `Agent` runs a turn from Python, `agentloom agent inspect` prints a definition, and turns get their own span attributes and metrics. `examples/41_agent.yaml` runs offline.
- Inter-agent event log and `agentloom verify-determinism` (#135). `--record` keeps the run's events in the recording (tasks, tool calls and returns, replies, state transitions), in causal order; `verify-determinism` replays it strictly `--runs` times and exits 1 with a diff when the status, outputs, final state or event log differ.
- Deterministic faults in mock recordings (#123). An entry can declare an `http_error`, `timeout`, `connection_reset` or `stream_truncate`, which `MockProvider` raises as the real adapters do, so retries, the circuit breaker and fallback take the real path. Faults are counted in `agentloom_mock_faults_total`; `examples/40_fault_replay.yaml` recovers from a 429 and a 503.
- Replay up to a cut point and run live after it (#165). `agentloom replay --live-from <step>` and `--force-live <step>` serve the steps before the cut from the recording and run the rest with the workflow's providers, reporting the source of every call.
- Record the live part of a cut replay (#178). `--record <file>`, with `--live-from` or `--force-live`, saves a new recording that replays strictly on its own; `--record-tools` adds the tool results.
- Response metadata in recordings (#166). `ProviderResponse.meta` (`ResponseMeta`) holds the sampling parameters sent, the model fingerprint or version, the request id, the rate-limit headers and when the response arrived. Recordings keep it outside the request hash, so old and new recordings replay alike.
- Tool results in recordings (#164). `--record-tools` stores each tool call of a `tools:` loop, and `agentloom replay --tools {live,replay,verify}` runs the tools, serves the recorded results, or runs them and compares.
- Generic OpenAI-compatible endpoint provider (#48). `provider: generic_openai` with `config.base_url` and an optional `config.api_key_env` talks to vLLM, Text Generation Inference, LM Studio or any OpenAI-compatible server; no other provider is registered for that run.
- Public contracts package (#159). `agentloom.contracts.stable` is covered by semver (`Message`, `ToolCall`, `TokenUsage`, `ProviderResponse`, the `WorkflowHandle` protocol and more) and `agentloom.contracts.experimental` holds the OTel names; importing them loads no runtime.
- Conversation history with token-budget trimming (#119). `llm_call` takes `conversation: state.<key>`, appends each turn and trims to `token_budget` with `drop_oldest`, `drop_pairs` or `summarize_oldest`; `speaker:` lets several agents share one conversation. Examples 38 and 39.
- Embeddings across providers (#118). A `type: embed` step and `BaseProvider.embed()` on OpenAI, Google and Ollama, with `dimensions`, fallback and cost as for a model call; Anthropic is skipped. `examples/37_embeddings.yaml`.
- Structured output across providers (#117). `response_schema` on `llm_call` (`pydantic`, `json_schema` or `json_object`) uses each provider's native mode, and instructions plus validation on Anthropic; an invalid reply is asked for again within the step's retries, and the parsed value goes to the state. `jsonschema` becomes a core dependency. `examples/36_structured_output.yaml`.
- Native tool calling across providers (#116). `tools`, `tool_choice` and `max_tool_iterations` on `llm_call` run the model's tool calls through the tool registry, in parallel, until it replies; the sandbox, budget and retries apply. `examples/35_tool_calling.yaml`.
- Run history (#77). Every run writes a JSON record (versions, workflow hash, providers and models, status, cost, tokens, duration) to `./agentloom_runs/` or `AGENTLOOM_RUNS_DIR`, and `agentloom history` lists and filters them.
- Quality annotations (#59). `WorkflowResult.annotate(target, quality_score=..., source=...)` records a score after a run and emits it as a `quality:<target>` span tied to the run.
- OTel GenAI semantic conventions (#125). Span, attribute and metric names live in one schema and follow the OTel GenAI registry: `agentloom_tokens_total` becomes `gen_ai.client.token.usage`, `agentloom_provider_latency_seconds` becomes `gen_ai.client.operation.duration` and `agentloom_time_to_first_token_seconds` becomes `gen_ai.client.operation.time_to_first_chunk`, with no aliases. Provider calls get child spans, `llm_call` spans carry prompt metadata (the content only with `capture_prompts: true`), and every span carries `workflow.run_id`. The `on_provider_call` observer hook and the `tokens` argument of `on_step_end` are removed.
- Engine collaborator protocols and shared retry helpers (#112). `agentloom.core.protocols` describes the engine's collaborators, and `agentloom.resilience` exports `compute_backoff`, `is_retryable_exception` and `DEFAULT_RETRYABLE_STATUS_CODES`.
- `retry.retryable_status_codes` (#112), by default `[429, 500, 502, 503, 504]`: a step stops retrying on any other HTTP status; errors without one are still retried.
- Reasoning tokens across providers (#127). `TokenUsage.reasoning_tokens`, `ProviderResponse.reasoning_content` and a `thinking:` block on steps (`enabled`, `budget_tokens`, `level`, `capture_reasoning`), translated for each provider. OpenAI and Gemini report reasoning tokens apart; Anthropic and Ollama count them in the output tokens.
- Weekly pricing refresh (#180). `scripts/update_pricing.py` updates the OpenAI, Anthropic and Gemini prices from LiteLLM's public list, and a workflow opens a pull request every Monday when they change. Prices are never fetched at run time.
- Source distribution and package checks (#190). A release publishes an sdist next to the wheel, `scripts/check_dist.sh` installs both in a clean environment before publishing, and the release notes come from this changelog.
- Python 3.14 (#192), with CI on macOS, Windows and the lowest dependency versions allowed.
- Config file for the CLI (#212). `--config FILE`, `AGENTLOOM_CONFIG` or `./agentloom.yaml` lists the providers with their `models`, `priority`, `base_url` and `is_fallback`; keys still come from the environment, and an `api_key` in the file is refused.
- `agentloom replay` and `verify-determinism` take `--provider` and `--model`, as `agentloom run` does.
- `persistence.existingClaim` in the Helm chart: a volume of yours keeps the run history and checkpoints, so a run paused in one pod resumes in another.
- Grafana: a row for tools, agents, conversations and embeddings, and a panel for mock faults.

### Changed

- `DAG.topological_sort` runs in O((V+E) log V) instead of O(V² log V), in the same order (#112).
- `WorkflowEngine` drops two unreachable exception handlers; behaviour is unchanged (#112).
- Pricing table refreshed (#180): 41 models added, among them Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5, GPT-6 and Gemini 4 Argon, and `gpt-5.6-sol`, `-terra` and `-luna` corrected to OpenAI's prices, checked against the vendors' pages on 2026-11-07.
- Comments and docstrings follow one rule, written in `AGENTS.md`, and the `no-history-comments` pre-commit hook rejects history markers (#213).
- `make docker-build-obs` is now `make docker-build-lite` (#213); the `BUILD_OBSERVABILITY` build arg, which did nothing, and two unreferenced scripts are gone.
- The documentation moved to https://cchinchilla.dev/agentloom/, which follows the version PyPI serves. MkDocs and the old GitHub Pages site are gone, and `scripts/check_docs.py` checks the pages in CI and `make check`.
- The Kustomize placeholder workflow uses the mock provider, so the Job runs in every overlay without keys or Ollama.
- The bundled OTel Collector keeps a series for 24 hours instead of 30 minutes, so the dashboard's totals cover about a day; it holds more series in memory.
- The `production` overlay pins the current release, and the version-linearity check fails when it falls behind.
- Grafana: a `Provider (calls)` selector for the panels labelled with AgentLoom's provider names.

### Breaking changes

- Recordings use format v2, with `_version: 2`, and a request hash that also covers the model, temperature, `max_tokens` and extra arguments (#107). v1 recordings keyed by `step_id` still replay; those keyed by the old hash must be re-recorded. Streams are recorded like `complete()` calls.
- The implicit Ollama fallback is opt-in: set `AGENTLOOM_OLLAMA_FALLBACK=1`, list `ollama` under the config file's `providers:`, or make it the workflow's provider. The Helm chart sets the variable when `ollama.enabled`.
- `agentloom replay` is strict by default: a request that misses the recording, or a step whose prompt, model or tools changed since it was recorded, raises `RecordingMismatchError`, and `--allow-default-fallback` restores the lenient match. Recordings are validated when they load, and those below v2 refused. `run --provider mock` stays lenient.
- Unknown keys in a step or under `config:` are refused at load, `config.responses:` among them (use `responses_file`).
- The default Docker image ships the `[observability]` extra and is about 30 MB larger; `--target production-lite` builds the image without it.
- The Docker images have no pip (#219). An image built `FROM` them that installs packages must bring its own installer, such as `uv` (`COPY --from=ghcr.io/astral-sh/uv`).
- A step missing what its type needs now fails when the workflow loads instead of when it runs: a `router` with neither `conditions` nor `default`, a `tool` without `tool_name`, a `subworkflow` with neither `workflow_path` nor `workflow_inline`, an `embed` without `inputs` and an `llm_call` with neither `prompt` nor `conversation`.
- Unknown keys are refused at load in `retry`, `sandbox`, `notify`, `thinking`, a router's `conditions`, `tool_choice`, `response_schema` and `state_schema`, so a misspelled setting no longer disappears silently.
- `additionalProperties: true` is refused at every level of a `json_schema` response schema, not only at its root; the error names where.
- Two steps of one layer on one `conversation` are refused at load, since parallel writes lose turns; branches of one router are exempt because only one of them runs.
- `speaker` must match `^[A-Za-z0-9_-]{1,64}$`, the names OpenAI accepts.
- A conversation's `summary` counts in `token_count` and in `token_budget`, so trimming starts earlier once a summary exists.
- `WorkflowBuilder.build()` raises `ValidationError` for missing dependencies, router targets and subworkflow files, for cycles and for steps of one layer sharing a conversation.
- `WorkflowParser.from_yaml` reports a missing file as `Workflow file not found` when the argument looks like a path (a `Path`, or one line without a `:`) instead of failing with `must be a mapping`.
- An attachment file is readable under `allowed_paths` as well as `readable_paths` when both are set, as for the file tools.
- An attachment whose `media_type` contradicts its `type` is refused with `AttachmentResolutionError`.
- `http_request` refuses a method outside `GET`, `POST`, `PUT` and `DELETE` before connecting.
- `file_write` reports `written` in bytes, not characters.
- Template rendering in `tool_args` reaches strings inside nested mappings and lists, with the same rules as top-level values. `{value: ..., template: false}` is the escape hatch only with exactly those two keys, so a payload that merely holds them passes through.
- Unknown keys in a step's `tools` and `attachments` entries are refused at load. `ToolDefinition` and `Attachment` themselves keep ignoring them for Python callers.
- A subworkflow's `input` is rendered with the parent's state, as `tool_args` are: `"{state.topic}"` is a template, `"state.items"` hands over the value, and `{value: ..., template: false}` keeps a string as written. Values from redacted keys stay redacted in the child.
- A child's own `state:` seeds its state in the default mode too, as defaults under the parent's values.
- The first retry waits `backoff_base` seconds instead of always 1: the waits are `backoff_base ** (attempt + 1)`, about 2, 4 and 8 seconds by default.
- `tool_choice: required` and `{name: ...}` force the first model call of a tool loop only; the later calls use `auto`, so the model can answer.
- The tools asked for by the reply that reaches `max_tool_iterations` no longer run, since no call would read their results. With `max_tool_iterations: 1` no tool runs.
- A named `tool_choice` that ends the loop without an answer reports `max_tool_iterations_no_answer`, as `required` does.
- A streamed reply that fails structured-output validation fails the step without a retry, as a reply that doesn't stream does.
- A failed router skips its branches under `on_step_failure: continue` too.
- The steps of a layer all read the state as the layer started, so a step held back by `max_concurrent_steps` no longer sees what its siblings wrote. A step still reads what it wrote itself, through `update` too.
- `drop_pairs` stops and lets the request go over the budget when the system messages and the summary reach it, as `drop_oldest` does, instead of emptying the conversation.
- `config.provider` and `--provider` now pick the primary provider: it goes first for any model it can serve, whatever its `priority`, and models listed under another provider included. The other providers that serve the name follow as fallbacks.
- A workflow without `model` takes its provider's default (`gpt-4o-mini`, `claude-haiku-4-5-20251001`, `gemini-2.5-flash`), and `--provider` brings the new provider's; on `ollama` and `generic_openai`, which have none, a step without a model fails at load.
- Models listed in a provider's `models`, and the built-in names such as `gpt-4o-mini`, now fall back to `is_fallback` providers when the provider that lists them fails.
- OpenAI reasoning models (o-series and GPT-5, `gpt-5-chat` apart) send `max_completion_tokens` instead of `max_tokens`, and no `temperature`.
- A step on a Claude model with `thinking` enabled and no `budget_tokens`, or whose `max_tokens` isn't above `thinking.budget_tokens`, fails at load.
- A `responses_file` or `--mock-responses` file that doesn't exist is an error instead of a run on placeholders (exit `2` for `--mock-responses`, `1` for `provider: mock`); `examples/32_yaml_mock.yaml` needs the recording example 31 makes.
- Strict replay of a list of turns without hashes raises `RecordingMismatchError` once the list runs out, instead of repeating its last turn. Lenient replay still repeats it.
- `verify-determinism` exits with `1` when a replay doesn't end in `success`; `--allow-failed` accepts a stable failure.
- Misuse exits with `2` in every command: conflicting options in `replay` and `resume`, a `--live-from` or `--force-live` step the workflow doesn't have, and `resume` of a subworkflow's own checkpoint, which names the run to resume.
- `run --record` on an existing file replaces it. A cut (`replay --record`) still merges into the file.
- A subworkflow's steps are recorded under their path (`sub/step`), LLM entries and `_tools`. Recordings holding them under the bare id still replay.
- Recordings mask the keys of the run's redaction policy in tool call arguments and in tool results that are JSON objects or arrays. `request_hash` still covers the real request, so a later turn that holds a masked value is served in order, unchecked.
- Embedding calls report `gen_ai.operation.name` `embeddings`: their spans are named `embeddings <model>` instead of `chat <model>`, and the `chat` series of `gen_ai.client.operation.duration` and `gen_ai.client.token.usage` lose those samples.
- Failed runs, steps and provider attempts set the span status to `ERROR`, so Jaeger's error filter and status queries now find them. A run counts as failed when it fails, times out or breaks its budget, not when it pauses.
- `gen_ai.request.model` on a step span is the model the step asked for, not the id the provider answered with.
- A subworkflow no longer counts as a run of its own: `agentloom_workflow_runs_total`, `agentloom_workflow_duration_seconds` and `agentloom_cost_usd_total` record only the outermost run, so a subworkflow's cost is no longer counted twice.
- OTLP metrics carry `service.name` `agentloom` instead of `unknown_service`.
- OTLP histograms use the explicit buckets of the Prometheus exporter, per instrument, so their `le` series and percentiles change; `agentloom_workflow_duration_seconds` and `gen_ai.client.token.usage` keep the SDK's defaults.
- Removed configuration that nothing read: the `AGENTLOOM_OBSERVABILITY`, `AGENTLOOM_MAX_CONCURRENT_STEPS`, `AGENTLOOM_BUDGET_LIMIT`, `AGENTLOOM_CHECKPOINT`, `AGENTLOOM_CHECKPOINT_DIR` and `AGENTLOOM_DEFAULT_PROVIDER` variables, their config-file keys, the `log_level` and `log_format` keys, and `timeout` and `max_retries` on a provider entry. A config file that sets one fails at load.
- Removed `retry_with_policy` and `RetryPolicy` from `agentloom.resilience`; the engine used neither.
- `--json` prints only JSON on stdout: `Running workflow:`, `Run ID:` and `Resuming workflow` lines go to stderr with it.
- `runs --json` and `history --json` print `[]` when they find nothing, instead of a sentence.
- `--state` without `=` exits 2, not 1.
- `history --since` and `--until` read a time without an offset as UTC, not local time.

### Fixed

- **Runs and resume.** `agentloom resume` reports a bad run id or checkpoint in one line and prints the run's own id at a later gate (#215). Ctrl-C, or SIGTERM as Kubernetes sends it, saves a `cancelled` checkpoint under `--checkpoint`, which `resume` continues. The workflow's `timeout` is enforced and ends the run `timeout` with a checkpoint (#120). A resumed run goes on from the budget it had spent, `resume` takes `--state` and `--budget`, a subworkflow resumed at its child's gate continues from the child's checkpoint, and resuming past a finished router keeps its untaken branch skipped. A run no longer writes into its workflow's `state:` (#159), `resume` runs `provider: mock` workflows against the mock, `run` without `--checkpoint` prints no `resume` commands, and a checkpoint from a newer AgentLoom is refused with a migration hint.
- **Approval gates and the callback server.** `timeout_seconds` and `on_timeout` are enforced (#210): an expired gate takes `on_timeout`, and `callback-server` resumes expired gates on its own and answers `202` or `409` to late and repeated decisions. `notify.callback_base_url` adds `approve_url` and `reject_url` to the default webhook body, for gates inside subworkflows too. The callback server answers malformed bodies and traversal run ids with `400`, takes `--token`, resumes `provider: mock` workflows and reports provider spans and metrics; webhook deliveries have a deadline.
- **Retries and failures.** A workflow's `config.max_retries` applies to the steps that don't set their own (#211). **Behavior change:** a workflow that sets it to anything but 3 now retries that many times. A failure that can't pass on another attempt fails its step at once: router errors, sandbox refusals, invalid templates, arguments or inputs, replay mismatches, `StateWriteError`, and a request no provider can serve (`UnsupportedRequestError`); `ToolNotFoundError` and `AttachmentResolutionError` are new. `retry.retryable_status_codes` applies to `llm_call` and `embed` steps (#182), `Retry-After` is honoured up to `backoff_max`, and `StepResult.error_classification` says `permanent` or `transient`. A failed step or router skips its dependents (`config.on_step_failure`, by default `skip_downstream`), a step that runs out of retries on its `timeout` fails the run (#120), and the failure log counts the attempts made.
- **Budget, cost and tokens.** Every charge goes through `BudgetEnforcer` under a lock, so a parallel layer can't race past `budget_usd`, and a pause wins over a budget breach in the same layer. Failed attempts, re-asks, summaries and steps cut short are charged and reported, a subworkflow's tokens include its children's reasoning tokens, and a run reports what it spent however it ends (#120). Reasoning tokens are priced at the output rate (#127) and charged once on OpenAI (#120), the pricing table gains undated aliases (#127), `MockProvider` replays the whole recorded usage (#178), and a streamed Anthropic call that fails after `message_start` counts its input tokens.
- **Workflow files, state and templates.** Duplicate step ids, `max_concurrent_steps` outside 1 to 1024 and negative timeouts are refused at load; parallel steps writing one `output` warn, and `config.strict_outputs: true` refuses them. Cycle detection no longer hits `RecursionError` on long chains (#108). `StateManager.update(key, fn)` updates atomically, a dotted write over a scalar raises `StateWriteError`, and `set_sync` and `get_sync` become `_set_sync_unsafe` and `_get_sync_unsafe` (#110). A missing key deep in a template renders empty, `SafeFormatDict(strict=True)` raises at the first one (#110), format specs on dicts and lists fall back to `str`, and `tool_args` treat only real placeholders as templates, with `{value: ..., template: false}` as the escape. Workflow, config, pricing, checkpoint and history files are read as UTF-8 whatever the locale (#120). Redaction patterns match the workflow's data, not AgentLoom's own fields, and reach Pydantic instances. The builder gains `add_embed_step`, `add_approval_gate_step`, `add_agent_step` and `set_state_schema`, and `WorkflowParser.from_yaml` reads long text as YAML.
- **Subworkflows.** `isolated_state`, `input` and `return_keys` keep a child's state apart from its parent's, and an approval gate inside a child pauses the parent as `sub.gate`, which `resume` continues. A child writes `output:` only when it succeeds (#185), and gets its parent's observer, checkpointer, stream callback and `run_id` (#111).
- **Routers.** Conditions can chain safe method calls (`state.x.strip().lower()`), compare lists and dictionaries with `==` and call their read-only methods (`get`, `keys`, `values`, `items`, `count`, `index`).
- **Tools and attachments.** Sync `@tool` functions run on a worker thread, generic type hints give correct schemas, and overwriting a registered tool warns. A tool call whose arguments are no JSON object gets an error back (#120), a missing `state.X` in `tool_args` fails the step, a tool declared without `parameters` uses the registered schema, and `http_request` sends a mapping `body` as JSON. `data:` URLs work as attachments, a raw base64 attachment is held to 20 MB, and a refused symlinked attachment names its target.
- **Providers and the gateway.** Gemini gets schemas it accepts, with unsupported keywords dropped, properties of those names kept, `$ref` inlined and Pydantic constraints translated, and tool loops work on Gemini 3, which needs its `thoughtSignature` back (#120). Anthropic and Google send every system turn (#120), conversations with tool turns reach each provider in its own format, and OpenAI `base_url`s with a path are kept. A 429 becomes `RateLimitError` with `Retry-After`; it no longer trips the circuit breaker, nor do a cancelled stream, a replay miss or a request over `max_tpm`, and a cancelled half-open call gives its probe slot back (#106, #109). The candidate cache is bounded (`AGENTLOOM_CANDIDATE_CACHE_MAX`, #109). A stream falls back to the next provider before its first chunk, runs the tool loop when the step has tools (#175), raises on Anthropic `error` events, and drops Ollama's inline `<think>` trace (#214). OpenAI reasoning models get `reasoning_effort`, Anthropic thinking leaves room for the reply, `model_aliases` maps names for fallbacks, `OLLAMA_BASE_URL` is honoured and Ollama embeddings send `dimensions`.
- **Record and replay.** Every entry carries a `request_hash`, which strict replay checks, and recordings are validated when they load (#107). Tool loops, retried steps and retried subworkflows record and replay as they ran (#163, #164), a loop that fell over to another provider included, and embeddings record their hash too. `run --record` refuses `provider: mock` (#120), `--record` names inline schemas, recordings are read as UTF-8, a cut replay needs no key for the steps it replays, lenient replay warns about a changed request, and the bundled recordings carry hashes. `verify-determinism` and `resume` import the project's modules, as `run` does (#120).
- **Observability.** Steps cut short, provider calls cut by a timeout or a cancel, and refused embedding attempts close their spans (#169), and parallel subworkflows keep their spans apart (#147). An observer that lacks a hook or raises no longer fails the step or sends a webhook again. `tool` steps emit the `execute_tool` span and the tool metrics, the budget and circuit gauges update on every change and stay bounded (#111), and `agentloom_workflow_duration_seconds` has buckets from 0.1 s.
- **Grafana.** Two panels no longer share an id. Panels read counters directly instead of `rate()` on one-shot series, leave out pending gates and paused runs, and show one series per provider or workflow; totals show the counter's value instead of adding up its samples, which reported 279 runs for 14.
- **CLI.** `--version`, `-h` on every command, `--log-level` and the `AGENTLOOM_LOG_*` variables work, with logs on stderr. `--state` decodes JSON values, `visualize` names every step type and rejects an unknown `--format`, Mermaid output keeps colliding ids apart and router conditions whole, and the run summary marks timeouts `[TIMEOUT]`. `--provider` over a `generic_openai` workflow warns that it drops `base_url`, the CLI's mock has no rate limit, `info` lists `jsonschema`, and the run history no longer warns about a read-only default directory.
- **Deployment and packaging.** The Argo CD Application, the example Jobs and the Helm chart work as shipped (#216): the Application passes a workflow and the provider keys' Secret and labels the namespace for Pod Security; the Jobs meet the `restricted` profile and call the right Python; the chart's quota drops `limits.cpu`, its run waits for Ollama's pull, its Ollama image is pinned and `values.schema.json` checks its values. The `production` overlay reaches the collector by its full name, and examples 36 and 41 run inside the image. A release publishes its Docker image and the chart installs the released one (#190), and the declared minimum dependency versions install a working package (#192).
- **Docs and examples.** The docs and `--help` texts promise only what the code does, the validation scripts and their Jobs pass (#213), and examples 16, 17, 18, 19, 22, 30 and 38 show what they describe.

### Security

- Router validator and state wrappers (#050). A subscript with a `_`-prefixed key or a non-constant index is refused, state keys are reachable only as `state.X`, and templates no longer reach a state wrapper's raw data, which leaked keys such as `_secret`.
- Router expressions hardened against dunder access and `type` (#104), closing [GHSA-c37m-mv4j-972v](https://github.com/cchinchilla-dev/agentloom/security/advisories/GHSA-c37m-mv4j-972v): router conditions could run arbitrary code. The validator raises the new `SecurityError`.
- `approval_gate.notify.url` passes the workflow sandbox (#051). With the sandbox disabled, a built-in deny-list still blocks loopback, link-local and cloud metadata addresses, private ranges and schemes other than HTTP(S), in IPv4-mapped, encoded and IDN forms too; `sandbox.allow_internal_webhook_targets: true` waives only the internal-host check. `shell_command` checks the paths in `--key=path` and `key=path` arguments.
- `ToolSandbox.validate_path` raises `SandboxViolationError` for every invalid path, null bytes, symlink loops and non-strings included (#051).
- State redaction for persisted artefacts (#052). `state_schema: {key: {redact: true}}` or `AGENTLOOM_REDACT_STATE_KEYS` mask the matching keys, glob patterns included, as `<REDACTED:sha256=...>` in checkpoints, results, `--json` output, webhooks and captured prompts; subworkflows inherit the policy and the sandbox. Memory keeps the real values, and a redacted checkpoint resumes with the sentinel, not the secret.
- Tool sandbox hardened (#105). Meta-executables (`env`, `sh`, `bash`, `python`, `xargs` and others) need `danger_opt_in`, one by one; relative paths can't leave the working directory; URL schemes default to `http` and `https` (`allowed_schemes`); process substitution is refused. **Behavior change:** list each meta-executable a workflow runs in `danger_opt_in`.
- The Docker images run on `python:3.14-slim` without pip (#219); their known vulnerabilities with a fix go from 14 to 3.
- Dependency updates and hardened workflows (#194): weekly Dependabot updates, workflows without stored git credentials and with write access only where needed, zizmor in `make check` and CI, and `SECURITY.md` for private reports.
- `anyio` 4.14.2 or later (#208), past GHSA-82r6-8w77-94w6 (TLS host names encoded with IDNA 2003), and locked dependencies past their published advisories.
- Tighter defaults for the tools, the callback server and the providers (#220). A model's tool call runs only if its step declares the tool; a sandboxed `http_request` refuses internal hosts that `allowed_domains` doesn't name and drops credential headers from its result; `shell_command` honours `timeout` and kills the processes it started; sandboxed attachments need `readable_paths` or `allowed_paths`; `callback-server --token` also guards `GET /pending`, compares in constant time and caps headers and bodies; the Google key goes in a header, not the URL; `agentloom info` shows no part of a key. **Behavior change:** list internal hosts in `allowed_domains` and readable paths for attachments, and don't expect credential headers in `http_request` results.
- `http_request`, approval-gate webhooks and URL attachments connect to the addresses their internal-host check resolved, keeping the host name for TLS, so DNS rebinding can't reach an internal host (#220). **Behavior change:** where that check runs, a host that doesn't resolve is refused.
- A sandboxed `shell_command` refuses anything the shell would expand (`$`, `~`, `*`, `?`, `[`, brace lists), even inside quotes.
- A sandboxed `shell_command` runs with only `PATH`, `HOME`, `LANG`, `LC_ALL` and `TZ` in its environment, so it can't read provider keys.
- A command given as a path to a sandboxed `shell_command` must be the file its allowlisted name resolves to on `PATH`.
- A subworkflow runs under the strictest of its sandbox and its parent's, and inherits the parent's when it has none; `sandbox.deny_internal_hosts` keeps a child's `allowed_domains` off the internal hosts its parent refuses.
- The sandbox checks paths in short flags (`grep -f/etc/shadow`), paths containing `=` and bare names that are symbolic links.
- `find` needs `danger_opt_in`, since its `-exec` runs anything.

## [0.4.0] - 2026-04-15

### Added

- `agentloom replay <workflow.yaml> --recording <file.json>` subcommand — re-executes a workflow against recorded responses with no API calls (#61)
- YAML-configured MockProvider — `provider: mock` with `responses_file`, `latency_model`, `latency_ms` fields on `WorkflowConfig` (#76)
- Production `MockProvider` and `RecordingProvider` for deterministic replay and offline evaluation (#76)
  - `MockProvider` loads responses from a JSON file, keyed by `step_id` or SHA-256 prompt hash
  - Latency models: `constant`, `normal` (gaussian with seed), `replay` (uses recorded `latency_ms`)
  - `RecordingProvider` wraps any provider, captures completions to JSON, flushes per-call
  - `agentloom run --mock-responses <file>` replays; `--record <file>` captures
- Webhook notifications for approval gates — outbound HTTP on pause (#42)
  - `WebhookConfig` on `StepDefinition.notify` with URL, custom headers, and body template
  - Async webhook sender with 3-retry exponential backoff (best-effort, never blocks pause)
  - `agentloom callback-server` command — lightweight HTTP server for programmatic approve/reject
  - Routes: `POST /approve/<run_id>`, `POST /reject/<run_id>`, `GET /pending`
  - Shared template utilities extracted to `core/templates.py`
  - `StepContext` now carries `run_id` and `workflow_name` for webhook context
  - Grafana dashboard "Human-in-the-Loop" row with approval gate and webhook panels
  - Prometheus metrics: `approval_gates_total`, `webhook_deliveries_total`, `webhook_latency_seconds`
  - OTel span attributes: `approval_gate.decision`, `webhook.status`, `webhook.latency_s`
  - Example workflow (30), validation script, and K8s smoke job
- Approval gate step type — human-in-the-loop decision point (#41)
  - `StepType.APPROVAL_GATE` pauses the workflow and waits for human approval or rejection
  - Decision injected via `_approval.<step_id>` state key on resume
  - `--approve` / `--reject` mutually exclusive flags on `agentloom resume`
  - `timeout_seconds` and `on_timeout` schema fields (consumed by webhook callback server in #42)
  - Example workflow (29), validation script, and K8s smoke job
- Workflow pause mechanism — foundation for human-in-the-loop (#40)
  - `PauseRequestedError` exception for step executors to signal a pause
  - `StepStatus.PAUSED` and `WorkflowStatus.PAUSED` status values
  - Engine catches pause requests, saves checkpoint with `status=paused` and `paused_step_id`, and returns cleanly
  - Resume from paused checkpoint skips completed steps and re-runs the paused step
  - CLI treats paused workflows as non-error (exit code 0)
  - Functional validation script (`scripts/validate_pause_resume.py`) and K8s smoke job
- Pluggable checkpoint backends with `BaseCheckpointer` protocol and `FileCheckpointer` default (JSON-to-disk) (#78)
  - `CheckpointData` Pydantic model with full workflow state serialization
  - Engine integration: auto-generates `run_id`, saves checkpoint on completion/failure, graceful handling of I/O errors
  - `WorkflowEngine.from_checkpoint()` classmethod to reconstruct and resume from a checkpoint, skipping completed steps
  - `agentloom run --checkpoint` and `--checkpoint-dir` flags
  - `agentloom resume <run_id>` CLI command to resume paused or failed workflows
  - `agentloom runs` CLI command to list all checkpointed runs
  - Example workflow (28) and documentation

## [0.3.0] - 2026-04-12

### Added

- Documentation site with mkdocs-material — getting started, architecture, providers, workflow YAML reference, Python DSL, graph API, examples, observability, deployment, contributing, and changelog pages. Auto-deployed to GitHub Pages on push to main (#72)
- Multi-modal input support for `llm_call` steps — images, PDFs, and audio via `attachments` field (#68)
  - Provider-native formatting: OpenAI (images, audio), Anthropic (images, PDFs), Google (images, PDFs, audio), Ollama (images)
  - URL fetching with `fetch: local` (default) or `fetch: provider` passthrough
  - SSRF protection: blocks private/reserved IP ranges (RFC 1918, loopback, link-local)
  - Sandbox integration: `allowed_domains`, `allow_network`, and `readable_paths` enforced for attachments
  - Attachment size limit (20 MB default)
  - `attachment_count` in `StepResult`, OTel span attribute, and `agentloom_attachments_total` metric
  - Grafana dashboard "Multi-modal" row with attachments panels
  - Multi-modal workflow examples (19–24)
- Streaming support for LLM responses with real-time token output (#3)
  - `StreamResponse` accumulator with per-provider SSE/NDJSON parsing
  - All 4 providers: OpenAI (SSE), Anthropic (SSE), Google (SSE), Ollama (NDJSON)
  - Gateway `stream()` with circuit breaker + rate limiter integration
  - `config.stream: true` (workflow-level) and per-step `stream:` override
  - CLI `--stream` flag for real-time terminal output
  - `time_to_first_token_ms` in `StepResult` and OTel span attributes
  - `agentloom_stream_responses_total` and `agentloom_time_to_first_token_seconds` metrics
  - Grafana "Streaming" dashboard row with TTFT quantiles
  - Streaming examples (25–26)
- `AGENTLOOM_*` env var prefix for all configuration overrides (#5)
- YAML-based pricing table replacing hardcoded Python dict (#6)
- Provider auto-discovery moved from CLI hack to `config.discover_providers()`
- Ollama e2e integration tests against a live Docker instance (5 smoke tests) (#71)
- CI workflow `e2e-ollama.yml` — weekly schedule, `release/**` branches, `e2e` label on PRs, manual dispatch
- Array index support in state paths (e.g., `state.items[0]`, `items[0].name`, `results[-1]`)
  - `_parse_path()` helper with regex-based bracket parsing in `StateManager`
  - `_resolve_key()` and `_set_nested()` handle list indexing with bounds checking
  - `DotAccessList` wrapper for `str.format_map()` template rendering
  - `ToolStep._resolve_args()` refactored to reuse `StateManager._resolve_key()`
  - CLI, Docker, and K8s smoke tests; example workflow (27)
- First-class graph API for workflow DAG analysis and export (#75)
  - `WorkflowGraph` class with `from_workflow()` and `from_dag()` factories
  - `GraphNode` and `GraphEdge` frozen Pydantic models
  - Path algorithms: `all_paths()`, `prime_paths()`, `critical_path()`
  - Export formats: `to_dict()`, `to_dot()` (Graphviz), `to_pnml()` (Petri Net), `to_mermaid()`
  - Optional `to_networkx()` via `pip install agentloom[graph]`
  - Properties: `nodes`, `edges`, `roots`, `leaves`, `layers`
- Test coverage reporting via Codecov with 85% minimum threshold and README badge (#70)

## [0.2.0] - 2026-03-30

### Added

- Kubernetes manifests with Kustomize overlays for dev, staging, and production (#24)
- Helm chart with Job/CronJob modes and render-time input validation (#25)
- Terraform configuration for local kind cluster with full observability stack (#26)
- ArgoCD Application CRD with automated sync and Job immutability handling (#27)
- Docker CI/CD workflow for multi-arch GHCR publishing (#23)
- Infrastructure audit scripts for static and integration validation
- Infrastructure documentation (#28)

### Fixed

- Production NetworkPolicy OTel egress restricted to observability namespace
- Read-only filesystem audit check no longer false-passes when root FS is writable
- Terraform audit phase passes KUBECONFIG to all kubectl poll commands
- Removed duplicate kubeconform invocation that hung without stdin
- Terraform secret uses `string_data` instead of `data` for plaintext values
- GitHub Actions and image versions pinned to commit SHAs

## [0.1.2] - 2026-03-26

### Added

- Sandbox enforcement for built-in tools — command allowlist, path restrictions (read/write separation), network domain filtering, shell operator injection prevention, write size limits (#4)
- `SandboxConfig` model in workflow YAML (`config.sandbox.*`)
- `SandboxViolationError` exception
- Sandbox workflow examples (`17_sandbox_allowed`, `18_sandbox_blocked`)

### Fixed

- Step executors (`llm_call`, `router`, `tool_step`) now use `await get_state_snapshot()` instead of sync `.state` access (#8)
- Removed deprecated `gemini-2.0-flash` model

## [0.1.1] - 2026-03-22

### Fixed

- Rate limiter now accounts for response tokens, not just prompt tokens (#11)
- README header image uses absolute URLs for PyPI compatibility (#2)

## [0.1.0] - 2026-03-19

First public release.

### Added

- YAML and Python DSL workflow definitions (DAGs with sequential + parallel steps)
- Step types: `llm_call`, `tool`, `router` (conditional), `subworkflow`
- Provider gateway with automatic fallback (OpenAI, Anthropic, Google, Ollama)
- Circuit breaker, rate limiter, and retry with exponential backoff per provider
- Budget enforcement (hard stop when USD limit exceeded)
- Cost tracking per step, model, and provider
- OpenTelemetry traces + Prometheus metrics (optional, `pip install agentloom[all]`)
- CLI commands: `run`, `validate`, `visualize` (ASCII + Mermaid), `info`
- Checkpointing: save and resume workflow state to disk
- 392 tests, mypy strict, ruff clean

### Known Limitations

- ~~No streaming support (falls back to full completion)~~ (fixed in Unreleased)
- Router expressions use first-match-wins, no priority ordering
- ~~Rate limiter doesn't account for response tokens (only prompt tokens)~~ (fixed in 0.1.1)
- ~~Provider discovery from env vars only, should be a config file~~ (fixed in Unreleased)
- ~~Shell command tool has no sandboxing (FIXME in code)~~ (fixed in 0.1.2)
- ~~File tools accept arbitrary paths (no path sanitization)~~ (fixed in 0.1.2)
- Router expressions use `eval()` — must be trusted input (not user-facing)
- ~~Pricing table hardcoded in Python, should be YAML config~~ (fixed in Unreleased)
- No array index support in state paths (e.g., `state.items[0]`)
- ~~Sync state access in step executors bypasses async lock~~ (fixed in 0.1.2)
- Budget enforcement is post-hoc — a single expensive step can overshoot before being stopped
- `budget_remaining` metric only emitted to Prometheus, not OTel
- Checkpoint `save_checkpoint` uses blocking I/O inside async method

### Design Decisions

- **httpx over provider SDKs** — keeps dependencies minimal (~5 core).
  Trade-off: we maintain thin adapters instead of using official SDKs.
- **anyio over raw asyncio** — structured concurrency via task groups.
  Slightly less familiar but much safer for parallel step execution.
- **str.format_map over Jinja2** — one fewer dependency; prompt templates
  don't need loops or conditionals. SafeFormatDict handles missing keys.
- **Observability optional** — core runs without opentelemetry or prometheus.
  NoopSpan/NoopTracer pattern gives zero overhead when not installed.
- **Pydantic v2** — validation and serialization worth the Rust compilation
  trade-off. Could revisit for truly minimal environments.

[Unreleased]: https://github.com/cchinchilla-dev/agentloom/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cchinchilla-dev/agentloom/releases/tag/v0.1.0
