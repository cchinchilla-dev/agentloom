# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native tool / function calling across providers (#116). New `tools`, `tool_choice`, and `max_tool_iterations` fields on `llm_call`; the LLM step dispatches via the existing `ToolRegistry`, feeds results back, and re-prompts until the model stops asking (capped by `max_tool_iterations`, default 5). Each adapter translates the unified `ToolDefinition` list to its native shape (OpenAI / Ollama / Anthropic / Google). Parallel tool calls dispatch concurrently and preserve order; failures are reported back as text so the model can recover. Sandbox (#105), budget (#108), and per-step retry (#106) apply unchanged. `MockProvider` recordings accept a list of turns per step so offline replay drives the loop end-to-end; `examples/35_tool_calling.yaml` ships a ReAct-style example. The OpenAI-shaped parser also handles Ollama-compat responses (no `type` field, `arguments` as a decoded dict) — without this Ollama tool calling silently dropped every call.
- Per-run experiment metadata logging (#77). Every workflow execution now writes a self-contained JSON record (`run_id`, ISO timestamp, AgentLoom version, Python version, workflow `sha256` hash, list of `provider/model` pairs used, status, total cost, total tokens, step count, duration) to `./agentloom_runs/<run_id>.json`. Override the directory via the `runs_dir` constructor argument on `RunHistoryWriter` or the `AGENTLOOM_RUNS_DIR` env var. Disk I/O happens in a worker thread so the write does not block the event loop. Records carry a `_schema_version: 1` field; failures during the write are logged and swallowed so a broken history directory cannot prevent the engine from returning the result. New `agentloom history` CLI subcommand lists records most-recent-first and accepts `--workflow`, `--provider`, `--since YYYY-MM-DD`, `--until YYYY-MM-DD`, `--min-cost`, `--max-cost`, `--limit`, and `--json` filters — covering the full filter surface (date, workflow, cost, provider) called for in the original issue.
- Quality annotations attachable to `WorkflowResult` (#59). New `WorkflowResult.annotate(target, quality_score=..., source=..., **metadata)` method appends a typed `QualityAnnotation` (`target`, `quality_score`, `source`, `metadata`) to the result so evaluators, human reviewers, or downstream scoring code can record output quality after the run completes. **The annotation is auto-emitted as an OTel span** the moment `annotate()` runs whenever the engine returned the result with a tracing context attached (the default for any workflow run with observability enabled) — `result.annotate("answer", quality_score=4.5, source="human_feedback")` becomes immediately visible in Jaeger with no additional plumbing on the caller side. Each annotation is published as a standalone `quality:<target>` span (the workflow span has already closed, so retroactive attribute attachment is not viable). Quality spans carry `workflow.run_id` and `workflow.name` plus `agentloom.quality.score`, `agentloom.quality.source`, `agentloom.quality.target`, and free-form `agentloom.quality.metadata.*` attributes — Jaeger / Tempo can group quality spans with the original trace by run_id, and dashboards can filter for `agentloom.quality.score < threshold` to surface regressions. Offline / replay paths that build a `WorkflowResult` without a live tracer keep working — `annotate()` still records the data on the result, the OTel emission just no-ops. The `agentloom.observability.quality.emit_quality_annotation` / `emit_quality_annotations` helpers remain available for callers that build annotations outside the engine flow (e.g. batch evaluators reading historical results from disk).
- OTel span and metric schema centralization with GenAI semantic conventions (#125). The schema is a clean break — no compatibility shims for pre-#125 attribute or metric names. New `agentloom.observability.schema` module is the single source of truth for span / attribute / metric names; downstream consumers (Grafana, AgentTest, Jaeger plugins) parse a stable contract instead of grepping for ad-hoc strings. **Metrics renamed and retyped** to match the OTel GenAI registry: `agentloom_tokens_total` (counter) → `gen_ai.client.token.usage` (histogram, `{token}` unit) with `gen_ai.token.type` attribute (`input` / `output` / `reasoning`); `agentloom_provider_latency_seconds` (histogram) → `gen_ai.client.operation.duration` (histogram, `s`) with `gen_ai.operation.name` + `gen_ai.provider.name` attributes; `agentloom_time_to_first_token_seconds` → `gen_ai.client.operation.time_to_first_chunk`. AgentLoom-specific metrics (`agentloom_workflow_*`, `agentloom_step_*`, `agentloom_provider_calls_total`, `agentloom_cost_usd_total`, `agentloom_circuit_breaker_state`, `agentloom_budget_remaining_usd`, HITL / mock / recording counters) keep their `agentloom_` prefix — they have no OTel equivalent. The bundled Grafana dashboard is updated to query the new metric / label names. The legacy `Observer.on_provider_call` hook (which duplicated the metric emission already done by `on_provider_call_end`) is removed; the engine no longer fires it. The `tokens: int` positional argument on `on_step_end` is removed — callers now pass `prompt_tokens` / `completion_tokens` as kwargs. Span attributes follow the **canonical OTel GenAI registry** as of the May 2026 spec — `gen_ai.provider.name` (the deprecated `gen_ai.system` is **not** emitted), `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.request.temperature`, `gen_ai.request.max_tokens`, `gen_ai.request.stream`, `gen_ai.response.model`, `gen_ai.response.finish_reasons` (array of strings, per spec), `gen_ai.response.time_to_first_chunk`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.reasoning.output_tokens`. Errored inference spans also emit the OTel general-conventions attribute `error.type` alongside the AgentLoom-specific `step.error` so OTel-aware consumers (Jaeger error filters, Tempo) light up. Inference spans use the canonical name template `"{operation_name} {model}"` (e.g. `"chat gpt-4o-mini"`); workflow / step orchestration spans keep the AgentLoom-specific `workflow:*` / `step:*` names. AgentLoom-specific fields stay under `workflow.*` / `step.*` / `tool.*` / `agentloom.*` namespaces. Provider names are translated from AgentLoom internal names to OTel registry values via `to_genai_provider_name` (e.g. `google` → `gcp.gemini`). Notable additions:
  - **Provider-level child spans** — gateway emits one `provider:<name>` span per fallback attempt nested under the parent step span, on both `complete()` and `stream()` paths, so the latency split between LLM API time and step orchestration overhead (and across fallback attempts) is visible in Jaeger.
  - **Prompt metadata capture** — `agentloom.prompt.hash`, `agentloom.prompt.length_chars`, `agentloom.prompt.template_id`, `agentloom.prompt.template_vars` land on `llm_call` spans by default. Full prompt content stays off; opt in via `config.capture_prompts: true` in the workflow YAML for debugging or trusted environments. Captured prompts ride as an `agentloom.prompt.captured` span event (not an attribute, to avoid blowing the attribute-size budget).
  - **`workflow.run_id` propagation** — every workflow / step / provider span inherits the per-execution UUID, so external systems can correlate traces with checkpoints.
  - **Reasoning tokens span attribute** — emitted under `gen_ai.usage.reasoning.output_tokens` (the canonical OTel registry name, available since the spec added it for o-series / Claude thinking / Gemini thinking). The earlier `step.reasoning_tokens` (#127) and the transitional `agentloom.gen_ai.usage.reasoning_tokens` are gone.
  - **`MetricName` drift detection** — the schema's `MetricName` constants now match exactly what `metrics.py` emits, and a regression test (`test_metric_names_match_emissions`) scans the source for `agentloom_*` literals so any drift becomes a CI failure rather than silent dashboard breakage.
- Engine collaborator protocols and shared retry primitives (#112). New `agentloom.core.protocols` module exposes `StateManagerProtocol`, `GatewayProtocol`, `ToolRegistryProtocol`, `ObserverProtocol`, `CheckpointerProtocol`, and `StreamCallbackProtocol` as `@runtime_checkable` `typing.Protocol` shapes — `StepContext` collaborator fields stay typed as `Any` plus a comment naming the protocol (the `checkpointer` field stays on the concrete `BaseCheckpointer` carried over from #111 to keep Pydantic v2 forward-ref resolution happy). Call sites that import the protocols still get type-checker coverage. `agentloom.resilience` re-exports `compute_backoff`, `is_retryable_exception`, and `DEFAULT_RETRYABLE_STATUS_CODES` so the engine and `retry_with_policy` share a single source of truth for retry waveforms and retryability rules.
- `RetryConfig.retryable_status_codes` (#112) — workflow YAML field, default `[429, 500, 502, 503, 504]`. Step retries now bail out immediately on permanent failures (4xx not in the list) instead of consuming the retry budget; status-less exceptions (network errors, generic provider failures) stay retried as transient. Configurable per step under `retry.retryable_status_codes`.
- Reasoning / extended-thinking tracking across providers (#127). `TokenUsage` gains a `reasoning_tokens` field plus a `billable_completion_tokens` property; `ProviderResponse` gains a `reasoning_content` field for providers that expose the chain-of-thought trace. A new `ThinkingConfig` submodel on `StepDefinition.thinking` lets workflow authors activate provider-side reasoning from YAML (`enabled`, `budget_tokens`, `level`, `capture_reasoning`); the step layer forwards the config object under a single `thinking_config` kwarg and each adapter translates it to its own request shape. Token metrics emit a third `direction="reasoning"` observation when reasoning tokens are non-zero, and the step span carries the canonical OTel `gen_ai.usage.reasoning.output_tokens` attribute (renamed from the original `step.reasoning_tokens` via the OTel GenAI alignment in #125). Per-provider coverage:
  - **OpenAI o-series** — parses `completion_tokens_details.reasoning_tokens` from chat-completion and streaming responses; o1 / o3 / o4-mini infer reasoning from the model name (`thinking_config` is accepted for YAML uniformity but not forwarded to the wire).
  - **Anthropic** — concatenates `type="thinking"` content blocks into `reasoning_content`; `ThinkingConfig` translates to the `thinking: {type: "enabled", budget_tokens}` payload, and an explicit raw `thinking={...}` kwarg still wins for advanced callers. Documented limitation: Anthropic rolls thinking tokens into `output_tokens` rather than exposing a separate field, so `reasoning_tokens` stays `0` for this provider — cost is automatically correct because `output_tokens` already includes the thinking volume, but the visible-vs-thinking split is unavailable from the wire.
  - **Google Gemini 2.5+** — parses `usageMetadata.thoughtsTokenCount` (defensive against the field's intermittent absence on `gemini-3-flash-preview`); `ThinkingConfig` translates to `generationConfig.thinkingConfig` (`thinkingBudget`, `thinkingLevel`, `includeThoughts`); thought summary parts (`thought=true`) are split into `reasoning_content` keeping `content` clean.
  - **Ollama 0.9+** — parses `message.thinking` into `reasoning_content`, with a regex fallback that strips legacy inline `<think>...</think>` tags from `content`. `ThinkingConfig` translates to the top-level `think` request param (`<level>` if set, else `true`). Documented limitation: Ollama does not split `eval_count` between thinking and visible tokens, so `reasoning_tokens` stays `0` for this provider; cost is unaffected (local models are free).

### Changed

- `DAG.topological_sort` switched from a sorted-list-as-priority-queue (O(V² log V)) to `heapq.heapify` + `heappop`/`heappush` (O((V+E) log V)) (#112). Same deterministic ordering — nodes still come out by alphabetical id at each layer — so existing workflows produce identical execution plans. Notable for runs with hundreds of steps.
- `WorkflowEngine` no longer carries dead `BudgetExceededError` / `WorkflowTimeoutError` handlers (#112). Anyio wraps both into `ExceptionGroup`, and the engine already unwraps them via the catch-all branch — the dedicated handlers were unreachable. Behaviour is unchanged; the resulting workflow status mapping (`BUDGET_EXCEEDED`, `TIMEOUT`) still applies.

### Breaking changes

- Recording file format bumped to v2 with a versioned envelope (#107). JSON files written by `RecordingProvider` (and consumed by `MockProvider` / `agentloom replay`) now carry a top-level `_version: 2` key alongside the captured entries (which sit at the top level themselves, keyed by `step_id` or request hash). The reader treats any top-level key starting with `_` as metadata and ignores it, so v1 recordings load without errors. **However**, the request-hash algorithm changed in 0.5.0 — it now mixes model + temperature + max_tokens + extra alongside messages — so v1 recordings keyed only by the legacy messages-only hash will not match under 0.5.0+ and need to be regenerated. v1 recordings keyed by `step_id` continue to replay unchanged. Streaming responses are now keyed under the same hash as `complete()` calls and persist the joined chunk content in the same entry shape (`content`, `usage`, `cost_usd`, `latency_ms`, `finish_reason`), so a recording captures both modes uniformly.

### Fixed

- Harden gateway resilience: stream cancellation no longer trips the circuit breaker, circuit-breaker check now precedes the rate limiter in `complete()`, retry backoff jitter is centralized in `_jittered_backoff`, `RateLimiter` validates `max_rpm >= 1` / `max_tpm >= 1` and fails fast when `token_count > max_tpm`, and `CircuitBreaker.state` is a pure read with the half-open transition isolated in `_maybe_transition_to_half_open()` (#106).
- Record/replay correctness — `anyio.Lock` around `_recorded` writes plus per-call flush, streaming captures persist chunks under the same key as `complete()`, `prompt_hash` now includes model/temperature/max_tokens/extra and uses `model_dump()` for Pydantic-aware hashing (#107).
- Normalize provider adapters — central `providers/_http.py` helper with `validate_extra_kwargs` + `raise_for_status`; each provider declares its own kwargs allowlist; HTTP 429 now becomes `RateLimitError` with `Retry-After` parsed; the gateway passes `RateLimitError` to `CircuitBreaker.call(exclude=...)` so rate-limit responses do not trip the breaker; pricing prefix-match runs longest-first; `OllamaProvider` honours `OLLAMA_BASE_URL`; the Google adapter warns when streaming responses lack `usageMetadata` (#109).
- Bound the gateway candidate cache with LRU eviction so long-lived workflows do not accumulate stale provider/model entries — default 1024 entries, override via `AGENTLOOM_CANDIDATE_CACHE_MAX` env var (#109).
- DAG correctness — skip propagation closes over transitive successors via `dag.transitive_successors`; pause requests no longer raise inside `_execute_step` and instead surface after the layer finishes; pre-dispatch budget gate rejects steps before they consume more budget; cycle detection switched to an iterative algorithm so deeply chained DAGs no longer hit `RecursionError`; `_set_nested` now reports auto-expansion of intermediate lists with a clear message (#108).
- Template hardening — opt-in strict mode for template rendering: `SafeFormatDict(strict=True)` and `DotAccessDict(strict=True)` raise `TemplateError` on missing keys; default behaviour (warn + render empty) is unchanged. `__format__` now honours `format_spec`. `ToolStep._resolve_args` renders `{state.x}` substitutions consistently with `llm_call` (#110).
- State and approval-gate cleanup — the unsafe `state_manager.{set_sync,get_sync}` accessors are renamed to `_set_sync_unsafe` / `_get_sync_unsafe` (legacy names removed in this release); approval-gate UX moved out of the step body into the CLI rendering layer for consistency (#110).
- Subworkflow observability + checkpointer propagation — `SubworkflowStep` forwards `observer`, `checkpointer`, `on_stream_chunk`, and `run_id` to the child engine; checkpoint JSON serialization is moved off the event loop into a worker thread; the `NoopObserver` now implements every hook; observer hooks accept `**kwargs` for forward-compat; webhook deliveries get a configurable deadline (default 5s) with status `timeout`; `StepContext.checkpointer` is now plumbed through (#111).
- Bound the metrics gauge dictionaries `_circuit_states` and `_budget_remaining` with LRU eviction so long-running deployments cannot grow per-provider or per-workflow cardinality without bound (#111).
- Cost calculation for reasoning models — `calculate_cost()` now sums `completion_tokens + reasoning_tokens` against the output rate, so workflows using OpenAI o-series, Gemini 2.5+ thinking, or Anthropic extended thinking are no longer undercharged. Budget enforcement and Prometheus cost metrics inherit the fix transitively (#127).
- Pricing table refresh — `pricing.yaml` extended to ~70 entries covering OpenAI GPT-5/5.1/5.2/5.3/5.4/5.5 family (with `*-codex`, `*-mini`, `*-nano`, `*-pro` tiers), GPT-4.1 family, the full o-series including `o1-pro` / `o3-pro` / `o3-deep-research` / `o4-mini-deep-research`; Anthropic adds `claude-opus-4-7`, `claude-3-7-sonnet`, `claude-opus-3`, `claude-haiku-3`, plus undated aliases (`claude-sonnet-4-5`, `claude-opus-4-1`, `claude-sonnet-4-0`, etc.) so callers pinning to a family name resolve correctly without the date suffix; Google adds `gemini-3-pro`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-image-preview`, `gemini-3.1-flash-lite-preview`, `gemini-2.0-flash-lite`. The longest-prefix lookup in `calculate_cost()` keeps dated entries authoritative when both alias and dated form are present (#127).
- Relax router AST validator to accept chained safe method calls (#053). Until 0.5.0 the validator refused any `ast.Call` whose receiver was itself a call, so the natural `state.x.strip().lower() == "value"` predicate failed AST validation with `SecurityError: Attribute calls are only allowed on names, attributes, or subscripts`. Seven official example YAMLs were affected (`05_content_moderation`, `06_lead_qualification`, `07_incident_triage`, `09_fraud_detection`, `11_insurance_claims`, `12_log_analysis`, `14_custom_tools_decorator`). A new recursive `_safe_receiver` helper accepts a chain as long as every link is an attribute on a `Name | Attribute | Subscript` base — and every link still passes through `_reject_attribute`, so the dunder/blocklist filter blocks `state.x.lower().__class__` and `''.join(['a','b'])` exactly as before. Regression suite in `tests/steps/test_router_security.py::TestRouterChainedSafeCalls` covers both the positive expansions (chained `.strip().lower()`, slice-then-method, membership-after-chain) and the negative boundary (calls on literals, dunder after safe chain, disallowed builtins).
- Cascade-skip dependents when a router or step ends in FAILED (#054). When a router step failed for any reason (AST error, no-match-no-default, evaluator exception) every downstream branch still executed against partial state — examples were observed shipping Slack notifications, Jira tickets, and approval webhooks off runs that should have aborted. Same defect on non-router steps: a step that exhausted retries left its dependents to run with empty state. The engine layer-loop now collects FAILED steps in each layer, computes `dag.transitive_successors` of their direct children, and marks the closure as `SKIPPED` with an `error` field naming the closest failed ancestor (BFS backward through the DAG) so the operator can audit the chain. New `config.on_step_failure: "skip_downstream" | "continue"` knob — default `skip_downstream` for the new safe behaviour, `continue` for fan-outs that explicitly want today's swallow-and-continue semantics. Regression suite at `tests/core/test_engine.py::TestFailureCascade` covers failed router → direct children, failed router → transitive descendants, no-match-no-default cascade, non-router step failure cascade, and the `continue` opt-out.
- Parse-time invariants on workflow definitions (#055). Three footguns became hard errors: (a) duplicate step ids — `WorkflowDefinition` gains a `@model_validator(mode="after")` that refuses two steps with the same `id` and names both source indices (`id='a' at indices [0, 1]`), closing the silent-shadowing path where one of the two steps was lost from `final_state.steps`; (b) `config.max_concurrent_steps` carries `Field(default=10, ge=1, le=1024)` — `0` (which deadlocked `anyio.CapacityLimiter(0)`) and negatives (which surfaced as cryptic `total_tokens must be >= 0`) now raise a clean Pydantic error at parse time; (c) parallel-eligible steps writing the same `output:` key emit a `UserWarning` listing both step ids — promote to a parse error via `config.strict_outputs: true`. Sequential overwrite via `depends_on` is exempt (intentional pattern). Regression suite at `tests/core/test_parser.py::TestParseTimeInvariants`.
- Atomic `StateManager.update` primitive and dotted-write scalar refusal (#056). The natural `cur = await sm.get('counter'); await sm.set('counter', cur + 1)` pattern collapsed 50 parallel writers to 1 because the lock dropped between the two awaits. The new `update(key, fn)` method holds the state lock across the full callback invocation, so 50 parallel `update('counter', lambda c: c + 1)` produces final 50 deterministically. `fn` must be synchronous and side-effect-free (it runs under the lock); for async transformations, compute outside the lock and pass a lambda that returns the already-computed value. The legacy `get` + `set` pair stays intentionally racy (over-locking would serialise every read against the write queue) — documented in the new docstring. Separately, `_set_nested` now refuses to silently overwrite a scalar intermediate: writing `output: "user.name"` when `state.user` was the string `"alice"` used to replace the string with `{"name": ...}` and lose the original value; it now raises a new `StateWriteError` naming the traversed prefix and the existing type. Auto-creation of missing intermediates is unchanged, so `set("user.name", "bob")` on an empty state still works. List intermediates with a string next-segment (e.g. `data.foo` when `state.data` is a list — write through the list with a dict-style key) now raise `StateWriteError` too, uniformly with the scalar-refusal contract; pre-0.5.0 this surfaced as a generic `TypeError` so callers that catch only the dedicated exception no longer miss the case. Regression suite at `tests/core/test_state.py::TestAtomicUpdate` (50 parallel writers + intentionally-racy `get` + `set` regression net) and `TestDottedScalarOverwriteRefused` (scalar, list intermediate, traversed-prefix message).
- Subworkflow state isolation and cross-boundary pause propagation (#057). Two related boundary defects in subworkflow semantics: (a) child engines saw a full copy of parent state by default and the child's entire final state propagated back up through the parent's `output:` key, with no way to encapsulate — `parent_secret` declared on the parent leaked into the child's prompt rendering and round-tripped back in the parent's recorded output; (b) when a subworkflow contained an `approval_gate`, the `PauseRequestedError` raised by the child engine surfaced as a generic exception at the parent boundary, the parent reported `failed` instead of `paused`, and `agentloom resume <run_id> --approve` had no resume path because no parent-level pause was checkpointed. Fixes: new opt-in `StepDefinition.isolated_state: bool = False` (default keeps the pre-0.5.0 leaky behaviour for backwards compatibility) — when `true`, the child receives a fresh `StateManager` seeded only from its own declared `state:` plus the new explicit `input: {...}` mapping; new `return_keys: list[str] | None` filter drops everything except the named keys when the child's final state propagates back. `SubworkflowStep` detects `WorkflowResult.status == PAUSED` and re-raises `PauseRequestedError` with a qualified `parent.child` step id (e.g. `sub.gate`); the engine's `_execute_step` PAUSED handler now preserves the qualified path on the `StepResult.error` field so the post-layer pass re-emits it for the parent checkpoint. The resume path strips the parent-step prefix when re-entering the subworkflow so the child engine sees `_approval.<local_id>` as expected. Regression suite at `tests/steps/test_subworkflow.py::TestSubworkflowStateIsolation` (parent-key hiding, return-keys filtering, default-leaky preserved) and `TestSubworkflowPausePropagation` (pause-at-`sub.gate` checkpoint, full resume round trip).

### Security

- Router AST validator and `DotAccessDict` runtime defence (#050). `state['__class__']`, `state['_data']`, `state['_secret']` previously bypassed the attribute-only guard in `_validate_expression` and reached `DotAccessDict.__getitem__('_x')`, which delegated to `object.__getattribute__` and returned the wrapper's raw underlying dict — leaking every user-seeded `_secret` / `_token` / `_internal` key. Two re-audit passes broadened the fix surface to: (a) `_reject_subscript` applies the same `_reject_attribute` check to `ast.Subscript` slices when the slice is an `ast.Constant` of type `str`; (b) non-constant slices — variables (`state[lookup]`), arithmetic (`state['_' + 'secret']`), conditionals, calls (`state[str(1)]`) — are refused outright; (c) integer-constant slices and `Slice` nodes with integer-constant or unary-`±` integer bounds remain accepted (`state['items'][::-1]`, `state['items'][-2:]`); (d) the router namespace no longer flattens state keys as bare names (`namespace.update(state_snapshot)` is dropped) so a state key called `len` cannot shadow the safe builtin, and arbitrary state keys are reachable only via the documented `state.X` prefix; (e) `DotAccessDict` / `DotAccessList` use name-mangled storage (`__data` → `_DotAccessDict__data`) AND override `__getattribute__` to refuse `__dict__` plus the mangled storage names so `{state._data}`, `{state.__dict__}`, and `{state._DotAccessDict__data}` no longer reach the raw underlying dict via `str.format_map`; (f) `DotAccessDict.__getattr__` no longer falls back to `object.__getattribute__` for any dynamic lookup; (g) the router runtime proxy wraps nested dict / list values in `_DictProxy` / `_ListProxy` that support both attribute and string-subscript access so the grammars the validator accepts (`state.user.name`, `state['user']['name']`, `state.items[0].label`) all resolve at runtime.
- `approval_gate.notify.url` passes through the workflow sandbox before the POST (#051). `webhooks/sender.send_webhook` accepts an optional `sandbox: ToolSandbox | None` and consults `ToolSandbox.validate_webhook_url` ahead of every delivery attempt; the approval-gate step builds the sandbox from `StepContext.sandbox_config` so the gate's notification surface is gated by the same allowlist as every other network operation. When the workflow declares `config.sandbox.enabled: true`, the destination must satisfy `allow_network`, `allowed_schemes`, and `allowed_domains`. When the sandbox is **disabled**, a built-in deny-list still blocks loopback, link-local (including cloud metadata at `169.254.169.254`), RFC 1918 / 100.64/10 CGNAT, link-local IPv6, unique-local IPv6, multicast, reserved ranges, and any non-`http`/`https` scheme. Two re-audit passes closed seven additional bypass classes: (a) IPv4-mapped IPv6 forms — `http://[::ffff:169.254.169.254]/`, `http://[::ffff:127.0.0.1]/` — normalised via `IPv6Address.ipv4_mapped` before the deny-list check; (b) the unspecified addresses `0.0.0.0` and `::`, which connect to localhost on most platforms; (c) trailing-dot hostnames (`http://127.0.0.1./`) which httpx and most resolvers accept but `ipaddress.ip_address` would otherwise reject — stripped before classification; (d) the `allow_internal_webhook_targets=true` opt-in is now split from the scheme deny, so a workflow that authorises in-cluster destinations still cannot send `file://`, `data:`, or `javascript:` webhooks; (e) percent-encoded and IDN homograph hostnames (`http://%6c%6f%63%61%6c%68%6f%73%74/`, `http://lоcalhost/`) are URL-decoded and IDNA-normalised before the literal-string check; (f) DNS resolution upgraded from `socket.gethostbyname` (single IPv4 result) to `socket.getaddrinfo` so an AAAA-only or split-horizon DNS response cannot smuggle a loopback target through the gate; (g) command-argument validation now extracts the value side of `--key=path` and `key=path` flag forms (`tee --output=/etc/passwd`, `dd of=/dev/sda`) — the prior `_looks_like_path` heuristic skipped every token starting with `-`. The host-classification path uses `ipaddress`'s stdlib flags (`is_loopback`, `is_link_local`, `is_private`, `is_reserved`, `is_unspecified`, `is_multicast`) on top of the explicit network list so reserved ranges the older containment-only check missed are caught automatically. Workflows that genuinely need to notify an in-cluster service can waive **only** the internal-host gate via `sandbox.allow_internal_webhook_targets: true` (also a new field on `SandboxConfig`); the scheme gate stays authoritative. A blocked URL is logged with the resolved hostname + reason and is emitted to the observer as `on_webhook_delivery(step_id, workflow_name, "sandbox_blocked", 0.0)`; the workflow's pause itself is unaffected because the pause and the notification are independent.
- `ToolSandbox.validate_path` wraps `ValueError` / `OSError` / `RuntimeError` / `TypeError` (#051). Null-byte paths, oversized components, symlink loops (which raise `RuntimeError` from `Path.resolve`), and non-string callers (`None` / `int` / `bytes`, which raise `TypeError`) all surface as a single `SandboxViolationError` with the original path in the message — callers that catch only the sandbox exception class no longer miss the case.
- State-value redaction policy for persisted artefacts (#052). New `agentloom.core.redact` module ships `RedactionPolicy` (glob patterns, env-var merge), `redact_state(state, policy)`, and a stable `<REDACTED:sha256=...>` sentinel; a workflow author declares per-key redaction via `state_schema:` in YAML (`state_schema: {api_key: {redact: true}, "*token*": {redact: true}}`) or deployment-wide via `AGENTLOOM_REDACT_STATE_KEYS=api_key,password,*token*`, and the engine merges the two into a single policy at construction time. After two re-audit passes the policy applies at every persistence boundary uniformly: `WorkflowEngine._save_checkpoint` redacts the runtime state snapshot, the literal `state:` block inside `workflow_definition`, every `step_results[id].output` (LLM calls that return structured payloads), and any step-level config field whose key matches the policy (`notify.headers.api_key`, `tool_args.api_key`, ...); `WorkflowResult.final_state` and `step_results` are redacted before the result crosses the process boundary so `agentloom run --json` and `result.model_dump_json()` see sentinels; the webhook sender redacts `body_template` rendering; `llm_call`'s opt-in `capture_prompts` span event is re-rendered against the redacted state. Subworkflows inherit the parent's redaction policy AND sandbox config — without inheritance a parent that locked down `api_key: {redact: true}` would have written the secret in plaintext via the child's checkpoint, and a parent's `sandbox.enabled=true` would have been bypassed by a child whose own config defaults to disabled. The in-memory state stays plaintext so a step that legitimately interpolates `{state.api_key}` against `api.openai.com` keeps working — only persisted copies are masked. The sentinel is hash-stable (sha256 of the value's string form, truncated to 16 hex chars) and redaction is idempotent: a second pass over an already-redacted value preserves it byte-for-byte so diffing across resume cycles stays consistent. `WorkflowDefinition` uses `extra="forbid"` so a typo (`stat_schema:` instead of `state_schema:`) fails at parse time instead of silently shipping the secret. The re-audit also identified three robustness gaps that are now closed: (a) circular state (a self-referential dict, or a list that contains itself) no longer triggers `RecursionError` — `_walk` tracks visited container ids and substitutes a literal `"<cycle>"` marker on the second visit; (b) non-string dict keys (int / tuple, common after JSON deserialisation) are coerced to `str` before `fnmatch` so they don't crash the entire checkpoint write; (c) `WorkflowEngine.from_checkpoint` logs an explicit warning listing the redacted keys it detects on resume — the redacted values are not magically restored, so a downstream step that references one receives the sentinel literal, and the warning surfaces this contract to the operator. Resume contract is documented: a redacted checkpoint cannot be resumed with the original secret. Lists of secrets collapse to a list of sentinels (one per element) so consumers that read shape don't break; nested dicts redact element-wise.
- Harden router expression sandbox against dunder access and type bypass (GHSA-c37m-mv4j-972v, #104)
  - Closes [GHSA-c37m-mv4j-972v](https://github.com/cchinchilla-dev/agentloom/security/advisories/GHSA-c37m-mv4j-972v): router conditions accepted arbitrary code via `type`/`__class__`/`__subclasses__()`/`__call__` chains. All three published payloads now raise `SecurityError` at parse time.
  - Reject `ast.Attribute` with `_`-prefix names; block `mro` / `format_map` / `__class__` traversal
  - Reject `ast.Name` with `_`-prefix; reject `kwargs` and starred args in `Call`
  - Drop `type` from safe-builtins (was usable as `type(x).__mro__[1].__subclasses__()`)
  - New `SecurityError` exception raised by the AST validator
  - Regression tests in `tests/steps/test_router_security.py`, including verbatim payloads from the advisory
- Harden tool sandbox against meta-executable, path, and url-scheme bypasses (#105)
  - Denylist of meta-executables (`env`, `sh`, `bash`, `python`, `python3`, `xargs`, `eval`, `exec`, ...) gated behind explicit `danger_opt_in`
  - Validate relative path arguments against the configured cwd (no `../` escapes)
  - URL schemes restricted to `http` / `https` by default; `file://`, `gopher://`, `ftp://` rejected unless listed in `allowed_schemes`
  - Shell-op regex now catches process substitution (`<(...)`, `>(...)`)
  - New `SandboxConfig` fields: `allowed_schemes` (default `["http", "https"]`), `danger_opt_in` (`list[str]`, default `[]`)
  - **Behavior change:** workflows that legitimately invoke `bash`, `python`, etc. must list each meta-executable explicitly in `danger_opt_in` — e.g. `danger_opt_in: ["bash", "python"]`. The opt-in is per-binary, not a global flag, so adding `bash` does not also enable `python`.

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

[Unreleased]: https://github.com/cchinchilla-dev/agentloom/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cchinchilla-dev/agentloom/releases/tag/v0.1.0
