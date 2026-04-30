# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

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
