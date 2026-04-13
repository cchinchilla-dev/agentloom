# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Approval gate step type** — human-in-the-loop decision point (#41)
    - `StepType.APPROVAL_GATE` pauses the workflow and waits for human approval or rejection
    - Decision injected via `_approval.<step_id>` state key on resume
    - `--approve` / `--reject` flags on `agentloom resume`
    - `timeout_seconds` and `on_timeout` schema fields (consumed by webhook callback server in #42)
    - Example workflow (29), validation script, and K8s smoke job
- **Workflow pause mechanism** — foundation for human-in-the-loop (#40)
    - `PauseRequestedError` exception for step executors to signal a pause
    - `StepStatus.PAUSED` and `WorkflowStatus.PAUSED` status values
    - Engine catches pause requests, saves checkpoint with `status=paused` and `paused_step_id`, and returns cleanly
    - Resume from paused checkpoint skips completed steps and re-runs the paused step
    - CLI treats paused workflows as non-error (exit code 0)
    - Functional validation script and K8s smoke job
- **Pluggable checkpoint backends** with `BaseCheckpointer` protocol and `FileCheckpointer` default (#78)
    - `CheckpointData` model with full workflow state serialization
    - Engine integration: auto `run_id`, checkpoint on completion/failure, graceful I/O error handling
    - `WorkflowEngine.from_checkpoint()` to reconstruct and resume, skipping completed steps
    - `agentloom run --checkpoint` and `--checkpoint-dir` flags
    - `agentloom resume <run_id>` and `agentloom runs` CLI commands
    - Example workflow (28) and documentation

## [0.3.0] — 2026-04-12

### Added

- **Documentation site** with mkdocs-material — full reference docs auto-deployed to GitHub Pages
- **Multi-modal input** for `llm_call` steps — images, PDFs, and audio via `attachments` field
    - Provider-native formatting: OpenAI (images, audio), Anthropic (images, PDFs), Google (images, PDFs, audio), Ollama (images)
    - URL fetching with `fetch: local` (default) or `fetch: provider` passthrough
    - SSRF protection: blocks private/reserved IP ranges (RFC 1918, loopback, link-local)
    - Sandbox integration: `allowed_domains`, `allow_network`, and `readable_paths` enforced
    - Grafana dashboard "Multi-modal" row with attachment panels
    - Multi-modal workflow examples (19-24)
- **Streaming** for LLM responses with real-time token output
    - `StreamResponse` accumulator with per-provider SSE/NDJSON parsing
    - All 4 providers: OpenAI (SSE), Anthropic (SSE), Google (SSE), Ollama (NDJSON)
    - Gateway `stream()` with circuit breaker + rate limiter integration
    - `config.stream: true` (workflow-level) and per-step `stream:` override
    - CLI `--stream` flag, `time_to_first_token_ms` in `StepResult`
    - Grafana "Streaming" dashboard row with TTFT quantiles
    - Streaming examples (25-26)
- **`AGENTLOOM_*` env var prefix** for all configuration overrides
- **YAML-based pricing table** replacing hardcoded Python dict
- **Provider auto-discovery** moved from CLI hack to `config.discover_providers()`
- **Ollama e2e integration tests** against a live Docker instance
- **Array index support** in state paths (`state.items[0]`, `results[-1]`)
- **First-class graph API** for workflow DAG analysis and export
    - `WorkflowGraph` class with path algorithms and export formats
    - Graphviz DOT, Mermaid, PNML, NetworkX, JSON
- **Test coverage reporting** via Codecov with 85% minimum threshold

## [0.2.0] — 2026-03-30

### Added

- Kubernetes manifests with Kustomize overlays for dev, staging, and production
- Helm chart with Job/CronJob modes and render-time input validation
- Terraform configuration for local kind cluster with full observability stack
- ArgoCD Application CRD with automated sync and Job immutability handling
- Docker CI/CD workflow for multi-arch GHCR publishing
- Infrastructure documentation

### Fixed

- Production NetworkPolicy OTel egress restricted to observability namespace
- Read-only filesystem audit check no longer false-passes
- Terraform audit phase passes KUBECONFIG to all kubectl poll commands
- GitHub Actions and image versions pinned to commit SHAs

## [0.1.2] — 2026-03-26

### Added

- **Sandbox enforcement** for built-in tools — command allowlist, path restrictions, network domain filtering, shell operator injection prevention, write size limits
- `SandboxConfig` model in workflow YAML (`config.sandbox.*`)
- Sandbox workflow examples (17, 18)

### Fixed

- Step executors now use `await get_state_snapshot()` instead of sync `.state` access
- Removed deprecated `gemini-2.0-flash` model

## [0.1.1] — 2026-03-22

### Fixed

- Rate limiter now accounts for response tokens, not just prompt tokens
- README header image uses absolute URLs for PyPI compatibility

## [0.1.0] — 2026-03-19

First public release.

### Added

- YAML and Python DSL workflow definitions (DAGs with sequential + parallel steps)
- Step types: `llm_call`, `tool`, `router` (conditional), `subworkflow`
- Provider gateway with automatic fallback (OpenAI, Anthropic, Google, Ollama)
- Circuit breaker, rate limiter, and retry with exponential backoff per provider
- Budget enforcement (hard stop when USD limit exceeded)
- Cost tracking per step, model, and provider
- OpenTelemetry traces + Prometheus metrics (optional)
- CLI commands: `run`, `validate`, `visualize` (ASCII + Mermaid), `info`
- Checkpointing: save and resume workflow state to disk

### Design Decisions

- **httpx over provider SDKs** — keeps dependencies minimal (~5 core)
- **anyio over raw asyncio** — structured concurrency via task groups
- **str.format_map over Jinja2** — one fewer dependency; prompt templates don't need loops
- **Observability optional** — core runs without opentelemetry or prometheus
- **Pydantic v2** — validation and serialization worth the compilation trade-off
