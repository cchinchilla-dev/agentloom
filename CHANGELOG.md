# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Ollama e2e integration tests against a live Docker instance (5 smoke tests) (#71)
- CI workflow `e2e-ollama.yml` — weekly schedule, `release/**` branches, `e2e` label on PRs, manual dispatch

### Planned

- Streaming responses from providers (currently falls back to full completion)
- YAML-based provider config (replace env var discovery)
- Load pricing table from YAML instead of hardcoded Python dict
- Array index support in state paths (e.g., `state.items[0]`)

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

- No streaming support (falls back to full completion)
- Router expressions use first-match-wins, no priority ordering
- ~~Rate limiter doesn't account for response tokens (only prompt tokens)~~ (fixed in 0.1.1)
- Provider discovery from env vars only, should be a config file
- ~~Shell command tool has no sandboxing (FIXME in code)~~ (fixed in 0.1.2)
- ~~File tools accept arbitrary paths (no path sanitization)~~ (fixed in 0.1.2)
- Router expressions use `eval()` — must be trusted input (not user-facing)
- Pricing table hardcoded in Python, should be YAML config
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

[Unreleased]: https://github.com/cchinchilla-dev/agentloom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cchinchilla-dev/agentloom/releases/tag/v0.1.0
