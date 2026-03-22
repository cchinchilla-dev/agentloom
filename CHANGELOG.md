# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Streaming responses from providers (currently falls back to full completion)
- YAML-based provider config (replace env var discovery)
- Load pricing table from YAML instead of hardcoded Python dict
- Sandbox shell_command tool execution (currently no isolation)
- Array index support in state paths (e.g., `state.items[0]`)

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
- Rate limiter doesn't account for response tokens (only prompt tokens)
- Provider discovery from env vars only, should be a config file
- Shell command tool has no sandboxing (FIXME in code)
- File tools accept arbitrary paths (no path sanitization)
- Router expressions use `eval()` — must be trusted input (not user-facing)
- Pricing table hardcoded in Python, should be YAML config
- No array index support in state paths (e.g., `state.items[0]`)
- Sync state access in step executors bypasses async lock (FIXME in `llm_call.py`)
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

[Unreleased]: https://github.com/cchinchilla-dev/agentloom/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cchinchilla-dev/agentloom/releases/tag/v0.1.0
