# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Wire up observer hooks to step execution
- Streaming support for providers
- Config file for provider setup (instead of env vars)
- Dynamic pricing from YAML
- Proper sandboxing for shell tool

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
- 187 tests, mypy strict, ruff clean

### Known Limitations

- No streaming support (falls back to full completion)
- Router expressions use first-match-wins, no priority ordering
- Rate limiter doesn't account for response tokens (only prompt tokens)
- Provider discovery from env vars only, should be a config file
- `observer` parameter on WorkflowEngine accepted but not wired up
- Shell command tool has no sandboxing
- Pricing table hardcoded in Python, should be YAML config
- No array index support in state paths (e.g., `state.items[0]`)

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
