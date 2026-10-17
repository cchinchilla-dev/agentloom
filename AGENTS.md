# AgentLoom

Project rules for contributors and AI coding assistants. Any assistant that reads `AGENTS.md` (Claude Code, Codex, Copilot, Cursor, …) picks these up; Claude Code reads them through `CLAUDE.md`.

## Build & test
- `uv sync --group dev` — install (add `--all-extras` for observability)
- `make check` — every CI gate in one go, check-only (no auto-fix)
- `uv run pytest` — tests (~5s)
- `uv run ruff check src/ tests/` — lint
- `uv run ruff format src/ tests/` — format
- `uv run mypy src/` — strict type check
- `uv run agentloom run examples/01_simple_qa.yaml` — run a workflow
- `uv run agentloom replay workflow.yaml --recording rec.json` — replay from recorded responses (offline, no API key)
- `uv run agentloom validate examples/03_router_workflow.yaml` — validate YAML

## Rules

**Async**: always `anyio`, never raw `asyncio`. Task groups for parallelism.

**Models**: Pydantic v2 everywhere. No `Any` in public signatures. `from __future__ import annotations` at the top of every module that uses type hints.

**Providers**: httpx direct — no SDKs. Each provider is a thin adapter in `providers/`.

**Templates**: `str.format_map()` with `SafeFormatDict`. No Jinja2, no f-strings with user input.

**Observability**: optional via `[observability]` extra. Code that needs the extra (OpenTelemetry, Prometheus) is reached only through `compat.try_import()`, so core degrades to noops. The dependency-free modules (`observability/schema.py`, `observability/quality.py`) may be imported directly; the engine does so to attach the quality emitter.

**Tests**: `respx` for HTTP mocking, `pytest-asyncio` auto mode. Mocks over real API calls.

**Commits**: Conventional-Commits with scope (`feat(providers): …`, `fix(core): …`, `docs(core): …`). Imperative, lowercase after the colon, single-line subject. Types and scopes documented in `CONTRIBUTING.md`. The PR title becomes the squash-merge commit on `main`, so the same rule applies to PR titles. Release commits are the one exception: exactly `bump version to X.Y.Z`, with no type or scope, because `auto-tag.yml` only tags commits that start with that prefix. Squash merge only.

**Versions**: when bumping the version, update `pyproject.toml`, `CHANGELOG.md` and `__version__` in `src/agentloom/__init__.py` in the same commit. The `version-linearity` CI job fails when they disagree.

## Architecture (read these first)
- `core/engine.py` — workflow executor, layer-based parallel DAG traversal
- `core/dag.py` — dependency graph, topological sort, cycle detection
- `core/state.py` — async-safe shared state with dotted key access
- `providers/gateway.py` — routes to providers with circuit breaker + rate limiter + fallback
- `steps/router.py` — conditional branching via AST-validated safe expressions
- `tools/sandbox.py` — allowlist-based sandbox for commands, paths, and network access

## Gotchas
- `steps/registry.py` uses lazy imports inside `create_default_registry()` — avoids circular imports between steps and providers at module load time
- `providers/pricing.yaml` is the source of truth for model prices — code reads it via `load_pricing()`
- Router expressions use AST-validated `eval()` on a hardened sandbox (#104); workflow YAML is trusted input, not user input

## Review guidelines

Reviews (local assistant passes during development, and automated PR review) check, in order:

1. **Async safety** — any `asyncio.` import or raw asyncio primitive.
2. **Types** — `Any` in public signatures; data structures that are not Pydantic models.
3. **Provider isolation** — vendor SDK imports; providers must stay thin httpx adapters.
4. **Observability decoupling** — core modules importing code that needs the `[observability]` extra without `compat.try_import()` (the dependency-free `schema` and `quality` modules are fine).
5. **Template safety** — Jinja2 or f-strings over user input instead of `SafeFormatDict`.
6. **Tests** — real network calls, sleeps used for synchronisation, assertions that cannot fail.
7. **Public API changes** without a `CHANGELOG.md` entry; version bumps that skip one of `pyproject.toml` / `CHANGELOG.md` / `__version__`.

Do not flag style or lint issues (ruff and `mypy --strict` run in CI), nor the items listed under Gotchas.

## Skills

Reusable task prompts (check, review, commit, pr, release, …) live in `.agents/skills/`. Claude Code loads them through the `.claude/skills` symlink; other assistants can read the `SKILL.md` files directly.
