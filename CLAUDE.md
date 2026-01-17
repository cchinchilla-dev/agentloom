# AgentLoom - Development Guide

## Project Structure
- Source code: `src/agentloom/`
- Tests: `tests/`
- Examples: `examples/`

## Commands
- Install: `uv sync --group dev`
- Install all extras: `uv sync --group dev --all-extras`
- Run tests: `uv run pytest`
- Run tests with coverage: `uv run pytest --cov=agentloom`
- Lint: `uv run ruff check src/`
- Format: `uv run ruff format src/`
- Type check: `uv run mypy src/`
- Run CLI: `uv run agentloom <command>`
- Run a workflow: `uv run agentloom run examples/01_simple_qa.yaml`
- Validate a workflow: `uv run agentloom validate examples/01_simple_qa.yaml`

## Conventions
- All async code uses `anyio` (not raw `asyncio`) for structured concurrency
- Pydantic v2 models for all data structures
- Type hints on all public interfaces — no `Any` in public APIs
- Never import from `observability/` directly in core code — use the `WorkflowObserver` protocol
- Optional dependencies use `compat.try_import()` — never bare imports of optional modules
- Tests use `respx` for httpx mocking, `pytest-asyncio` in auto mode

## Architecture Notes
- Observability is optional via `[observability]` extra
- No Jinja2 — use `str.format_map()` for prompt templates
- No provider SDKs — httpx direct for all LLM providers
