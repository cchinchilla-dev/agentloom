# AgentLoom

## Build & test
- `uv sync --group dev` — install (add `--all-extras` for observability)
- `uv run pytest` — tests (187, ~5s)
- `uv run ruff check src/ tests/` — lint
- `uv run ruff format src/ tests/` — format
- `uv run mypy src/` — strict type check
- `uv run agentloom run examples/01_simple_qa.yaml` — run a workflow
- `uv run agentloom validate examples/03_router_workflow.yaml` — validate YAML

## Rules

**Async**: always `anyio`, never raw `asyncio`. Task groups for parallelism.

**Models**: Pydantic v2 everywhere. No `Any` in public signatures.

**Providers**: httpx direct — no SDKs. Each provider is a thin adapter in `providers/`.

**Templates**: `str.format_map()` with `SafeFormatDict`. No Jinja2.

**Observability**: optional via `[observability]` extra. Core code never imports from `observability/` directly — use `compat.try_import()` so it degrades to noops.

**Tests**: `respx` for HTTP mocking, `pytest-asyncio` auto mode. Mocks over real API calls.

## Architecture (read these first)
- `core/engine.py` — workflow executor, layer-based parallel DAG traversal
- `core/dag.py` — dependency graph, topological sort, cycle detection
- `core/state.py` — async-safe shared state with dotted key access
- `providers/gateway.py` — routes to providers with circuit breaker + rate limiter + fallback
- `steps/router.py` — conditional branching via AST-validated safe expressions

## Gotchas
- `steps/registry.py` uses lazy imports inside `create_default_registry()` — don't move them to module level
- `steps/llm_call.py` uses sync `state.state` in async context (marked FIXME, don't make it worse)
- Gateway `register()` accepts `**kwargs` to stay compatible with the simple version — don't remove
- Pricing table in `providers/pricing.py` is hardcoded — there's a TODO to load from YAML
