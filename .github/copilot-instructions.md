# Copilot Code Review Instructions

## Project

AgentLoom is a deterministic LLM workflow orchestrator. Python 3.11+, async (anyio), Pydantic v2.

## Review priorities

1. **Async safety** — all async code must use `anyio`, never raw `asyncio`. Flag any `asyncio.` import.
2. **Type annotations** — no `Any` in public signatures. Pydantic models for all data structures.
3. **Provider isolation** — providers use `httpx` directly, no vendor SDKs. Each is a thin adapter.
4. **Observability decoupling** — core code must never import from `observability/` directly. Use `compat.try_import()` so it degrades to noops without the extra.
5. **Template safety** — string templates use `str.format_map()` with `SafeFormatDict`. No Jinja2, no f-strings with user input.
6. **Test quality** — HTTP mocking via `respx`, `pytest-asyncio` auto mode. No real API calls in tests.

## Style

- Commit messages: short imperative phrase, no body, lowercase.
- Squash merge only.
- Ruff for lint and format (config in `pyproject.toml`).
- mypy strict mode.

## Known limitations (don't flag these)

- `steps/llm_call.py` uses sync `state.state` in async context (FIXME exists, fixing breaks step interface).
- `steps/registry.py` uses lazy imports inside `create_default_registry()` to avoid circular imports.
- Built-in shell/file tools have no sandboxing (documented in CHANGELOG as known limitation).
- Router expressions use `eval()` on AST-validated input (trusted YAML from developer, not user input).
