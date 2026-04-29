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

- **Commit messages**: Conventional-Commits with scope (`feat(providers): …`, `fix(core): …`, `chore(release): …`). Imperative, lowercase after the colon, no body unless explicitly required. Types: `feat`, `fix`, `chore`, `ci`, `test`, `docs`, `refactor`, `perf`, `build`, `style`. Common scopes: `core`, `cli`, `providers`, `observability`, `steps`, `resilience`, `tools`, `webhooks`, `checkpointing`, `record-replay`, `release`, `deps`, `version`, `meta`, `examples`, `infrastructure`. The PR title becomes the squash-merge commit on `main`, so the same convention applies to PR titles.
- **Versions**: when bumping the version, update **both** `pyproject.toml` and `CHANGELOG.md` in the same commit. The `version-linearity` CI job fails when they disagree.
- **Squash merge only.**
- **`from __future__ import annotations`** at the top of every Python module that uses type hints.
- Ruff for lint and format (config in `pyproject.toml`).
- mypy strict mode.

## Known limitations (don't flag these)

- `steps/registry.py` uses lazy imports inside `create_default_registry()` to avoid circular imports.
- Pricing table in `providers/pricing.yaml` is the source of truth — code reads it via `load_pricing()`.
- Router expressions use AST-validated `eval()` on a hardened sandbox (#104 hardening); trusted YAML, not user input.
