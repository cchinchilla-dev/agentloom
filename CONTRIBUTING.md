# Contributing to AgentLoom

Thanks for considering a contribution. Here's how to get started.

## Setup

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-user>/agentloom.git
cd agentloom
git remote add upstream https://github.com/cchinchilla-dev/agentloom.git

# Install dev dependencies (requires uv)
uv sync --group dev --all-extras

# Verify
uv run pytest
```

## Development workflow

1. Sync your fork: `git fetch upstream && git rebase upstream/main`
2. Create a branch: `git checkout -b my-feature`
3. Make your changes
4. Run the quality gate:
   ```bash
   uv run pytest                    # tests
   uv run ruff check src/ tests/    # lint
   uv run ruff format src/ tests/   # format
   uv run mypy src/                 # type check
   ```
5. Push to your fork and open a pull request against `upstream/main`

## Code style

- **Python 3.11+** with type hints on all public APIs
- **Pydantic v2** for models and validation
- **anyio** for async (never raw `asyncio`)
- **httpx** for HTTP (no provider SDKs)
- `ruff` handles formatting and linting — run it before committing
- `mypy --strict` must pass

## Tests

- Use `pytest` with `pytest-asyncio` (auto mode)
- Mock HTTP calls with `respx` — no real API keys in tests
- Focus on behavior, not implementation details
- Place tests mirroring the source tree: `tests/core/`, `tests/providers/`, etc.

## Commit messages

Short, lowercase, imperative mood. No body needed for small changes.

```
add retry jitter configuration
fix router expression parsing for nested attributes
```

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Reference the issue number if applicable (`Closes #123`)
- All CI checks must pass before merge

## Adding a provider

1. Create `src/agentloom/providers/<name>.py` extending `BaseProvider`
2. Register it in `providers/gateway.py`
3. Add pricing data to `providers/pricing.py`
4. Add tests in `tests/providers/`
5. Update the README comparison table

## Adding a step type

1. Create `src/agentloom/steps/<name>.py` extending `BaseStep`
2. Register it in `steps/registry.py`
3. Add tests in `tests/steps/`

## Reporting bugs

Use the [bug report template](https://github.com/cchinchilla-dev/agentloom/issues/new?template=bug_report.yml) and include your workflow YAML if applicable.
