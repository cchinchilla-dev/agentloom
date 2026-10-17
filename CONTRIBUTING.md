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

# Install the git hooks (ruff and version-linearity on every commit)
uv run pre-commit install

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

Conventional-Commits style with scope:

```
feat(providers): add streaming usage extraction for Anthropic
fix(steps): respect router skip closure across parallel branches
chore(deps): bump httpx to 0.28.1
ci(version): add version-linearity gate
docs(observability): document OTel span schema migration
refactor(gateway): extract candidate cache into LRU helper
test(providers): add 429 contract suite parametrised across adapters
perf(state): swap dict-snapshot for copy-on-write read view
build(deps): pin httpx to 0.28.x for connection-pool compatibility
style(core): apply ruff format
```

- Imperative mood, lowercase after the colon. Single-line subject under
  ~72 chars; body only when justification adds value.
- Types: `feat`, `fix`, `chore`, `ci`, `test`, `docs`, `refactor`,
  `perf`, `build`, `style`.
- Common scopes: `core`, `cli`, `providers`, `observability`, `steps`,
  `resilience`, `tools`, `webhooks`, `checkpointing`, `record-replay`,
  `release`, `deps`, `version`, `meta`, `examples`, `infrastructure`.
- Several scopes go comma-separated, without spaces: `feat(core,steps): …`.
- PR-merge commits append `(#NN)` and may include `(#issue)` references —
  e.g. `feat(providers): add structured output (#117) (#42)`.
- Release commits are the one exception: exactly `bump version to X.Y.Z`,
  with no type or scope. `auto-tag.yml` only tags commits that start with
  that prefix.

The PR title becomes the squash-merge commit on `main`, so the same
convention applies to PR titles. The `pr` skill enforces this.

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Link issues: `Closes #123` in the PR description
- All CI checks (tests, ruff, mypy) must pass before merge
- PRs are squash-merged to keep history clean

## AI-assisted contributions

AI coding assistants are welcome. The project rules they need live in
[`AGENTS.md`](https://github.com/cchinchilla-dev/agentloom/blob/main/AGENTS.md), which most assistants read automatically.

The repository ships a small set of shared skills in [`.agents/skills/`](https://github.com/cchinchilla-dev/agentloom/tree/main/.agents/skills)
for routine work on this repo: `check` (the local quality gate), `review`,
`commit`, `pr`, `release`, and a few more. Codex reads them from that folder,
Claude Code through the `.claude/skills` symlink, and any other assistant can
follow the `SKILL.md` files as plain instructions. They are optional: CI does
not depend on them.

Whatever tool wrote the code, the author of the pull request is responsible for
it: run the quality gate, read the diff, and be ready to explain any line.

## Review

- **During development** — the deep review happens before a pull request is
  opened. Review your own changes with an AI assistant, by hand, or both; a model
  from a different vendor than the one that wrote the code tends to catch more.
  The checklist is the *Review guidelines* section of `AGENTS.md`.
- **On the pull request** — CI must pass. [CodeRabbit](https://www.coderabbit.ai)
  posts a light automated review focused on bugs, security and the rules in
  `AGENTS.md`. Its comments are suggestions; a maintainer decides what to merge.

## Adding a provider

1. Create `src/agentloom/providers/<name>.py` extending `BaseProvider`
2. Register it in `_PROVIDER_CLASSES` (`cli/run.py`) and give it defaults in `_PROVIDER_DEFAULTS` (`config.py`)
3. Add pricing data to `providers/pricing.yaml`
4. Add tests in `tests/providers/`
5. Update the capability matrix in `docs/providers.md` and the providers table in the README

## Adding a step type

1. Create `src/agentloom/steps/<name>.py` extending `BaseStep`
2. Register it in `steps/registry.py`
3. Add tests in `tests/steps/`

## Reporting bugs

Use the [bug report template](https://github.com/cchinchilla-dev/agentloom/issues/new?template=bug_report.yml) and include your workflow YAML if applicable.
