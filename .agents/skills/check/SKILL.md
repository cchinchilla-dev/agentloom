---
name: check
description: Run the local quality gate (format, lint, types, tests) with the same checks as CI. Use after any code change and before committing or opening a PR.
---

Run the gate in this order, stopping at the first real failure. Each step mirrors a job in `.github/workflows/ci.yml`; exact commands also live in AGENTS.md.

1. **Format** — `uv run ruff format src/ tests/`. Reformatted files are expected; list them and leave staging to the user (or the `commit` skill).
2. **Lint** — `uv run ruff check src/ tests/`. Auto-fix when safe (`--fix`); escalate anything that needs a design call.
3. **Types** — `uv run mypy src/`. Report offending files and lines.
4. **Tests** — `uv run pytest -q --cov=agentloom`. CI fails below the 85% `fail_under` in `pyproject.toml`; report the total. CI installs every extra (`uv sync --group dev --all-extras`); sync the same way first if extras are missing, or optional paths go untested. On failure, identify the test and the root cause — don't just relay the traceback.
5. **Version linearity** — `uv run python scripts/check_version_linearity.py` (`pyproject.toml` and `CHANGELOG.md` must agree).
6. **K8s manifests** — only if the change touches `deploy/`. With `kustomize`, `kubeconform` and `helm` on `PATH`, run the commands of the `manifests` job in `ci.yml`; otherwise say it was skipped and CI will cover it.

`make check` runs the same gates non-interactively (check-only, no auto-fix). Fix what you can. Only escalate when a failure needs a decision. Codecov patch coverage only runs in CI; use the `coverage` skill if it fails there.

If everything passes, one line: `<N> passed, coverage <X>%, format+lint+types+version clean` (plus `+manifests` when step 6 ran).
