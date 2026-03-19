---
name: check
description: Run the full quality gate — tests, lint, type check. Use after making changes to verify nothing is broken.
---

Run the complete quality pipeline for AgentLoom. Stop at the first failure.

1. `uv run pytest --tb=short -q` — if tests fail, report which ones and why
2. `uv run ruff check src/ tests/` — if lint fails, fix with `ruff check --fix` and report what changed
3. `uv run mypy src/` — if types fail, report the errors

If everything passes, say so in one line. If something fails, focus on fixing it — don't just report.
