---
name: check
description: Run the full quality gate — format, lint, tests, types. Use after changes to verify nothing is broken.
---

Run the pipeline in this order, stopping at the first real failure. Exact commands live in CLAUDE.md.

1. **Format** — run `ruff format`. Reformatted files are expected; stage them.
2. **Lint** — run `ruff check`. Auto-fix when safe (`--fix`); escalate anything that needs a design call.
3. **Tests** — `pytest -q`. On failure, identify the test and the root cause — don't just relay the traceback.
4. **Types** — `mypy`. Report offending files and lines.

Fix what you can. Only escalate when a failure needs a decision.

If everything passes, one line: `<N> passed, format+lint+types clean`.
