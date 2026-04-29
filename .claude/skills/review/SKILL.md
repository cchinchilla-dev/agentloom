---
name: review
description: Review staged or unstaged changes before committing. Catches bugs, style drift, and missing coverage.
---

Review the current changes in the working tree:

1. Run `git diff --stat` and `git diff` to see what changed
2. For each modified file, check:
   - Does the change introduce bugs or break existing behavior?
   - Are there missing error cases or edge cases?
   - Is the change consistent with the patterns in CLAUDE.md?
   - Are there any TODOs or FIXMEs that should be addressed now?
   - If it's new functionality, are there tests?
3. Run `uv run pytest --tb=short -q` to verify tests pass
4. Summarize: what's good, what needs attention, and whether it's ready to commit

Be direct. Don't praise for the sake of it.
