---
name: review
description: Review the current changes against the project's review guidelines plus logic, security and performance, and report MUST FIX / SHOULD FIX / CONSIDER. Use before committing or opening a PR, or when asked to review a diff.
---

Review the current changes before they are committed or pushed. The argument, if any, narrows the scope (a path, `staged`, or a range such as `main...HEAD`); without one, review all uncommitted changes.

## Process

1. **Scope** — `git diff --stat` and `git diff` (`--cached` for staged only, `git diff main...HEAD` for a branch). Read every change, and the surrounding code wherever a change depends on it.

2. **Project checklist** — apply the `## Review guidelines` section of AGENTS.md, in its order. Automated PR review uses the same list, so whatever is caught here does not come back on the PR.

3. **Beyond the checklist**
   - **Logic** — new code paths without error handling; inputs that crash (None, empty string, missing key); async calls not awaited; shared state mutated without locks or snapshots.
   - **Architecture** — imports in the wrong layer; new behaviour without a test; TODO/FIXME that should be fixed now.
   - **Security** — unsanitized input reaching expressions, shell commands, paths or URLs; internal state or credentials exposed in logs, errors or traces.
   - **Performance** — O(n²) patterns, redundant async calls, objects created in hot paths.

4. **Verify** — run `make check` (check-only: it never rewrites or stages files). Any failure is a MUST FIX.

5. **Report** — findings grouped as MUST FIX / SHOULD FIX / CONSIDER, each with `file:line`, the problem, and a concrete fix. End with whether the change is ready to commit.

Be direct; no praise for its own sake. If the change is clean, say so in one line. Skip style and lint (ruff and mypy cover them) and anything AGENTS.md says not to flag.
