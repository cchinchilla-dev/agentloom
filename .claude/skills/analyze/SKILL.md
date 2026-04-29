---
name: analyze
description: Deep analysis of current changes before committing. Goes beyond lint — catches logic bugs, architectural violations, security, and performance issues.
---

Analyze the current working tree changes for issues that static tools miss.

## Process

1. Run `git diff --stat` to see scope of changes
2. Run `git diff` to read every change in detail
3. For each changed file, analyze:

### Logic
- Are there new code paths without error handling?
- Could any input cause a crash (None, empty string, missing key)?
- Are async operations properly awaited?
- Is state mutated safely (locks, snapshots)?

### Architecture
- Does this change respect the project conventions in CLAUDE.md?
- Are new imports in the right layer? (core never imports from cli, etc.)
- Is there new functionality that needs a corresponding test?

### Security
- Are user inputs sanitized before use in expressions/shell commands?
- Could the change expose internal state or credentials?

### Performance
- Are there new O(n²) patterns or redundant async calls?
- Does it create new objects in hot paths unnecessarily?

4. Run `/check` to verify tests/lint/types pass
5. Report findings as: MUST FIX / SHOULD FIX / CONSIDER

Be direct. If the change is clean, say so in one line.
