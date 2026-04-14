---
name: commit
description: Stage and commit changes following the project's observed flow. Propose message + files for approval first, never push automatically.
---

Create a commit for the current changes.

## Rules (non-negotiable)

- **Propose first**: show the draft message and the exact files to stage, then wait for approval.
- **Message**: brief, lowercase, imperative ("add X", "fix Y", "lift Z"), matching `git log --oneline -20` style. No scopes like `feat:`/`fix:` unless the existing log uses them.
- **Stage explicitly**: `git add <paths>`, never `git add -A` / `git add .`. Skip `scripts/audit_*.py` and other untracked unrelated files unless the user asked for them.
- **Format before commit**: run `ruff format` (and fix `ruff check`) before staging. See CLAUDE.md for commands.
- **Natural increments**: if the diff spans unrelated concerns, propose multiple commits, not one big bang.
- **Never push** until the user says so explicitly.

## Amending

- Only amend when the user explicitly asks. Amending a pushed commit requires `git push --force-with-lease`, and pushing still needs explicit authorization.
