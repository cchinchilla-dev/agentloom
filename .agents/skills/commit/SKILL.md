---
name: commit
description: Stage and commit changes with a Conventional Commits message, proposing message and files for approval first. Use when the user asks to commit; never pushes.
---

Create a commit for the current changes.

## Rules (non-negotiable)

- **Propose first**: show the draft message and the exact files to stage, then wait for approval.
- **Message**: Conventional Commits with scope, as in AGENTS.md (`fix(core): …`): imperative, lowercase after the colon, single line. Check `git log --oneline -20` for the scopes in use. Release commits are exactly `bump version to <X.Y.Z>`, no type or scope (see the `release` skill).
- **Stage explicitly**: `git add <paths>`, never `git add -A` / `git add .`. Skip untracked files unrelated to the change unless the user asked for them.
- **Format before commit**: run `ruff format` (and fix `ruff check`) before staging. See AGENTS.md for commands.
- **Natural increments**: if the diff spans unrelated concerns, propose multiple commits, not one big bang.
- **Never push** until the user says so explicitly.

## Amending

- Only amend when the user explicitly asks. Amending a pushed commit requires `git push --force-with-lease`, and pushing still needs explicit authorization.
