---
name: pr
description: Create a pull request from the current branch, matching the repo's PR conventions (What/Why/Testing for features, Summary/Test-plan for small PRs).
---

Open a PR for the current branch.

## Process

1. **Pre-flight** — run `/check`. Warn (don't block); the user may want a draft anyway.

2. **Gather context**
   - `git log main..HEAD --oneline`
   - `git diff main...HEAD --stat`
   - `git diff main...HEAD` — read the actual diff
   - Look at the last 3–5 merged PRs for title/label conventions: `gh pr list --state merged --limit 5 --json number,title,labels`

3. **Title**
   - Brief lowercase imperative. No `feat:`/`fix:`/`chore:` scopes unless the existing log already uses them.
   - If the PR closes an issue, append `(#<issue>)` — e.g. `add webhook notifications for approval gates (#42)`.
   - <70 chars.

4. **Body** — pick the format by PR size:

   **Feature / substantive PRs** (new functionality, non-trivial refactor):
   ```
   ## What

   <1–2 sentence summary>

   - <bullet describing change 1>
   - <bullet describing change 2>

   ## Why

   <motivation, link to related issues/PRs>

   Closes #<issue>

   ## Testing

   - [x] `uv run pytest` passes (<N> tests)
   - [x] `uv run ruff check src/ tests/` clean
   - [x] `uv run mypy src/` clean
   - [x] <any manual / docker / k8s validation actually run>

   ## Notes

   <caveats, follow-ups, implementation choices worth calling out>
   ```

   **Small PRs** (docs, single-file fixes, coverage bumps):
   ```
   ## Summary

   - <1–3 behavior-focused bullets>

   ## Test plan

   - [x] <auto-check if /check passed, else manual>
   ```

   No emojis.

5. **Labels** — suggest based on changed paths; mirror labels from recent similar PRs. Common: `core`, `cli`, `providers`, `observability`, `infrastructure`, `e2e`, `enhancement`.

6. **Push** (ask authorization first)
   - New branch: `git push --set-upstream origin <branch>`
   - Existing: `git push`

7. **Create**
   ```
   gh pr create --title "..." --body "$(cat <<'EOF'
   ...
   EOF
   )" --label "..."
   ```
   Add `--draft` if `$ARGUMENTS` contains "draft".

8. **Report** the PR URL. Don't poll CI — let the user run `/ci-status`.
