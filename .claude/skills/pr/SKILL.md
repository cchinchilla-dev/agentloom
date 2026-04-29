---
name: pr
description: Create a pull request from the current branch, matching the repo's PR conventions (What/Why/Testing for features, Summary/Test-plan for small PRs).
---

Open a PR for the current branch.

## Process

1. **Pre-flight**
   - Run `/check`. Warn (don't block); the user may want a draft anyway.
   - Verify the lockfile is in sync: `uv lock --check`. If it fails, the lock drifted from `pyproject.toml` (common after a `version` bump or a dependency edit without re-locking). Run `uv lock` to refresh, stage `uv.lock`, and include it — as its own commit when it's housekeeping, folded into the feature commit when it's a direct consequence of a dependency change in the same branch. A stale lock causes CI reproducibility failures downstream, so do not skip.

2. **Gather context**
   - `git log main..HEAD --oneline`
   - `git diff main...HEAD --stat`
   - `git diff main...HEAD` — read the actual diff
   - Look at the last 3–5 merged PRs for title/label conventions: `gh pr list --state merged --limit 5 --json number,title,labels`

3. **Title** — Conventional Commits with scope (matches the commit-message convention enforced in `CLAUDE.md` / `CONTRIBUTING.md`):

   ```
   <type>(<scope>): <imperative lowercase description>
   ```

   - Types: `feat`, `fix`, `chore`, `ci`, `test`, `docs`, `refactor`, `perf`, `build`, `style`.
   - Common scopes: `core`, `cli`, `providers`, `observability`, `steps`, `resilience`, `tools`, `webhooks`, `checkpointing`, `record-replay`, `release`, `deps`, `version`, `meta`, `examples`, `infrastructure`. Pick the most representative scope of the PR — for cross-cutting work, `meta` (docs/skills) or the dominant area.
   - The PR title becomes the squash-merge commit on `main`, so consistency with the commit-message style matters.
   - If the PR closes an issue, append `(#<issue>)` — e.g. `feat(webhooks): add approval-gate notifications (#42)`.
   - Under 72 chars.

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

5. **Labels** — `pr-labeler` auto-applies labels from `.github/labeler.yml` based on changed paths (`core`, `cli`, `providers`, `observability`, `resilience`, `infrastructure`, `ci`, `dependencies`, `release`). Add general categories (`enhancement`, `bug`, `breaking`, `e2e`) explicitly via `--label` because the labeler does not infer them from paths.

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
