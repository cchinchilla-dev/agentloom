---
name: pr
description: Create a pull request from the current branch. Auto-generates description, suggests labels, and runs quality checks first.
---

Create a pull request for the current branch.

## Process

1. **Pre-flight**
   - Run `uv run pytest --tb=short -q` — warn if tests fail
   - Run `uv run ruff check src/ tests/` — auto-fix if needed
   - Run `uv run mypy src/` — warn if types fail

2. **Gather context**
   - `git log main..HEAD --oneline` — list commits on this branch
   - `git diff main...HEAD --stat` — list changed files
   - `git diff main...HEAD` — read the actual diff

3. **Auto-generate PR content**
   - **Title**: derive from branch name or first commit (imperative mood, <70 chars)
   - **Body**: fill in the PR template format:
     - What: summarize the changes from the diff
     - Why: infer from commit messages or ask
     - Testing: auto-check the boxes if tests/lint/types pass
   - **Labels**: suggest based on changed file paths (use .github/labeler.yml mapping)

4. **Show draft** and ask for confirmation/edits

5. **Create PR**
   - Push branch if needed: `git push -u origin HEAD`
   - `gh pr create --title "..." --body "..." --label "..."`
   - If $ARGUMENTS contains "draft", add `--draft`

6. **Report**: PR URL and CI status link
