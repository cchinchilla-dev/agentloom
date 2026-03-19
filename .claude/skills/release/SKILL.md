---
name: release
description: Prepare and publish a new release — bump version, update changelog, tag, and push. Use when ready to ship.
---

Prepare a release. $ARGUMENTS should be the version bump type: `patch`, `minor`, or `major` (default: patch).

## Process

1. **Determine new version**
   - Read current version from `src/agentloom/__init__.py`
   - Compute new version based on $ARGUMENTS (patch: 0.1.0→0.1.1, minor: 0.1.0→0.2.0, major: 0.1.0→1.0.0)

2. **Pre-flight checks**
   - Run `uv run pytest --tb=short -q` — abort if tests fail
   - Run `uv run ruff check src/` — abort if lint fails
   - Run `uv run mypy src/` — abort if types fail
   - Check `git status` — abort if there are uncommitted changes

3. **Generate changelog entry**
   - Get previous tag: `git describe --tags --abbrev=0 2>/dev/null`
   - List commits since last tag: `git log --oneline <prev_tag>..HEAD`
   - Group by type (fix:, feat:, etc.) if conventional commits are used
   - Prepend new section to CHANGELOG.md under `## [<version>] - <today>`

4. **Bump version**
   - Update `__version__` in `src/agentloom/__init__.py`
   - Update `version` in `pyproject.toml`

5. **Commit, tag, push**
   - `git add pyproject.toml src/agentloom/__init__.py CHANGELOG.md`
   - `git commit -m "release: v<version>"`
   - `git tag v<version>`
   - Ask for confirmation before: `git push && git push --tags`

6. **Post-release**
   - Check CI status: `gh run list --limit 1`
   - Report: version, tag, and expected PyPI publish time
