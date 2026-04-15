---
name: release
description: Cut a release — promote Unreleased → versioned in CHANGELOG, bump version, commit. CI auto-tags and publishes from the version-bump commit.
---

Prepare a release. `$ARGUMENTS` is the bump type: `patch` (default), `minor`, `major`.

## How this repo ships

Releases go through a **pull request**, same as any other change. No direct pushes to `main`, no admin bypass — even for version bumps. This keeps the audit trail clean and matches the historical pattern (see PRs #18, #32, #36, #92).

CI does the tagging and publishing once the PR merges:

- `.github/workflows/auto-tag.yml` watches `main` for commits whose message starts with `bump version to` → reads the version from `pyproject.toml` and pushes `v<X.Y.Z>`.
- `.github/workflows/release.yml` triggers on the tag push → builds the wheel, creates a GitHub Release with auto-generated notes, publishes to PyPI via OIDC.

So the commit message prefix (`bump version to`) is load-bearing. Don't paraphrase it.

## Process

1. **Version**
   - Read current from `pyproject.toml` and `src/agentloom/__init__.py`.
   - Compute new version per semver.

2. **Pre-flight** — run `/check`. Abort on any failure. Also abort if `git status` is dirty.

3. **CHANGELOG** — Keep a Changelog convention already used in the repo:
   - An `## [Unreleased]` section accumulates `### Added` / `### Changed` / `### Fixed` / `### Removed` entries during development.
   - On release, **rename** `## [Unreleased]` to `## [<version>] - YYYY-MM-DD` (not prepend — rename in place).
   - Insert a fresh empty `## [Unreleased]` at the top for future work.
   - Do not rewrite or regroup existing entries. Do not auto-generate from `git log`; Unreleased is the source of truth.
   - Sync `docs/changelog.md` if it mirrors `CHANGELOG.md`.

4. **Bump**
   - `version` in `pyproject.toml`
   - `__version__` in `src/agentloom/__init__.py`

5. **Branch + commit** — create `release/<X.Y.Z>` from an up-to-date `main`. Commit message **must** start with `bump version to <X.Y.Z>`. Follow `/commit` rules otherwise (no scopes, no Co-Authored-By). Do **not** run `git tag` locally — the workflow creates it on merge.

6. **Open PR** — ask authorization first. Title matches the commit (`bump version to <X.Y.Z>`). Body: link the `## [X.Y.Z]` section of CHANGELOG.md and summarize highlights. Never push directly to `main`, never use admin bypass.

7. **Wait for checks + squash-merge** — all required status checks must pass. **Squash-merge only** (this repo's convention). When completing the squash merge, override the default `Merge pull request ...` title so the resulting commit on `main` starts with exactly `bump version to <X.Y.Z>` — the `auto-tag.yml` workflow greps that prefix and will skip tagging otherwise. Once that squash commit lands on `main`:
   - Auto Tag creates `v<X.Y.Z>`.
   - Release builds + publishes. Watch: `gh run list --limit 5`.

8. **Post-release** — report PR URL, tag, GitHub Release URL, and PyPI job status. If auto-tag/release fails, inspect with `/ci-status` instead of trying to recover by hand.
