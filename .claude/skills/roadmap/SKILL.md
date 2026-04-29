---
name: roadmap
description: Create a release-tracking issue that organizes open issues into a phased roadmap. Pass a target version (e.g., `0.5.0`) to scope the plan.
---

Create a release-tracking issue (a "roadmap") that organizes open issues into a phased implementation plan. Matches the pattern of #133 (0.5.0 roadmap).

`$ARGUMENTS` is the target version (e.g., `0.5.0`, `1.0.0`). If missing, ask.

## When to use this

For minor or major releases with coordinated scope — multiple issues with interdependencies, parallelization decisions worth communicating, or deferred items that need to be explicitly listed so they're not forgotten. Skip for patch releases (hotfixes, small cleanups); those ship directly from the Unreleased CHANGELOG section via `/release`.

## Process

1. **Scope conversation** — ask:
   - What drives this release? (security fixes, new capabilities, both)
   - What is explicitly in-scope vs. deferred?
   - Critical path driver if any (e.g., a downstream project that needs specific features)

2. **Inventory** — run `gh issue list --state open --limit 200 --json number,title,labels`. Read bodies of non-trivial issues to understand cross-refs. Flag closed issues still referenced in open work.

3. **Phase structure** — organize issues by:
   - Criticality (security/correctness first)
   - Dependencies (foundations before features)
   - Parallelizability (independent work batched)
   - User-visible outcomes (what each phase unlocks)

   Typical shape:
   - Phase 0: bug-fix sweep, correctness
   - Phase 1: internal refactor (only if required to enable later phases)
   - Phase 2: observability / schema foundation
   - Phase 3+: feature waves
   - Last phase: testing infrastructure

4. **Draft** — follow the body format of #133:
   - `### Description` with scope drivers and version-jump justification
   - `### How to use this issue` explaining the task-list convention
   - `## Phase N — <title>` per phase with rationale, task list of `- [ ] #NNN …`, parallelization notes
   - `## Cross-phase dependency map` in ASCII
   - `## What is deliberately not in <version>` with deferred issues grouped by category
   - `## What <version> unlocks` with 3–5 concrete outcomes
   - `## Issue inventory` table with per-phase counts
   - `## Notes` — living document caveat

   GitHub renders `- [ ] #NNN` as tracked tasks; use that syntax for every child reference.

   **Formatting constraints:**
   - No emojis, no colored markers (red/yellow/green dots). Plain text only.
   - ASCII arrows (`->`, `|`, `+`) in dependency maps, not unicode arrows.
   - Title format: `ship <version>: <short description>` — brief, lowercase imperative.

5. **Confirm** — show the full draft. Iterate on phase membership, deferred items, outcomes before creating.

6. **Create** — `gh issue create --title "ship <version>: …" --body-file /tmp/roadmap.md --label "release,enhancement"`. Return the issue URL.

## Notes

- The planning issue is a living document — editable after creation as scope clarifies.
- Task-list checkboxes auto-update when child issues close; progress aggregates at the top of the issue.
- `/release` references this issue for pre-flight validation: all referenced sub-tasks must be closed (or explicitly demoted from scope) before cutting the release.
