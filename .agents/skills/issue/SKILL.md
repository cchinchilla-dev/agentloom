---
name: issue
description: View, create or triage GitHub issues. Use when the user mentions an issue number, wants to report a bug or request a feature, or asks what to work on next.
---

Work with GitHub issues for AgentLoom.

## If the argument is an issue number

1. `gh issue view <number>`
2. Read body + comments, explore referenced code paths.
3. Propose an implementation plan:
   - Files to touch
   - Mechanism (1–2 paragraphs)
   - Complexity: trivial / moderate / significant
   - Open questions / decisions needed
4. Ask before starting work.

## If the argument is a description

1. Dedup check: `gh issue list --search "<keywords>" --state all`.
2. Draft following the repo's issue convention:

   **Title** — brief lowercase imperative (`add X`, `expose Y`, `fix Z`). No scopes.

   **Body** — this structure (drop a section only when it would be empty):
   ```
   ### Description

   <what's missing or broken, why it matters, related issues with #NN refs>

   ### Proposal

   <concrete approach; include code/YAML blocks where they clarify the API>

   ### Scope

   <files or modules to touch, one bullet each with what changes>

   ### Regression tests

   <test names that pin the behaviour>

   ### Notes

   <trade-offs, decisions, `Depends on: #NN` / `Blocks: #NN`>
   ```
   Bugs may swap `### Proposal` for `### Repro` + `### Expected vs actual`, but keep the `###` depth.

3. **Labels** — reuse existing ones; don't invent new. Pick from the label list shown by `gh label list`. Common picks: `enhancement`, `bug`, `core`, `providers`, `cli`, `observability`, `infrastructure`.
4. Show the draft, confirm, then `gh issue create --title "..." --body "..." --label "..."`.

## If there is no argument

1. `gh issue list --state open --limit 20`
2. Group by label/area.
3. Suggest next picks by complexity + dependency graph (note blockers like "depends on #NN").
