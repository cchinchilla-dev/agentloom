---
name: issue
description: Create, view, or triage GitHub issues from the terminal. Pass an issue number to read it, or a description to create one.
---

Work with GitHub issues for AgentLoom.

## If `$ARGUMENTS` is an issue number

1. `gh issue view $ARGUMENTS`
2. Read body + comments, explore referenced code paths.
3. Propose an implementation plan:
   - Files to touch
   - Mechanism (1–2 paragraphs)
   - Complexity: trivial / moderate / significant
   - Open questions / decisions needed
4. Ask before starting work.

## If `$ARGUMENTS` is a description

1. Dedup check: `gh issue list --search "$ARGUMENTS" --state all`.
2. Draft following the repo's issue convention:

   **Title** — brief lowercase imperative (`add X`, `expose Y`, `fix Z`). No scopes.

   **Body** — this exact structure:
   ```
   ### Description

   <what's missing or broken, why it matters, related issues with #NN refs>

   ### Proposal

   <concrete approach; include code/YAML blocks where they clarify the API>

   <optional: trade-offs, alternatives, notes on scope>
   ```
   Bugs may swap `### Proposal` for `### Repro` + `### Expected vs actual`, but keep the `###` depth.

3. **Labels** — reuse existing ones; don't invent new. Pick from the label list shown by `gh label list`. Common picks: `enhancement`, `bug`, `core`, `providers`, `cli`, `observability`, `infrastructure`.
4. Show the draft, confirm, then `gh issue create --title "..." --body "..." --label "..."`.

## If no `$ARGUMENTS`

1. `gh issue list --state open --limit 20`
2. Group by label/area.
3. Suggest next picks by complexity + dependency graph (note blockers like "depends on #NN").
