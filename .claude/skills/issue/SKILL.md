---
name: issue
description: Create, view, or triage GitHub issues from the terminal. Pass an issue number to read it, or a description to create one.
---

Work with GitHub issues for AgentLoom.

## If $ARGUMENTS is an issue number (e.g., "42")

1. Fetch issue details: `gh issue view $ARGUMENTS`
2. Read the issue body and comments
3. Explore the relevant code in the codebase
4. Propose an implementation plan:
   - Which files need to change
   - What the change involves
   - Estimated complexity (trivial / moderate / significant)
   - Suggested approach
5. Ask if I should start working on it

## If $ARGUMENTS is a description (e.g., "add timeout per workflow")

1. Search existing issues for duplicates: `gh issue list --search "$ARGUMENTS"`
2. If no duplicate, determine:
   - **Title**: concise, imperative mood
   - **Labels**: pick from the project's labels (bug, enhancement, core, providers, etc.)
   - **Body**: structured with Problem, Proposed Solution, and any relevant code references
3. Show the draft and ask for confirmation
4. Create: `gh issue create --title "..." --body "..." --label "..."`

## If no $ARGUMENTS

1. List open issues: `gh issue list --state open --limit 20`
2. Group by label/area
3. Suggest which ones to tackle next based on complexity and dependencies
