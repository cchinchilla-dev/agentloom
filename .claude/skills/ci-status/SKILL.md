---
name: ci-status
description: Check GitHub Actions CI status. Shows recent runs, and if there are failures, reads the logs and suggests fixes.
---

Check CI status for AgentLoom.

## Process

1. **List recent runs**
   - `gh run list --limit 12`
   - Show: status, conclusion, workflow name, branch, duration

2. **If there are failures**
   - Get the failed run ID: `gh run list --status failure --limit 1 --json databaseId -q '.[0].databaseId'`
   - View details: `gh run view <id>`
   - Download logs: `gh run view <id> --log-failed`
   - Parse the error output:
     - If test failure: identify which test and read the relevant source + test files
     - If lint failure: show the specific ruff/mypy error
     - If build failure: check dependency issues
   - Suggest a fix based on the error

3. **If all green**
   - Report: last successful run, branch, duration
   - One-liner: "CI green, ready to merge/release"

4. **If $ARGUMENTS is a run ID**
   - `gh run view $ARGUMENTS --log`
   - Parse and summarize the output
