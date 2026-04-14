---
name: gen-tests
description: Generate tests for a module or function. Focuses on edge cases and real failure modes, not happy paths.
---

Generate tests for the file or module specified in `$ARGUMENTS`. (Testing stack — pytest, `pytest-asyncio` auto mode, `respx` — is documented in CLAUDE.md.)

## Process

1. Read the source file end-to-end. Map every branch and raised exception.
2. Read existing tests for that module to avoid duplication.
3. Prioritise what's NOT tested:
   - Error paths and exception handling
   - Edge cases (empty / None / boundary / unexpected types)
   - Async behaviour (cancellation, timeout, concurrent access, `get_state_snapshot`)
   - Observer hook callbacks — mock the observer and assert on call args
   - Integration points (does the step actually write to state?)
4. Use fixtures from `tests/conftest.py` (MockProvider, mock_gateway, etc.) rather than rebuilding them.
5. Test classes grouped by behaviour, not by method. Names describe the scenario: `test_circuit_breaker_resets_after_timeout`, not `test_cb_1`.

## Decide: test vs pragma

For each uncovered line, pick one:

- **Reachable from a public API with small setup** → write a test. Prefer `FakeStream`-style doubles for socket I/O.
- **Defensive branch that the caller already guards** (e.g. `if notify is None: return`) → `# pragma: no cover — <reason>`.
- **Fallback only active in a different backend** (e.g. prom branch when OTel is installed) → `# pragma: no cover — prom fallback`.
- **Runtime-only callbacks** (OTel observable-gauge callbacks) → `# pragma: no cover — fires on OTel export`.

Never a bare `# pragma: no cover` — always include the reason after the em dash.

## Regression tests

If the work includes a bug fix, a Copilot/review finding, or a branch that was silently broken:

- Add a dedicated test named `test_<scenario>_regression` or `test_<bug>_does_not_<recur>`.
- Co-locate it with related tests (same file / same class).
- It must fail against the pre-fix code and pass with the fix — verify by temporarily reverting the fix or by reasoning from the diff.
- Only include regression tests that match the scope of the PR. Don't bolt them on for unrelated gaps just to pad coverage.

## Rules

- Don't test getters/setters or trivial constructors.
- Don't mock what you can test directly.
- Every test asserts something specific — no `assert True`.
- If the module has a TODO/FIXME, write a test that pins the current behaviour so the bug fix has a target.
- Verify: `uv run pytest tests/<file> -v` then `/check`.
