---
name: gen-tests
description: Write tests for a module or function, focused on error paths and edge cases rather than happy paths. Use after adding or changing code that lacks tests, or with a bug fix that needs a regression test.
---

Generate tests for the file or module given as the argument. (Testing stack — pytest, `pytest-asyncio` auto mode, `respx` — is documented in AGENTS.md.)

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

## Untestable lines

When a line can't be reached with a small test, apply the test-vs-pragma rules of the `coverage` skill. Never a bare `# pragma: no cover`.

## Regression tests

If the work includes a bug fix, a review finding, or a branch that was silently broken:

- Add a dedicated test named `test_<scenario>_regression` or `test_<bug>_does_not_<recur>`.
- Co-locate it with related tests (same file / same class).
- It must fail against the pre-fix code and pass with the fix — verify by temporarily reverting the fix or by reasoning from the diff.
- Only include regression tests that match the scope of the PR. Don't bolt them on for unrelated gaps just to pad coverage.

## Rules

- Don't test getters/setters or trivial constructors.
- Don't mock what you can test directly.
- Every test asserts something specific — no `assert True`.
- If the module has a TODO/FIXME, write a test that pins the current behaviour so the bug fix has a target.
- Verify: `uv run pytest tests/<file> -v` then run the `check` skill.
