---
name: gen-tests
description: Generate tests for a module or function. Focuses on edge cases and real failure modes, not happy paths.
---

Generate tests for the file or module specified in $ARGUMENTS.

## Process

1. Read the source file thoroughly. Understand every code path.
2. Read existing tests for that module (if any) to avoid duplicating coverage.
3. Identify what's NOT tested:
   - Error paths and exception handling
   - Edge cases (empty inputs, None, boundary values)
   - Async behavior (cancellation, timeout, concurrent access)
   - Integration between components (e.g., does the step actually update state?)
4. Write the tests following project conventions:
   - `pytest` + `pytest-asyncio` (auto mode)
   - `respx` for HTTP mocking — never hit real APIs
   - Test classes grouped by behavior, not by method
   - Descriptive names: `test_circuit_breaker_resets_after_timeout`, not `test_cb_1`
   - Use fixtures from `tests/conftest.py` (MockProvider, mock_gateway, etc.)

## Rules
- Don't test obvious things (getters, setters, trivial constructors)
- Don't mock what you can test directly
- Every test must assert something specific — no "assert True"
- If the module has TODOs or FIXMEs, write tests that document the current (broken) behavior
- Run `uv run pytest tests/<new_test_file> -v` to verify they pass
