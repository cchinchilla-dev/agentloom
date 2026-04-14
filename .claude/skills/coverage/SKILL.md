---
name: coverage
description: Raise Codecov patch coverage above the 85% threshold by testing real edge cases or marking genuinely untestable code with `# pragma: no cover`.
---

Close gaps reported by Codecov or by `pytest --cov-report=term-missing`.

## Process

1. **Find the gaps**
   ```
   uv run pytest tests/<area>/ --cov=agentloom.<module> --cov-report=term-missing -q --no-cov-on-fail
   ```
   Read the `Missing` column — each number is a line or a range.

2. **Classify each gap**
   - **Testable** (real branch reachable from public API) → write a test. Prefer `FakeStream`-style doubles for I/O, `respx` for HTTP, mock observers for hook callbacks.
   - **Defensive / unreachable in practice** (e.g. `if notify is None: return` after the caller already checked) → `# pragma: no cover — <reason>`.
   - **Fallback code path** (e.g. the prometheus_client branch when OTel is installed) → `# pragma: no cover — prom fallback`.
   - **Runtime-only hooks** (e.g. OTel observable-gauge callbacks fired on export) → `# pragma: no cover — fires on OTel export`.

3. **Always include a reason** after the `—`. A bare `# pragma: no cover` is a smell.

4. **Prefer tests over pragmas** when the branch is reachable from unit tests in under ~20 lines of setup. Use pragmas for:
   - Protocol/adapter entrypoints that only run under real sockets/processes
   - `if TYPE_CHECKING:` blocks
   - Defensive returns that exist only for type narrowing

6. **Verify**
   ```
   uv run pytest tests/<area>/ --cov=agentloom.<module> --cov-report=term-missing -q --no-cov-on-fail
   ```
   Then `/check` once the module is clean.
