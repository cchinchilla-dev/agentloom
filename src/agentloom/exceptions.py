"""Custom exceptions for AgentLoom."""


class AgentLoomError(Exception):
    """Base exception for all AgentLoom errors."""


class WorkflowError(AgentLoomError):
    """Error during workflow execution."""


class StepError(AgentLoomError):
    """Error during step execution."""

    def __init__(self, step_id: str, message: str) -> None:
        self.step_id = step_id
        super().__init__(f"Step '{step_id}': {message}")


class ProviderError(AgentLoomError):
    """Error from an LLM provider."""

    def __init__(self, provider: str, message: str, status_code: int | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"Provider '{provider}': {message}")


class CircuitOpenError(ProviderError):
    """Circuit breaker is open for this provider."""

    def __init__(self, provider: str) -> None:
        super().__init__(provider, "Circuit breaker is open — provider temporarily disabled")


class RateLimitError(ProviderError):
    """Rate limit exceeded for this provider.

    Distinct from generic ``ProviderError`` so the gateway can back off the
    rate-limiter bucket instead of charging the failure against the circuit
    breaker (a throttled provider is healthy, just overused).
    """

    def __init__(self, provider: str, retry_after_s: float | None = None) -> None:
        self.retry_after_s = retry_after_s
        suffix = f" (retry_after={retry_after_s}s)" if retry_after_s is not None else ""
        super().__init__(provider, f"Rate limit exceeded{suffix}", status_code=429)


class BudgetExceededError(AgentLoomError):
    """Workflow budget has been exceeded."""

    def __init__(self, budget: float, spent: float) -> None:
        self.budget = budget
        self.spent = spent
        super().__init__(f"Budget exceeded: spent ${spent:.4f} of ${budget:.4f} limit")


class SandboxViolationError(AgentLoomError):
    """Tool execution blocked by sandbox policy."""

    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        super().__init__(f"Sandbox violation ({tool}): {message}")


class SecurityError(AgentLoomError):
    """An expression or input was rejected by a security policy.

    Distinct from StepError so that logs and metrics can flag attempted
    sandbox bypasses without conflating them with normal step failures.
    """

    def __init__(self, message: str, *, expression: str | None = None) -> None:
        self.expression = expression
        super().__init__(message)


class ValidationError(AgentLoomError):
    """Workflow or step definition validation error."""


class StateWriteError(AgentLoomError):
    """Refused state write: dotted path traverses a wrong-type intermediate.

    Raised by ``StateManager.set`` (and the underlying ``_set_nested``) when
    a dotted key writes through an intermediate segment whose existing value
    cannot accept the next segment: a scalar parent that the write would
    silently overwrite with a dict (``set("user.name", ...)`` when
    ``state.user`` is the string ``"alice"``), or a list parent traversed
    with a string segment (``set("users.name", ...)`` when ``state.users``
    is a list — the caller meant ``users[0].name``). The pre-0.5.0
    behaviour replaced the scalar with an empty dict and continued, or
    leaked a generic ``TypeError`` for the list case; this error surfaces
    both uniformly.
    """


class WorkflowTimeoutError(AgentLoomError):
    """Workflow exceeded its maximum execution time."""


class StepTimeoutError(StepError):
    """Step exceeded its maximum execution time."""

    def __init__(self, step_id: str, timeout: float) -> None:
        super().__init__(step_id, f"Timed out after {timeout}s")


class PauseRequestedError(AgentLoomError):
    """A step has requested the workflow to pause.

    Raised by step executors (e.g. an approval gate) to signal that the
    engine should save a checkpoint and stop execution until a human
    resumes the workflow.
    """

    def __init__(self, step_id: str, message: str = "") -> None:
        self.step_id = step_id
        super().__init__(message or f"Pause requested at step '{step_id}'")
