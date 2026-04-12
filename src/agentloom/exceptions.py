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
    """Rate limit exceeded for this provider."""

    def __init__(self, provider: str) -> None:
        super().__init__(provider, "Rate limit exceeded")


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


class ValidationError(AgentLoomError):
    """Workflow or step definition validation error."""


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
