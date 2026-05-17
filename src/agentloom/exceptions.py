"""Custom exceptions for AgentLoom.

Errors carry an ``is_retryable`` class attribute that the resilience layer
consults when deciding whether to consume the retry budget. Subclasses
that represent a deterministic permanent failure (sandbox violation,
attachment-not-found, tool-not-found, template typo, validation error)
override the default to ``False`` so the retry loop bails out on the
first attempt instead of wasting 10–127 s on backoff for a failure that
never recovers.
"""


class AgentLoomError(Exception):
    """Base exception for all AgentLoom errors.

    Default ``is_retryable = True`` mirrors the historic "retry on
    status-less exceptions" behaviour — a network error or generic
    provider hiccup is presumed transient unless a subclass declares
    otherwise.
    """

    is_retryable: bool = True


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
    """Workflow budget has been exceeded.

    Non-retryable: the spend doesn't decrease between attempts, so the
    next call sees the same overrun. Pre-0.5.0 a budget breach inside a
    subworkflow step would propagate as a generic step failure, get
    retried 4×, and only then reach the parent's terminal classifier;
    the marker stops the retry loop and lets the parent surface
    ``BUDGET_EXCEEDED`` directly.
    """

    is_retryable = False

    def __init__(self, budget: float, spent: float) -> None:
        self.budget = budget
        self.spent = spent
        super().__init__(f"Budget exceeded: spent ${spent:.4f} of ${budget:.4f} limit")


class SandboxViolationError(AgentLoomError):
    """Tool execution blocked by sandbox policy.

    Non-retryable: the policy is deterministic, so the next attempt will
    refuse identically. Pre-0.5.0 the resilience layer burned the full
    retry budget here, adding 10 s of backoff to every blocked tool call.
    """

    is_retryable = False

    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        super().__init__(f"Sandbox violation ({tool}): {message}")


class SecurityError(AgentLoomError):
    """An expression or input was rejected by a security policy.

    Distinct from StepError so that logs and metrics can flag attempted
    sandbox bypasses without conflating them with normal step failures.
    """

    is_retryable = False

    def __init__(self, message: str, *, expression: str | None = None) -> None:
        self.expression = expression
        super().__init__(message)


class ValidationError(AgentLoomError):
    """Workflow or step definition validation error."""

    is_retryable = False


class AttachmentResolutionError(AgentLoomError, ValueError):
    """Attachment resolution refused for a deterministic reason.

    Raised when an LLM-call attachment cannot be resolved because of a
    shape/policy failure that never recovers: unsupported attachment
    type, size limit exceeded, empty source, unsupported URL passthrough.
    Inherits from ``ValueError`` so the pre-0.5.0 ``except ValueError``
    call sites and ``pytest.raises(ValueError)`` assertions keep working;
    the only behavioural change is the resilience layer now sees the
    ``is_retryable = False`` marker and skips retries.

    Transient failures (network errors, connection timeouts, 5xx
    responses) are intentionally NOT wrapped — those still surface as
    raw ``httpx.HTTPError`` / ``httpx.NetworkError`` so the existing
    status-code-driven retry rules continue to apply.
    """

    is_retryable = False

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        # Skip ``AgentLoomError.__init__`` chaining — Python's MRO
        # ensures ``ValueError.__init__`` runs via ``super().__init__``;
        # we just need ``str(exc)`` to render the formatted message.
        super().__init__(f"Attachment '{source}': {message}")


class ToolNotFoundError(AgentLoomError, KeyError):
    """Tool name was not registered in the workflow's tool registry.

    Inherits from ``KeyError`` so the pre-0.5.0 ``except KeyError`` at
    ``tool_step._resolve_args`` keeps catching it — the only thing that
    changes is the resilience layer now skips retries.
    """

    is_retryable = False

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        rendered = ", ".join(sorted(available)) or "(none)"
        # ``KeyError.__str__`` adds quotes around the message — call
        # ``AgentLoomError.__init__`` directly so the formatted message
        # surfaces cleanly. Storing the rendered string on ``.args[0]``
        # keeps ``str(exc)`` and ``repr(exc)`` informative either way.
        AgentLoomError.__init__(self, f"Tool '{name}' not found. Available: {rendered}")


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
