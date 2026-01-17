"""Custom exceptions for AgentLoom."""


class AgentLoomError(Exception):
    """Base exception for all AgentLoom errors."""


class WorkflowError(AgentLoomError):
    """Error during workflow execution."""


class StepError(AgentLoomError):
    """Error during step execution."""

    def __init__(self, step_id: str, message: str) -> None:
        self.step_id = step_id
        super().__init__(f"Step \'{step_id}\': {message}")


class ValidationError(AgentLoomError):
    """Workflow or step definition validation error."""
