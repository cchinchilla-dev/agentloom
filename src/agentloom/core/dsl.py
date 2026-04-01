"""Python DSL for defining workflows with decorators."""

from __future__ import annotations

from typing import Any

from agentloom.core.models import (
    Attachment,
    Condition,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)


class WorkflowBuilder:
    """Builder for constructing workflows programmatically."""

    def __init__(self, name: str, description: str = "", **config_kwargs: Any) -> None:
        self.name = name
        self.description = description
        self.config = WorkflowConfig(**config_kwargs)
        self.initial_state: dict[str, Any] = {}
        self._steps: list[StepDefinition] = []

    def set_state(self, **kwargs: Any) -> WorkflowBuilder:
        """Set initial state variables."""
        self.initial_state.update(kwargs)
        return self

    def add_llm_step(
        self,
        step_id: str,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        output: str | None = None,
        depends_on: list[str] | None = None,
        attachments: list[Attachment] | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> WorkflowBuilder:
        """Add an LLM call step."""
        self._steps.append(
            StepDefinition(
                id=step_id,
                type=StepType.LLM_CALL,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                output=output,
                depends_on=depends_on or [],
                attachments=attachments or [],
                stream=stream,
                **kwargs,
            )
        )
        return self

    def add_tool_step(
        self,
        step_id: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        output: str | None = None,
        depends_on: list[str] | None = None,
    ) -> WorkflowBuilder:
        """Add a tool execution step."""
        self._steps.append(
            StepDefinition(
                id=step_id,
                type=StepType.TOOL,
                tool_name=tool_name,
                tool_args=tool_args or {},
                output=output,
                depends_on=depends_on or [],
            )
        )
        return self

    def add_router_step(
        self,
        step_id: str,
        conditions: list[tuple[str, str]],
        default: str | None = None,
        depends_on: list[str] | None = None,
    ) -> WorkflowBuilder:
        """Add a conditional routing step.

        Args:
            step_id: Step identifier.
            conditions: List of (expression, target_step_id) tuples.
            default: Default target if no condition matches.
            depends_on: Dependencies.
        """
        self._steps.append(
            StepDefinition(
                id=step_id,
                type=StepType.ROUTER,
                conditions=[
                    Condition(expression=expr, target=target) for expr, target in conditions
                ],
                default=default,
                depends_on=depends_on or [],
            )
        )
        return self

    def add_subworkflow_step(
        self,
        step_id: str,
        workflow_path: str | None = None,
        workflow_inline: dict[str, Any] | None = None,
        output: str | None = None,
        depends_on: list[str] | None = None,
    ) -> WorkflowBuilder:
        """Add a subworkflow step."""
        self._steps.append(
            StepDefinition(
                id=step_id,
                type=StepType.SUBWORKFLOW,
                workflow_path=workflow_path,
                workflow_inline=workflow_inline,
                output=output,
                depends_on=depends_on or [],
            )
        )
        return self

    def build(self) -> WorkflowDefinition:
        """Build and validate the workflow definition."""
        return WorkflowDefinition(
            name=self.name,
            description=self.description,
            config=self.config,
            state=self.initial_state,
            steps=self._steps,
        )


def workflow(name: str, description: str = "", **config_kwargs: Any) -> WorkflowBuilder:
    """Create a new workflow builder.

    Usage:
        wf = (
            workflow("my-workflow", provider="ollama", model="llama3")
            .set_state(question="What is Python?")
            .add_llm_step("answer", prompt="Answer: {question}", output="answer")
            .build()
        )
    """
    return WorkflowBuilder(name=name, description=description, **config_kwargs)
