"""Python builder DSL for defining workflows in code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from agentloom.core.models import (
    AgentDefinition,
    Attachment,
    Condition,
    StateKeyConfig,
    StepDefinition,
    StepType,
    WebhookConfig,
    WorkflowConfig,
    WorkflowDefinition,
)
from agentloom.core.parser import WorkflowParser


class WorkflowBuilder:
    """Builder for constructing workflows programmatically."""

    def __init__(self, name: str, description: str = "", **config_kwargs: Any) -> None:
        self.name = name
        self.description = description
        self.config = WorkflowConfig(**config_kwargs)
        self.initial_state: dict[str, Any] = {}
        self.state_schema: dict[str, StateKeyConfig] = {}
        self._steps: list[StepDefinition] = []

    def set_state(self, **kwargs: Any) -> WorkflowBuilder:
        """Set initial state variables."""
        self.initial_state.update(kwargs)
        return self

    def set_state_schema(
        self, schema: Mapping[str, StateKeyConfig | Mapping[str, bool]]
    ) -> WorkflowBuilder:
        """Set per-key state settings, such as ``{"api_key": {"redact": True}}``; globs work."""
        self.state_schema.update(
            {key: StateKeyConfig.model_validate(config) for key, config in schema.items()}
        )
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
            conditions: List of (expression, target_step_id) tuples.
            default: Default target if no condition matches.
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

    def add_embed_step(
        self,
        step_id: str,
        inputs: str,
        model: str | None = None,
        dimensions: int | None = None,
        output: str | None = None,
        depends_on: list[str] | None = None,
    ) -> WorkflowBuilder:
        """Add an embedding step; ``inputs`` is a state path such as ``state.documents``."""
        self._steps.append(
            StepDefinition(
                id=step_id,
                type=StepType.EMBED,
                inputs=inputs,
                model=model,
                dimensions=dimensions,
                output=output,
                depends_on=depends_on or [],
            )
        )
        return self

    def add_approval_gate_step(
        self,
        step_id: str,
        timeout_seconds: int | None = None,
        on_timeout: Literal["approve", "reject"] | None = None,
        notify: WebhookConfig | None = None,
        output: str | None = None,
        depends_on: list[str] | None = None,
    ) -> WorkflowBuilder:
        """Add a step that pauses the workflow until a person approves or rejects it."""
        self._steps.append(
            StepDefinition(
                id=step_id,
                type=StepType.APPROVAL_GATE,
                timeout_seconds=timeout_seconds,
                on_timeout=on_timeout,
                notify=notify,
                output=output,
                depends_on=depends_on or [],
            )
        )
        return self

    def add_agent_step(
        self,
        step_id: str,
        agent: AgentDefinition | str | Path,
        prompt: str | None = None,
        conversation: str | None = None,
        output: str | None = None,
        depends_on: list[str] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> WorkflowBuilder:
        """Add a turn of an agent, given as a definition or the path of its YAML file."""
        # ``model_validate`` because the path form is only a valid input, not the field's type
        self._steps.append(
            StepDefinition.model_validate(
                {
                    "id": step_id,
                    "type": StepType.AGENT,
                    "agent": agent,
                    "prompt": prompt,
                    "conversation": conversation,
                    "output": output,
                    "depends_on": depends_on or [],
                    "attachments": attachments or [],
                }
            )
        )
        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition and check its references and dependency graph.

        Raises:
            ValidationError: A step names a step or file that doesn't exist, the
                dependencies form a cycle, or parallel steps share a conversation.
        """
        workflow = WorkflowDefinition(
            name=self.name,
            description=self.description,
            config=self.config,
            state=self.initial_state,
            state_schema=self.state_schema,
            steps=self._steps,
        )
        WorkflowParser._validate_references(workflow)
        WorkflowParser._validate_shared_conversations(workflow, WorkflowParser.build_dag(workflow))
        return workflow


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
