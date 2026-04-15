"""Parse workflow definitions from YAML or dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentloom.core.dag import DAG
from agentloom.core.models import WorkflowDefinition
from agentloom.exceptions import ValidationError


class WorkflowParser:
    """Parses and validates workflow definitions."""

    @staticmethod
    def from_yaml(path_or_str: str | Path) -> WorkflowDefinition:
        """Parse a workflow from a YAML file path or YAML string.

        Args:
            path_or_str: Path to a YAML file, or a YAML string.

        Returns:
            Validated WorkflowDefinition.

        Raises:
            ValidationError: If the YAML is invalid or the workflow is malformed.
        """
        path = Path(path_or_str)
        text = path.read_text() if path.exists() else str(path_or_str)

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML: {e}") from e

        if not isinstance(raw, dict):
            raise ValidationError("Workflow YAML must be a mapping at the top level")

        return WorkflowParser.from_dict(raw)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WorkflowDefinition:
        """Parse a workflow from a dictionary.

        Args:
            data: Raw workflow data (e.g., from yaml.safe_load).

        Returns:
            Validated WorkflowDefinition.

        Raises:
            ValidationError: If the data is invalid.
        """
        try:
            workflow = WorkflowDefinition.model_validate(data)
        except Exception as e:
            raise ValidationError(f"Invalid workflow definition: {e}") from e

        WorkflowParser._validate_references(workflow)

        steps = [(s.id, s.depends_on) for s in workflow.steps]
        dag = DAG.from_steps(steps)
        errors = dag.validate()
        if errors:
            raise ValidationError(
                "DAG validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return workflow

    @staticmethod
    def _validate_references(workflow: WorkflowDefinition) -> None:
        """Validate that all step references (depends_on, router targets) exist."""
        step_ids = set(workflow.step_ids())
        errors: list[str] = []

        for step in workflow.steps:
            # Validate depends_on
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on non-existent step '{dep}'")

            # Validate router targets
            if step.type.value == "router":
                for cond in step.conditions:
                    if cond.target not in step_ids:
                        errors.append(
                            f"Router step '{step.id}' references non-existent "
                            f"target '{cond.target}'"
                        )
                if step.default and step.default not in step_ids:
                    errors.append(
                        f"Router step '{step.id}' has non-existent default target '{step.default}'"
                    )

            # Validate subworkflow path
            if step.type.value == "subworkflow" and step.workflow_path:
                path = Path(step.workflow_path)
                if not path.exists():
                    errors.append(
                        f"Subworkflow step '{step.id}' references non-existent "
                        f"file '{step.workflow_path}'"
                    )

        if errors:
            detail = "\n".join(f"  - {e}" for e in errors)
            raise ValidationError(f"Workflow validation errors:\n{detail}")

    @staticmethod
    def build_dag(workflow: WorkflowDefinition) -> DAG:
        """Build a DAG from a workflow definition."""
        steps = [(s.id, s.depends_on) for s in workflow.steps]
        dag = DAG.from_steps(steps)

        errors = dag.validate()
        if errors:
            raise ValidationError(
                "DAG validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return dag
