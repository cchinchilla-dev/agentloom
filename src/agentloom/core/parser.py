"""Parse workflow definitions from YAML or dicts."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import yaml

from agentloom.core.dag import DAG
from agentloom.core.models import WorkflowDefinition
from agentloom.exceptions import ValidationError

logger = logging.getLogger("agentloom.parser")


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

        WorkflowParser._validate_output_collisions(workflow, dag)

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
    def _validate_output_collisions(workflow: WorkflowDefinition, dag: DAG) -> None:
        """Warn (or error under ``config.strict_outputs``) on shared output keys.

        Two parallel-eligible steps writing the same ``output:`` key collapse
        silently at runtime under last-writer-wins semantics: the survivor
        depends on layer scheduling order, which is implementation-defined.
        Steps where one transitively depends on the other are exempt — that
        chain is an explicit, intentional overwrite (and the second step
        sees the first's value beforehand).

        Default surface is a single ``UserWarning`` so existing workflows
        depending on the last-writer-wins pattern keep running while the
        author notices. ``config.strict_outputs: true`` promotes to a parse
        error for workflows that want the strict contract.
        """
        outputs: dict[str, list[str]] = {}
        for s in workflow.steps:
            if s.output:
                outputs.setdefault(s.output, []).append(s.id)

        collisions: list[tuple[str, list[str]]] = []
        for key, owners in outputs.items():
            if len(owners) < 2:
                continue
            # Filter to truly parallel pairs: drop owners that sit in each
            # other's transitive ancestry. Anything left in ``parallel`` is
            # reachable concurrently in the layer scheduler.
            parallel: list[str] = []
            for sid in owners:
                ancestors_of_sid: set[str] = set()
                # collect transitive predecessors
                stack = [sid]
                seen_pred = {sid}
                while stack:
                    node = stack.pop()
                    for pred in dag.predecessors(node):
                        if pred not in seen_pred:
                            seen_pred.add(pred)
                            ancestors_of_sid.add(pred)
                            stack.append(pred)
                if not any(other in ancestors_of_sid for other in owners if other != sid):
                    parallel.append(sid)
            if len(parallel) >= 2:
                collisions.append((key, sorted(parallel)))

        if not collisions:
            return

        details = "; ".join(f"output={key!r} written by {owners}" for key, owners in collisions)
        msg = (
            "Parallel steps share an output key under last-writer-wins "
            f"semantics: {details}. Set config.strict_outputs: true to "
            "promote this to a parse error, or chain the steps via "
            "depends_on if the overwrite is intentional."
        )
        if workflow.config.strict_outputs:
            raise ValidationError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

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
