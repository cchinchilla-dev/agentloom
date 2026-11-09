"""Parse workflow definitions from YAML or dicts."""

from __future__ import annotations

import errno
import logging
import warnings
from pathlib import Path
from typing import Any

import yaml

from agentloom.core.dag import DAG
from agentloom.core.models import StepType, WorkflowDefinition
from agentloom.exceptions import ValidationError

logger = logging.getLogger("agentloom.parser")

_MAX_FILE_NAME = 255


class WorkflowParser:
    """Parser and validator for workflow definitions."""

    @staticmethod
    def from_yaml(path_or_str: str | Path) -> WorkflowDefinition:
        """Parse a workflow from a YAML file path or YAML string.

        A string with no file behind it is read as YAML, unless it looks like a
        path: one non-empty line without a ``:``.

        Raises:
            ValidationError: If the file is missing, the YAML is invalid or the
                workflow is malformed.
        """
        text = WorkflowParser._read_source(path_or_str)

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML: {e}") from e

        if not isinstance(raw, dict):
            raise ValidationError("Workflow YAML must be a mapping at the top level")

        return WorkflowParser.from_dict(raw)

    @staticmethod
    def _read_source(path_or_str: str | Path) -> str:
        """Return the YAML text behind *path_or_str*."""
        text = str(path_or_str)
        path = Path(path_or_str)
        # A name over the OS limit: Python 3.14 answers False, earlier versions raise
        try:
            is_file = path.is_file()
        except OSError as e:
            if e.errno != errno.ENAMETOOLONG:
                raise
            is_file = False
        if is_file:
            return path.read_text(encoding="utf-8")
        stripped = text.strip()
        one_line = bool(stripped) and "\n" not in stripped and ":" not in text
        nameable = all(len(part) <= _MAX_FILE_NAME for part in path.parts)
        if isinstance(path_or_str, Path) or (one_line and nameable):
            raise ValidationError(f"Workflow file not found: {text}")
        return text

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WorkflowDefinition:
        """Parse a workflow from a dictionary, such as ``yaml.safe_load`` returns.

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
        WorkflowParser._validate_shared_conversations(workflow, dag)

        return workflow

    @staticmethod
    def _validate_references(workflow: WorkflowDefinition) -> None:
        """Validate that every depends_on, router target and subworkflow file exists."""
        step_ids = set(workflow.step_ids())
        errors: list[str] = []

        for step in workflow.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on non-existent step '{dep}'")

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

        # Test pairs: an owner downstream of another can still race a third, independent one
        def _ancestors(sid: str) -> set[str]:
            seen: set[str] = {sid}
            stack = [sid]
            while stack:
                node = stack.pop()
                for pred in dag.predecessors(node):
                    if pred not in seen:
                        seen.add(pred)
                        stack.append(pred)
            seen.discard(sid)
            return seen

        collisions: list[tuple[str, list[str]]] = []
        for key, owners in outputs.items():
            if len(owners) < 2:
                continue
            ancestry = {sid: _ancestors(sid) for sid in owners}
            concurrent: set[str] = set()
            for i, a in enumerate(owners):
                for b in owners[i + 1 :]:
                    # Neither upstream of the other: the scheduler may run them in either order
                    if a not in ancestry[b] and b not in ancestry[a]:
                        concurrent.add(a)
                        concurrent.add(b)
            if len(concurrent) >= 2:
                collisions.append((key, sorted(concurrent)))

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
    def _validate_shared_conversations(workflow: WorkflowDefinition, dag: DAG) -> None:
        """Refuse two steps of one layer on one conversation; parallel writes lose turns.

        Each step loads the whole conversation and writes it back, so the last
        to finish erases the other's turns. Branches of one router are exempt:
        only one of them runs.
        """
        routers = [s.id for s in workflow.steps if s.type == StepType.ROUTER]
        branches = {
            router: {child: dag.transitive_successors({child}) for child in dag.successors(router)}
            for router in routers
        }

        def _exclusive(a: str, b: str) -> bool:
            for subtrees in branches.values():
                of_a = {child for child, nodes in subtrees.items() if a in nodes}
                of_b = {child for child, nodes in subtrees.items() if b in nodes}
                # Different children of the router: taking one skips the other's whole subtree
                if of_a and of_b and len(of_a | of_b) > 1:
                    return True
            return False

        errors: list[str] = []
        for layer in dag.execution_layers():
            sharing: dict[str, list[str]] = {}
            for step_id in layer:
                step = workflow.get_step(step_id)
                if step is not None and step.conversation:
                    sharing.setdefault(step.conversation.removeprefix("state."), []).append(step_id)
            for key, owners in sharing.items():
                racing = sorted(
                    {
                        step_id
                        for i, a in enumerate(owners)
                        for b in owners[i + 1 :]
                        if not _exclusive(a, b)
                        for step_id in (a, b)
                    }
                )
                if racing:
                    errors.append(f"conversation {key!r} is used by steps {racing}")

        if errors:
            raise ValidationError(
                "Steps in the same layer share a conversation and would lose each "
                f"other's turns: {'; '.join(errors)}. Chain them with depends_on."
            )

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
