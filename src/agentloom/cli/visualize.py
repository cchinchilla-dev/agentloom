"""CLI command: visualize a workflow DAG."""

from __future__ import annotations

from pathlib import Path

import typer

from agentloom.core.models import StepType
from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import ValidationError


def visualize(
    workflow_path: Path = typer.Argument(..., help="Path to the workflow YAML file.", exists=True),
    format: str = typer.Option("ascii", "--format", "-f", help="Output format: ascii or mermaid."),
) -> None:
    """Visualize a workflow as an ASCII diagram or Mermaid graph."""
    try:
        workflow = WorkflowParser.from_yaml(workflow_path)
        dag = WorkflowParser.build_dag(workflow)
    except (ValidationError, Exception) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "mermaid":
        _print_mermaid(workflow, dag)
    else:
        _print_ascii(workflow, dag)


def _print_ascii(workflow: object, dag: object) -> None:
    """Print an ASCII representation of the workflow DAG."""
    from agentloom.core.dag import DAG
    from agentloom.core.models import WorkflowDefinition

    w: WorkflowDefinition = workflow  # type: ignore[assignment]
    d: DAG = dag  # type: ignore[assignment]
    layers = d.execution_layers()

    step_types = {s.id: s.type for s in w.steps}
    type_icons = {
        StepType.LLM_CALL: "LLM",
        StepType.TOOL: "TOOL",
        StepType.ROUTER: "IF/ELSE",
        StepType.SUBWORKFLOW: "SUB",
    }

    typer.echo(f"\n  Workflow: {w.name}")
    typer.echo(f"  {'=' * 50}")

    for i, layer in enumerate(layers):
        if i > 0:
            # Draw arrows from previous layer
            typer.echo(f"  {'  |':^50}")
            typer.echo(f"  {'  v':^50}")

        # Draw steps in this layer
        boxes = []
        for step_id in layer:
            stype = step_types.get(step_id, StepType.LLM_CALL)
            icon = type_icons.get(stype, "?")
            box = f"[{icon}: {step_id}]"
            boxes.append(box)

        if len(boxes) == 1:
            typer.echo(f"  {boxes[0]:^50}")
        else:
            # Parallel steps side by side
            line = "  " + "   ".join(boxes)
            typer.echo(line)

    typer.echo(f"  {'=' * 50}\n")


def _print_mermaid(workflow: object, dag: object) -> None:
    """Print a Mermaid graph definition."""
    from agentloom.core.models import WorkflowDefinition

    w: WorkflowDefinition = workflow  # type: ignore[assignment]

    step_types = {s.id: s.type for s in w.steps}

    typer.echo("```mermaid")
    typer.echo("graph TD")

    for step in w.steps:
        stype = step_types.get(step.id, StepType.LLM_CALL)
        if stype == StepType.ROUTER:
            typer.echo(f"    {step.id}{{{step.id}}}")
        elif stype == StepType.TOOL:
            typer.echo(f"    {step.id}[/{step.id}/]")
        elif stype == StepType.SUBWORKFLOW:
            typer.echo(f"    {step.id}[[{step.id}]]")
        else:
            typer.echo(f"    {step.id}[{step.id}]")

    for step in w.steps:
        for dep in step.depends_on:
            dep_step = w.get_step(dep)
            if dep_step and dep_step.type == StepType.ROUTER:
                for cond in dep_step.conditions:
                    if cond.target == step.id:
                        label = cond.expression[:20]
                        typer.echo(f"    {dep} -->|{label}| {step.id}")
                        break
                else:
                    if dep_step.default == step.id:
                        typer.echo(f"    {dep} -->|default| {step.id}")
                    else:
                        typer.echo(f"    {dep} --> {step.id}")
            else:
                typer.echo(f"    {dep} --> {step.id}")

    typer.echo("```")
