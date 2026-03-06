"""CLI command: validate a workflow definition."""

from __future__ import annotations

from pathlib import Path

import typer

from agentloom.core.parser import WorkflowParser
from agentloom.exceptions import ValidationError


def validate(
    workflow_path: Path = typer.Argument(..., help="Path to the workflow YAML file.", exists=True),
) -> None:
    """Validate a workflow YAML file for correctness."""
    try:
        workflow = WorkflowParser.from_yaml(workflow_path)
    except ValidationError as e:
        typer.echo(f"Validation FAILED:\n{e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Also validate the DAG
    try:
        dag = WorkflowParser.build_dag(workflow)
    except ValidationError as e:
        typer.echo(f"DAG validation FAILED:\n{e}", err=True)
        raise typer.Exit(1)

    layers = dag.execution_layers()

    typer.echo(f"Workflow '{workflow.name}' is valid.")
    typer.echo(f"  Version:  {workflow.version}")
    typer.echo(f"  Steps:    {len(workflow.steps)}")
    typer.echo(f"  Layers:   {len(layers)}")
    typer.echo(f"  Provider: {workflow.config.provider}")
    typer.echo(f"  Model:    {workflow.config.model}")

    if workflow.config.budget_usd:
        typer.echo(f"  Budget:   ${workflow.config.budget_usd:.2f}")

    typer.echo("\n  Execution plan:")
    for i, layer in enumerate(layers):
        parallel = " (parallel)" if len(layer) > 1 else ""
        typer.echo(f"    Layer {i}: {', '.join(layer)}{parallel}")
