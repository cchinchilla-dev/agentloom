"""CLI command: list checkpoint runs."""

from __future__ import annotations

import json

import anyio
import typer


def runs(
    checkpoint_dir: str = typer.Option(
        ".agentloom/checkpoints", "--checkpoint-dir", help="Checkpoint storage directory."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all checkpointed workflow runs."""
    anyio.run(_runs_async, checkpoint_dir, output_json)


async def _runs_async(checkpoint_dir: str, output_json: bool) -> None:
    from agentloom.checkpointing.file import FileCheckpointer

    checkpointer = FileCheckpointer(checkpoint_dir=checkpoint_dir)
    entries = await checkpointer.list_runs()

    if not entries:
        typer.echo("No checkpoint runs found.")
        return

    if output_json:
        typer.echo(json.dumps([e.model_dump() for e in entries], indent=2, default=str))
        return

    # Table output
    typer.echo(f"{'RUN ID':<14} {'WORKFLOW':<25} {'STATUS':<12} {'UPDATED'}")
    typer.echo("-" * 72)
    for entry in entries:
        updated = entry.updated_at[:19] if entry.updated_at else "—"
        typer.echo(f"{entry.run_id:<14} {entry.workflow_name:<25} {entry.status:<12} {updated}")
