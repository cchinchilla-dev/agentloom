"""CLI command: replay a workflow from a recorded responses file.

Thin alias over ``run`` that forces ``provider=mock`` and loads responses
from the given recording file. Observability is disabled by default.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import typer

from agentloom.cli.run import _run_async


def replay(
    workflow_path: Path = typer.Argument(..., help="Path to the workflow YAML file.", exists=True),
    recording: Path = typer.Option(
        ..., "--recording", "-r", help="Path to a recorded responses JSON file.", exists=True
    ),
    state: list[str] = typer.Option(
        [], "--state", "-s", help="State variables as key=value pairs."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    observability: bool = typer.Option(
        False, "--observability", help="Enable observability (disabled by default in replay)."
    ),
) -> None:
    """Replay a workflow using previously recorded LLM responses (no API calls)."""
    anyio.run(
        _run_async,
        workflow_path,
        state,
        None,  # provider_override
        None,  # model_override
        None,  # budget
        not observability,  # lite
        output_json,
        False,  # stream
        False,  # checkpoint
        ".agentloom/checkpoints",
        recording,  # mock_responses
        None,  # record
    )
