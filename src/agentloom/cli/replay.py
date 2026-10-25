"""CLI command: replay a workflow from a recorded responses file.

Thin alias over ``run`` that forces ``provider=mock`` and loads responses
from the given recording file, or, with a cut, serves the recording through the
workflow's own providers. Observability is disabled by default.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import typer

from agentloom.cli.run import _run_async
from agentloom.providers.tool_tape import ToolMode


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
    allow_default_fallback: bool = typer.Option(
        False,
        "--allow-default-fallback",
        help=(
            "Return the placeholder response on a recording miss instead of "
            "failing. Off by default — a replay miss should be a hard error."
        ),
    ),
    tools: ToolMode = typer.Option(
        ToolMode.LIVE,
        "--tools",
        help=(
            "Tool calls: `live` runs them, `replay` serves the results recorded with "
            "--record-tools, `verify` runs them and fails when a result differs."
        ),
    ),
    live_from: str | None = typer.Option(
        None,
        "--live-from",
        help=(
            "Serve the steps before this one from the recording, and run it and every step "
            "that depends on it live, with the workflow's providers and their keys."
        ),
    ),
    force_live: list[str] = typer.Option(
        [], "--force-live", help="Run this step live even when the recording matches it."
    ),
) -> None:
    """Replay a workflow using previously recorded LLM responses (no API calls).

    Replay is strict by default: a request that misses the recording, or
    matches a step whose prompt drifted since capture, raises
    ``RecordingMismatchError``. Pass ``--allow-default-fallback`` to
    restore the pre-0.5.0 lenient behaviour (placeholder response on a
    miss). ``--tools replay`` and ``--tools verify`` apply the same
    strictness to tool calls.

    ``--live-from`` and ``--force-live`` cut the replay (#165): those steps run
    live, and so does any other step from its first call that misses the
    recording. Every call is listed with where it came from.
    """
    if allow_default_fallback and (live_from or force_live):
        typer.echo(
            "Error: --allow-default-fallback does not combine with --live-from or "
            "--force-live, which run a step that misses the recording live.",
            err=True,
        )
        raise typer.Exit(1)
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
        not allow_default_fallback,  # mock_strict
        False,  # record_tools
        tools,
        live_from,
        force_live,
    )
