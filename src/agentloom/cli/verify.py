"""``agentloom verify-determinism``: replay a recording several times and compare (#135)."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import typer

from agentloom.core.inter_agent import (
    EVENTS_KEY,
    EventLog,
    dump_events,
    load_events,
    order_events,
    recording_events,
)
from agentloom.providers.tool_tape import ToolMode

if TYPE_CHECKING:
    from agentloom.contracts._inter_agent import InterAgentEvent
    from agentloom.core.results import WorkflowResult


def verify_determinism(
    workflow_path: Path = typer.Argument(..., help="Path to the workflow YAML file.", exists=True),
    recording: Path = typer.Option(
        ..., "--recording", "-r", help="Path to a recorded responses JSON file.", exists=True
    ),
    runs: int = typer.Option(2, "--runs", min=2, help="How many replays to compare."),
    state: list[str] = typer.Option(
        [], "--state", "-s", help="State variables as key=value pairs."
    ),
    tools: ToolMode = typer.Option(
        ToolMode.LIVE,
        "--tools",
        help="Tool calls, as in `replay`: `live`, `replay` or `verify`.",
    ),
    events_out: Path | None = typer.Option(
        None, "--events-out", help="Write the event log of the first replay as JSONL."
    ),
) -> None:
    """Replay a recording several times and check that every replay gives the same result.

    Each replay is strict, as in ``agentloom replay``. The runs are compared on
    their status, each step's status, output and error, the final state and
    the inter-agent event log, leaving out run ids, durations and timestamps;
    the event log is also compared with the one in the recording, when it has
    one. On a difference, the exit code is 1 and a unified diff goes to stderr.
    """
    try:
        recorded = json.loads(recording.read_text(encoding="utf-8"))
        logged = recorded.get(EVENTS_KEY) if isinstance(recorded, dict) else None
        expected = order_events(load_events(logged)) if logged is not None else None
    except (json.JSONDecodeError, ValueError) as e:  # pydantic's ValidationError is one
        typer.echo(f"Error: {recording} is not a recording with a valid event log: {e}", err=True)
        raise typer.Exit(1) from e

    replays = [anyio.run(_replay, workflow_path, recording, state, tools) for _ in range(runs)]
    first, events = replays[0]
    if events_out is not None:
        lines = (json.dumps(event) for event in dump_events(events))
        events_out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

    baseline = _lines(first, events)
    for number, (result, run_events) in enumerate(replays[1:], start=2):
        if diff := _diff(baseline, _lines(result, run_events), "replay 1", f"replay {number}"):
            typer.echo(f"Replay {number} differs from replay 1:", err=True)
            typer.echo(diff, err=True)
            raise typer.Exit(1)
    if expected is not None:
        want, got = _event_lines(expected), _event_lines(events)
        if diff := _diff(want, got, "recorded events", "replayed events"):
            typer.echo("The replayed event log differs from the recorded one:", err=True)
            typer.echo(diff, err=True)
            raise typer.Exit(1)

    against = ", and the recorded event log" if expected is not None else ""
    typer.echo(
        f"Deterministic: {runs} replays of {recording} give the same result{against} "
        f"(status {first.status.value}, {len(events)} events)."
    )


async def _replay(
    workflow_path: Path, recording: Path, state_args: list[str], tools: ToolMode
) -> tuple[WorkflowResult, list[InterAgentEvent]]:
    """One strict replay of ``recording``, with a fresh engine, mock and tools."""
    from agentloom.cli.run import _initial_state, _tool_registry
    from agentloom.core.engine import WorkflowEngine
    from agentloom.core.parser import WorkflowParser
    from agentloom.core.state import StateManager
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.providers.mock import MockProvider
    from agentloom.providers.tool_tape import ToolReplayer

    try:
        workflow = WorkflowParser.from_yaml(workflow_path)
        mock = MockProvider(responses_file=recording, workflow_name=workflow.name, strict=True)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    workflow.config.provider = "mock"
    gateway = ProviderGateway()
    gateway.register(mock, priority=0)
    tool_registry = _tool_registry(workflow)
    if tools is not ToolMode.LIVE:
        tool_registry.tape = ToolReplayer(recording, tools, strict=True)
    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=StateManager(initial_state=_initial_state(workflow, state_args)),
        provider_gateway=gateway,
        tool_registry=tool_registry,
    )
    log = EventLog()
    with recording_events(log):
        result = await engine.run()
    await gateway.close()
    return result, log.events()


def _lines(result: WorkflowResult, events: list[InterAgentEvent]) -> list[str]:
    """What two replays must agree on, as JSON lines."""
    canonical: dict[str, Any] = {
        "status": result.status.value,
        "steps": {
            step_id: {
                "status": step.status.value,
                "output": step.output,
                "error": step.error,
                "error_classification": step.error_classification,
            }
            for step_id, step in result.step_results.items()
        },
        "final_state": result.final_state,
    }
    text = json.dumps(canonical, indent=2, sort_keys=True, default=str)
    return [*text.splitlines(), *_event_lines(events)]


def _event_lines(events: list[InterAgentEvent]) -> list[str]:
    return [
        json.dumps({k: v for k, v in event.items() if k != "timestamp_ns"}, sort_keys=True)
        for event in dump_events(events)
    ]


def _diff(a: list[str], b: list[str], name_a: str, name_b: str) -> str:
    return "\n".join(difflib.unified_diff(a, b, name_a, name_b, lineterm=""))
