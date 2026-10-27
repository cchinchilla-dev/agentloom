"""CLI command: run a workflow."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import typer

from agentloom.core.inter_agent import EVENTS_KEY, EventLog, dump_events, recording_events
from agentloom.core.parser import WorkflowParser
from agentloom.core.results import WorkflowStatus
from agentloom.core.state import StateManager
from agentloom.exceptions import ProviderError

if TYPE_CHECKING:
    from agentloom.core.models import WorkflowConfig, WorkflowDefinition
    from agentloom.observability.observer import WorkflowObserver
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.providers.mock import MockProvider
    from agentloom.providers.replay_or_live import ReplayCut
    from agentloom.tools.registry import ToolRegistry


def _parse_state_value(raw: str) -> Any:
    """Decode a CLI ``--state`` value, parsing JSON-shaped inputs.

    ``--state items=[1,2,3]`` and ``--state user={"name":"x"}`` now yield
    a real list / dict instead of the string literal. The heuristic only
    runs ``json.loads`` on plausibly-JSON inputs — a value that opens
    with ``[``, ``{`` or ``"``, the bare tokens ``true`` / ``false`` /
    ``null``, or a number — so common string values (``msg=hello``, a
    URL like ``url=https://x?a=b&c=d``) are left untouched. A
    JSON-shaped value that fails to parse also falls back to the raw
    string rather than erroring.
    """
    stripped = raw.strip()
    if not stripped:
        return raw
    looks_json = (
        stripped[0] in '[{"'
        or stripped in ("true", "false", "null")
        or stripped[0].isdigit()
        or (stripped[0] == "-" and len(stripped) > 1 and stripped[1].isdigit())
    )
    if looks_json:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return raw


def run(
    workflow_path: Path = typer.Argument(..., help="Path to the workflow YAML file.", exists=True),
    state: list[str] = typer.Option(
        [], "--state", "-s", help="State variables as key=value pairs."
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Override default provider."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override default model."),
    budget: float | None = typer.Option(None, "--budget", "-b", help="Maximum budget in USD."),
    lite: bool = typer.Option(False, "--lite", help="Run in lite mode (no observability)."),
    output_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM output in real-time."),
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Enable checkpointing."),
    checkpoint_dir: str = typer.Option(
        ".agentloom/checkpoints", "--checkpoint-dir", help="Checkpoint storage directory."
    ),
    mock_responses: Path | None = typer.Option(
        None, "--mock-responses", help="Replay responses from a JSON file via MockProvider."
    ),
    record: Path | None = typer.Option(
        None, "--record", help="Record provider responses to a JSON file for later replay."
    ),
    record_tools: bool = typer.Option(
        False,
        "--record-tools",
        help="With --record, also record tool results so `replay --tools` can use them.",
    ),
) -> None:
    """Execute a workflow from a YAML definition file."""
    if mock_responses is not None and record is not None:
        typer.echo("Error: --mock-responses and --record are mutually exclusive.", err=True)
        raise typer.Exit(2)
    if record_tools and record is None:
        typer.echo("Error: --record-tools needs --record.", err=True)
        raise typer.Exit(2)
    anyio.run(
        _run_async,
        workflow_path,
        state,
        provider,
        model,
        budget,
        lite,
        output_json,
        stream,
        checkpoint,
        checkpoint_dir,
        mock_responses,
        record,
        False,  # mock_strict — `run` keeps the lenient dev fallback; `replay` is strict
        record_tools,
    )


async def _run_async(
    workflow_path: Path,
    state_args: list[str],
    provider_override: str | None,
    model_override: str | None,
    budget: float | None,
    lite: bool,
    output_json: bool,
    stream: bool = False,
    checkpoint: bool = False,
    checkpoint_dir: str = ".agentloom/checkpoints",
    mock_responses: Path | None = None,
    record: Path | None = None,
    mock_strict: bool = False,
    record_tools: bool = False,
    tool_mode: str = "live",
    live_from: str | None = None,
    force_live: list[str] | None = None,
) -> None:
    """Async implementation of the run command."""
    from agentloom.core.engine import WorkflowEngine
    from agentloom.providers.gateway import ProviderGateway

    _import_from_cwd()

    try:
        workflow = WorkflowParser.from_yaml(workflow_path)
    except Exception as e:
        typer.echo(f"Error loading workflow: {e}", err=True)
        raise typer.Exit(1)

    if provider_override:
        workflow.config.provider = provider_override
    if model_override:
        workflow.config.model = model_override
    if budget is not None:
        workflow.config.budget_usd = budget
    if stream:
        workflow.config.stream = True

    state_manager = StateManager(initial_state=_initial_state(workflow, state_args))

    gateway = ProviderGateway()
    # Setup observability (unless --lite) — done early so providers can hook into it
    observer = _setup_observer(lite)

    cut = None
    if mock_responses is not None:
        from agentloom.providers.mock import MockProvider

        live = live_from is not None or bool(force_live)  # a cut keeps the workflow's providers
        if not live:
            workflow.config.provider = "mock"
        try:
            mock = MockProvider(
                responses_file=mock_responses,
                observer=observer,
                workflow_name=workflow.name,
                strict=mock_strict,
            )
        except ValueError as e:
            # A malformed / wrong-version recording file is a startup
            # error — surface it cleanly instead of letting the workflow
            # run green against a fixture that never loaded.
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from e
        if live:
            cut = _setup_cut(gateway, workflow, mock, live_from, force_live or [])
            if record is not None:
                from agentloom.providers.recorder import RecordingProvider

                # Outermost, so a step's replayed and live turns land in one list, in order (#178).
                gateway.wrap_providers(lambda p: RecordingProvider(p, record, observer=observer))
        else:
            gateway.register(mock, priority=0)
    elif workflow.config.provider == "mock" and record is None:
        from agentloom.providers.mock import MockProvider

        gateway.register(
            MockProvider(
                responses_file=workflow.config.responses_file,
                latency_model=workflow.config.latency_model,
                latency_ms=workflow.config.latency_ms,
                observer=observer,
                workflow_name=workflow.name,
            ),
            priority=0,
        )
    else:
        _setup_live(gateway, workflow.config)
        if record is not None:
            from agentloom.providers.recorder import RecordingProvider

            gateway.wrap_providers(lambda p: RecordingProvider(p, record, observer=observer))

    tool_registry = _tool_registry(workflow)
    replayer = None
    if mock_responses is not None and tool_mode != "live":
        from agentloom.providers.tool_tape import ToolMode, ToolReplayer

        replayer = ToolReplayer(
            mock_responses,
            ToolMode(tool_mode),
            strict=mock_strict,
            live=cut.is_live if cut is not None else None,
        )
    if record is not None and record_tools:
        from agentloom.providers.tool_tape import ToolRecorder

        tool_registry.tape = ToolRecorder(record, serve_from=replayer)
    elif replayer is not None:
        tool_registry.tape = replayer

    stream_callback = None
    if stream and not output_json:

        def _on_chunk(step_id: str, text: str) -> None:
            typer.echo(text, nl=False)

        stream_callback = _on_chunk

    checkpointer = None
    if checkpoint:
        from agentloom.checkpointing.file import FileCheckpointer

        checkpointer = FileCheckpointer(checkpoint_dir=checkpoint_dir)

    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=state_manager,
        provider_gateway=gateway,
        tool_registry=tool_registry,
        observer=observer,
        on_stream_chunk=stream_callback,
        checkpointer=checkpointer,
    )

    typer.echo(f"Running workflow: {workflow.name}")
    # ``engine.run_id`` is always populated now (used for trace correlation),
    # but the user-facing "Run ID" echo stays gated on checkpoint mode — that's
    # the only context where the id is useful to the operator (resume from
    # checkpoint).
    if checkpointer is not None:
        typer.echo(f"Run ID: {engine.run_id}")
    events = EventLog() if record is not None else None  # #135
    with recording_events(events) if events is not None else nullcontext():
        result = await engine.run()
    if record is not None and events is not None:
        await anyio.to_thread.run_sync(_record_events, record, events)

    if stream and not output_json:
        typer.echo()  # Newline after streamed output

    if output_json and cut is not None:
        calls = [call.model_dump() for call in cut.calls]
        typer.echo(json.dumps({**result.model_dump(mode="json"), "calls": calls}, indent=2))
    elif output_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_result(result, run_id=engine.run_id)
        if cut is not None:
            typer.echo("Calls (replay up to the cut, live after it):")
            typer.echo("\n".join(cut.summary()))

    if observer:
        observer.shutdown()
    await gateway.close()

    if result.status not in (WorkflowStatus.SUCCESS, WorkflowStatus.PAUSED):
        raise typer.Exit(1)


def _setup_observer(lite: bool) -> WorkflowObserver | None:
    """Create the observability observer unless running in lite mode."""
    if lite:
        return None

    import os

    from agentloom.compat import is_available, try_import
    from agentloom.observability.observer import WorkflowObserver

    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    tracing_mod = try_import("opentelemetry.trace", extra="observability")
    metrics_mod = try_import("opentelemetry.sdk.metrics", extra="observability")

    tracing = None
    metrics = None

    if is_available(tracing_mod):
        from agentloom.observability.tracing import TracingManager

        tracing = TracingManager(endpoint=otel_endpoint)

    if is_available(metrics_mod):
        from agentloom.observability.metrics import MetricsManager

        metrics = MetricsManager(endpoint=otel_endpoint)

    if tracing or metrics:
        return WorkflowObserver(tracing=tracing, metrics=metrics)

    return None


def _setup_live(gateway: ProviderGateway, config: WorkflowConfig) -> None:
    """Register the workflow's own providers."""
    try:
        _setup_generic_openai(gateway, config)
    except ProviderError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    _setup_providers(gateway, config.provider)


def _setup_cut(
    gateway: ProviderGateway,
    workflow: WorkflowDefinition,
    recording: MockProvider,
    live_from: str | None,
    force_live: list[str],
) -> ReplayCut:
    """Serve the recording through the workflow's own providers, live from the cut on (#165)."""
    from agentloom.providers.replay_or_live import ReplayCut, ReplayOrLiveProvider, cut_steps

    try:
        cut = ReplayCut(recording, cut_steps(workflow, live_from, force_live))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    _setup_live(gateway, workflow.config)
    gateway.wrap_providers(lambda provider: ReplayOrLiveProvider(provider, cut))
    return cut


def _setup_generic_openai(gateway: ProviderGateway, config: WorkflowConfig) -> None:
    """Register ``provider: generic_openai`` from the workflow's own ``config:`` (#48)."""
    if config.provider != "generic_openai":
        return
    from agentloom.providers.generic_openai import GenericOpenAIProvider

    if not config.base_url:  # ``--provider generic_openai`` on a workflow without one
        raise ProviderError("generic_openai", "needs config.base_url in the workflow")
    api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
    if config.api_key_env and not api_key:
        raise ProviderError(
            "generic_openai", f"environment variable {config.api_key_env!r} is not set"
        )
    gateway.register(GenericOpenAIProvider(base_url=config.base_url, api_key=api_key), priority=0)


def _initial_state(workflow: WorkflowDefinition, state_args: list[str]) -> dict[str, Any]:
    """The workflow's state with the ``--state key=value`` pairs applied."""
    initial_state = dict(workflow.state)
    for item in state_args:
        if "=" not in item:
            typer.echo(f"Invalid state format '{item}'. Use key=value.", err=True)
            raise typer.Exit(1)
        key, _, raw = item.partition("=")
        initial_state[key] = _parse_state_value(raw)
    return initial_state


def _import_from_cwd() -> None:
    """Prepend the working directory so dotted imports from workflow YAML resolve."""
    import sys

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _tool_registry(workflow: WorkflowDefinition) -> ToolRegistry:
    """The built-in tools behind the workflow's sandbox."""
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry
    from agentloom.tools.sandbox import ToolSandbox

    sandbox_cfg = workflow.config.sandbox
    sandbox = ToolSandbox(
        enabled=sandbox_cfg.enabled,
        allowed_commands=sandbox_cfg.allowed_commands,
        allowed_paths=sandbox_cfg.allowed_paths,
        allow_network=sandbox_cfg.allow_network,
        readable_paths=sandbox_cfg.readable_paths,
        writable_paths=sandbox_cfg.writable_paths,
        allowed_domains=sandbox_cfg.allowed_domains,
        allowed_schemes=sandbox_cfg.allowed_schemes,
        max_write_bytes=sandbox_cfg.max_write_bytes,
        danger_opt_in=sandbox_cfg.danger_opt_in,
        allow_internal_webhook_targets=sandbox_cfg.allow_internal_webhook_targets,
    )
    tool_registry = ToolRegistry()
    register_builtins(tool_registry, sandbox=sandbox)
    return tool_registry


def _record_events(path: Path, events: EventLog) -> None:
    """Keep the run's inter-agent events in its recording (#135)."""
    from agentloom.providers.recorder import merge_into_recording

    logged = dump_events(events.events())
    merge_into_recording(path, lambda recording: recording.__setitem__(EVENTS_KEY, logged))


def _setup_providers(gateway: ProviderGateway, default_provider: str) -> None:
    """Setup providers from config-driven discovery."""
    # The endpoint is the only route: a discovered provider would take mapped models or fallbacks
    if default_provider == "generic_openai":
        return

    import importlib

    from agentloom.config import load_config

    config = load_config(default_provider_override=default_provider)

    _PROVIDER_CLASSES: dict[str, tuple[str, str]] = {
        "openai": ("agentloom.providers.openai", "OpenAIProvider"),
        "anthropic": ("agentloom.providers.anthropic", "AnthropicProvider"),
        "google": ("agentloom.providers.google", "GoogleProvider"),
        "ollama": ("agentloom.providers.ollama", "OllamaProvider"),
    }

    for pc in config.providers:
        entry = _PROVIDER_CLASSES.get(pc.name)
        if entry is None:
            continue
        mod_path, cls_name = entry
        mod = importlib.import_module(mod_path)
        provider_cls = getattr(mod, cls_name)

        kwargs: dict[str, object] = {}
        if pc.api_key:
            kwargs["api_key"] = pc.api_key
        if pc.base_url:
            kwargs["base_url"] = pc.base_url

        gateway.register(
            provider_cls(**kwargs),
            priority=pc.priority,
            is_fallback=pc.is_fallback,
            models=pc.models if pc.models else None,
        )


def _print_result(result: object, *, run_id: str | None = None) -> None:
    """Pretty-print a workflow result."""
    from agentloom.core.results import StepStatus, WorkflowResult

    r: WorkflowResult = result  # type: ignore[assignment]

    status_icon = {
        "success": "[OK]",
        "failed": "[FAIL]",
        "timeout": "[TIMEOUT]",
        "budget_exceeded": "[BUDGET]",
        "paused": "[PAUSED]",
    }

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Workflow: {r.workflow_name}")
    typer.echo(f"Status:   {status_icon.get(r.status.value, '?')} {r.status.value}")
    typer.echo(f"Duration: {r.total_duration_ms:.1f}ms")
    typer.echo(f"Tokens:   {r.total_tokens}")
    typer.echo(f"Cost:     ${r.total_cost_usd:.4f}")

    if r.error:
        typer.echo(f"Error:    {r.error}")

    typer.echo("\nSteps:")
    for step_id, sr in r.step_results.items():
        step_icons = {
            StepStatus.SUCCESS: "[OK]",
            StepStatus.SKIPPED: "[SKIP]",
            StepStatus.PAUSED: "[PAUSED]",
        }
        icon = step_icons.get(sr.status, "[FAIL]")
        line = f"  {icon} {step_id} ({sr.duration_ms:.0f}ms)"
        if sr.cost_usd > 0:
            line += f" ${sr.cost_usd:.4f}"
        if sr.attachment_count > 0:
            line += f" [{sr.attachment_count} attachment{'s' if sr.attachment_count > 1 else ''}]"
        typer.echo(line)

    final_state = r.final_state
    typer.echo(f"\nFinal State Keys: {list(final_state.keys())}")

    # Render pending approvals after the result table so concurrent gates
    # don't interleave on stdout. Each PAUSED step produces one block; the
    # run_id is what the operator types into `agentloom resume`.
    paused_steps = [
        step_id for step_id, sr in r.step_results.items() if sr.status == StepStatus.PAUSED
    ]
    if paused_steps:
        typer.echo("")
        run_label = run_id or "<run_id>"
        for step_id in paused_steps:
            # The hint must use only flags ``agentloom resume`` actually
            # accepts: it resolves the paused step from the checkpoint,
            # so there is no ``--step`` option. Pre-0.5.0 the hint
            # printed ``--step <id>`` verbatim and a user who copied it
            # hit ``Error: No such option: --step``.
            typer.echo(f"[APPROVAL REQUIRED] Step '{step_id}' is waiting for a decision.")
            typer.echo(f"  Approve: agentloom resume {run_label} --approve")
            typer.echo(f"  Reject:  agentloom resume {run_label} --reject")

    typer.echo(f"{'=' * 60}")
