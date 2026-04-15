"""CLI command: run a workflow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import typer

from agentloom.core.parser import WorkflowParser
from agentloom.core.results import WorkflowStatus
from agentloom.core.state import StateManager

if TYPE_CHECKING:
    from agentloom.observability.observer import WorkflowObserver
    from agentloom.providers.gateway import ProviderGateway


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
) -> None:
    """Execute a workflow from a YAML definition file."""
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
) -> None:
    """Async implementation of the run command."""
    from agentloom.core.engine import WorkflowEngine
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry
    from agentloom.tools.sandbox import ToolSandbox

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

    initial_state = dict(workflow.state)
    for item in state_args:
        if "=" not in item:
            typer.echo(f"Invalid state format '{item}'. Use key=value.", err=True)
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        initial_state[key] = value

    state_manager = StateManager(initial_state=initial_state)

    gateway = ProviderGateway()
    _setup_providers(gateway, workflow.config.provider)

    sandbox_cfg = workflow.config.sandbox
    sandbox = ToolSandbox(
        enabled=sandbox_cfg.enabled,
        allowed_commands=sandbox_cfg.allowed_commands,
        allowed_paths=sandbox_cfg.allowed_paths,
        allow_network=sandbox_cfg.allow_network,
        readable_paths=sandbox_cfg.readable_paths,
        writable_paths=sandbox_cfg.writable_paths,
        allowed_domains=sandbox_cfg.allowed_domains,
        max_write_bytes=sandbox_cfg.max_write_bytes,
    )
    tool_registry = ToolRegistry()
    register_builtins(tool_registry, sandbox=sandbox)

    observer = _setup_observer(lite)

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
    if engine.run_id:
        typer.echo(f"Run ID: {engine.run_id}")
    result = await engine.run()

    if stream and not output_json:
        typer.echo()  # Newline after streamed output

    if output_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_result(result)

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


def _setup_providers(gateway: ProviderGateway, default_provider: str) -> None:
    """Setup providers from config-driven discovery."""
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


def _print_result(result: object) -> None:
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
    typer.echo(f"{'=' * 60}")
