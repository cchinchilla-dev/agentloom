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
) -> None:
    """Execute a workflow from a YAML definition file."""
    anyio.run(_run_async, workflow_path, state, provider, model, budget, lite, output_json)


async def _run_async(
    workflow_path: Path,
    state_args: list[str],
    provider_override: str | None,
    model_override: str | None,
    budget: float | None,
    lite: bool,
    output_json: bool,
) -> None:
    """Async implementation of the run command."""
    from agentloom.core.engine import WorkflowEngine
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry

    # Parse workflow
    try:
        workflow = WorkflowParser.from_yaml(workflow_path)
    except Exception as e:
        typer.echo(f"Error loading workflow: {e}", err=True)
        raise typer.Exit(1)

    # Apply overrides
    if provider_override:
        workflow.config.provider = provider_override
    if model_override:
        workflow.config.model = model_override
    if budget is not None:
        workflow.config.budget_usd = budget

    # Parse state overrides
    initial_state = dict(workflow.state)
    for item in state_args:
        if "=" not in item:
            typer.echo(f"Invalid state format '{item}'. Use key=value.", err=True)
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        initial_state[key] = value

    state_manager = StateManager(initial_state=initial_state)

    # Setup provider gateway
    gateway = ProviderGateway()
    _setup_providers(gateway, workflow.config.provider)

    # Setup tools
    tool_registry = ToolRegistry()
    register_builtins(tool_registry)

    # Setup observability (unless --lite)
    observer = _setup_observer(lite)

    # Run engine
    engine = WorkflowEngine(
        workflow=workflow,
        state_manager=state_manager,
        provider_gateway=gateway,
        tool_registry=tool_registry,
        observer=observer,
    )

    typer.echo(f"Running workflow: {workflow.name}")
    result = await engine.run()

    # Output results
    if output_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_result(result)

    if observer:
        observer.shutdown()
    await gateway.close()

    if result.status != WorkflowStatus.SUCCESS:
        raise typer.Exit(1)


def _setup_observer(lite: bool) -> WorkflowObserver | None:
    """Create the observability observer unless running in lite mode."""
    if lite:
        return None

    from agentloom.compat import is_available, try_import
    from agentloom.observability.observer import WorkflowObserver

    tracing_mod = try_import("opentelemetry.trace", extra="observability")
    metrics_mod = try_import("opentelemetry.sdk.metrics", extra="observability")

    tracing = None
    metrics = None

    if is_available(tracing_mod):
        from agentloom.observability.tracing import TracingManager

        tracing = TracingManager()

    if is_available(metrics_mod):
        from agentloom.observability.metrics import MetricsManager

        metrics = MetricsManager()

    if tracing or metrics:
        return WorkflowObserver(tracing=tracing, metrics=metrics)

    return None


def _setup_providers(gateway: ProviderGateway, default_provider: str) -> None:
    """Setup providers based on available API keys."""
    # HACK: provider discovery from env vars — should really be in a config file
    import os

    if os.environ.get("OPENAI_API_KEY"):
        from agentloom.providers.openai import OpenAIProvider

        gateway.register(
            OpenAIProvider(),
            priority=0 if default_provider == "openai" else 10,
            models=[
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1",
                "o4-mini",
            ],
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        from agentloom.providers.anthropic import AnthropicProvider

        gateway.register(
            AnthropicProvider(),
            priority=0 if default_provider == "anthropic" else 10,
            models=[
                "claude-haiku-4-5-20251001",
            ],
        )

    if os.environ.get("GOOGLE_API_KEY"):
        from agentloom.providers.google import GoogleProvider

        gateway.register(
            GoogleProvider(),
            priority=0 if default_provider == "google" else 10,
            models=[
                "gemini-2.0-flash",
                "gemini-2.5-flash",
            ],
        )

    # Ollama is always available as fallback (local/LAN)
    from agentloom.providers.ollama import OllamaProvider

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    gateway.register(
        OllamaProvider(base_url=ollama_url),
        priority=0 if default_provider == "ollama" else 100,
        is_fallback=True,
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
        icon = (
            "[OK]"
            if sr.status == StepStatus.SUCCESS
            else ("[SKIP]" if sr.status == StepStatus.SKIPPED else "[FAIL]")
        )
        line = f"  {icon} {step_id} ({sr.duration_ms:.0f}ms)"
        if sr.cost_usd > 0:
            line += f" ${sr.cost_usd:.4f}"
        typer.echo(line)

    # Show final output
    final_state = r.final_state
    typer.echo(f"\nFinal State Keys: {list(final_state.keys())}")
    typer.echo(f"{'=' * 60}")
