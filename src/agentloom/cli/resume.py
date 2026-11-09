"""CLI command: resume a checkpointed workflow run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import typer

from agentloom.core.results import WorkflowStatus


def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume."),
    checkpoint_dir: str = typer.Option(
        ".agentloom/checkpoints", "--checkpoint-dir", help="Checkpoint storage directory."
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Override default provider."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override default model."),
    state: list[str] = typer.Option(
        [],
        "--state",
        "-s",
        help="State variables as key=value pairs, such as a key redacted from the checkpoint.",
    ),
    budget: float | None = typer.Option(
        None, "--budget", "-b", help="Maximum budget in USD, what the run has spent included."
    ),
    approve: bool = typer.Option(False, "--approve", help="Approve a pending approval gate."),
    reject: bool = typer.Option(False, "--reject", help="Reject a pending approval gate."),
    lite: bool = typer.Option(False, "--lite", help="Run in lite mode (no observability)."),
    output_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM output in real-time."),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Config file with a providers: block. Default: $AGENTLOOM_CONFIG, "
            "then ./agentloom.yaml when it exists."
        ),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Resume a paused or failed workflow from its last checkpoint."""
    if approve and reject:
        typer.echo("Cannot use --approve and --reject together.", err=True)
        raise typer.Exit(2)

    from agentloom.cli.run import _config_file, _sigterm_as_interrupt, _state_overrides

    state_overrides = _state_overrides(state)
    decision: str | None = None
    if approve:
        decision = "approved"
    elif reject:
        decision = "rejected"

    with _sigterm_as_interrupt():
        anyio.run(
            _resume_async,
            run_id,
            checkpoint_dir,
            provider,
            model,
            decision,
            lite,
            output_json,
            stream,
            state_overrides,
            budget,
            _config_file(config_file),
        )


async def _resume_async(
    run_id: str,
    checkpoint_dir: str,
    provider_override: str | None,
    model_override: str | None,
    decision: str | None,
    lite: bool,
    output_json: bool,
    stream: bool,
    state_overrides: dict[str, Any],
    budget: float | None,
    config_path: Path | None = None,
) -> None:
    from agentloom.checkpointing.file import FileCheckpointer
    from agentloom.cli.run import (
        _import_from_cwd,
        _print_result,
        _setup_live,
        _setup_mock,
        _setup_observer,
        _switch_provider,
    )
    from agentloom.core.engine import WorkflowEngine
    from agentloom.providers.gateway import ProviderGateway
    from agentloom.steps import approval_gate
    from agentloom.tools.builtins import register_builtins
    from agentloom.tools.registry import ToolRegistry
    from agentloom.tools.sandbox import ToolSandbox

    _import_from_cwd()  # as ``run``: a response_schema model of the project must import
    checkpointer = FileCheckpointer(checkpoint_dir=checkpoint_dir)
    try:
        checkpoint_data = await checkpointer.load(run_id)
    except KeyError:
        typer.echo(f"No checkpoint found for run '{run_id}'.", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    # ``run.step`` is a subworkflow's own checkpoint: resuming it would strand its parent
    if "." in run_id:
        root = run_id.partition(".")[0]
        typer.echo(f"Run '{run_id}' is part of run '{root}': resume '{root}'.", err=True)
        raise typer.Exit(2)

    approval_decisions: dict[str, str] = {}
    if decision:
        if checkpoint_data.status != "paused" or not checkpoint_data.paused_step_id:
            typer.echo(
                f"Run '{run_id}' is not paused at an approval gate.",
                err=True,
            )
            raise typer.Exit(1)
        approval_decisions[checkpoint_data.paused_step_id] = decision

    if decision and checkpoint_data.gate_expired(approval_gate._now()):
        flag = "--approve" if decision == "approved" else "--reject"
        typer.echo(
            f"The gate expired at {checkpoint_data.expires_at}: {flag} is ignored. Its "
            f"on_timeout decides, and without one the run fails with timeout.",
            err=True,
        )

    # On stderr with --json, so stdout parses whole.
    if decision:
        typer.echo(
            f"Resuming workflow '{checkpoint_data.workflow_name}' "
            f"(run {run_id}, decision={decision} for step '{checkpoint_data.paused_step_id}')",
            err=output_json,
        )
    else:
        typer.echo(
            f"Resuming workflow '{checkpoint_data.workflow_name}' "
            f"(run {run_id}, status={checkpoint_data.status})",
            err=output_json,
        )

    # In the checkpoint, not on the engine: it sets up its budget from the definition.
    checkpoint_data.state.update(state_overrides)
    if budget is not None:
        checkpoint_data.workflow_definition.setdefault("config", {})["budget_usd"] = budget

    engine = await WorkflowEngine.from_checkpoint(
        checkpoint_data=checkpoint_data,
        checkpointer=checkpointer,
        approval_decisions=approval_decisions or None,
    )

    # The model goes first, as in ``run``.
    if model_override:
        engine.workflow.config.model = model_override
    if provider_override:
        _switch_provider(engine.workflow, provider_override)
    if stream:
        engine.workflow.config.stream = True

    observer = _setup_observer(lite)
    gateway = ProviderGateway()
    if engine.workflow.config.provider == "mock":
        _setup_mock(gateway, engine.workflow.config, observer, engine.workflow.name)
    else:
        _setup_live(gateway, engine.workflow.config, config_path)
    engine.provider_gateway = gateway

    sandbox_cfg = engine.workflow.config.sandbox
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
        deny_internal_hosts=sandbox_cfg.deny_internal_hosts,
    )
    tool_registry = ToolRegistry()
    register_builtins(tool_registry, sandbox=sandbox)
    engine.tool_registry = tool_registry

    engine.observer = observer
    if observer and gateway:  # pragma: no cover — requires OTel extra
        set_obs = getattr(gateway, "set_observer", None)
        if set_obs:
            set_obs(observer)

    if stream and not output_json:

        def _on_chunk(step_id: str, text: str) -> None:
            typer.echo(text, nl=False)

        engine._stream_callback = _on_chunk

    result = await engine.run()

    if stream and not output_json:
        typer.echo()

    if output_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_result(result, run_id=run_id)

    if observer:  # pragma: no cover — requires OTel extra
        observer.shutdown()
    await gateway.close()

    if result.status not in (WorkflowStatus.SUCCESS, WorkflowStatus.PAUSED):
        raise typer.Exit(1)
