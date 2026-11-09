"""Main CLI application for AgentLoom."""

from __future__ import annotations

import os
from enum import StrEnum

import typer

from agentloom import __version__

app = typer.Typer(
    name="agentloom",
    help="AgentLoom: Production-ready agentic workflow orchestrator.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

from agentloom.cli.agent import agent_app  # noqa: E402
from agentloom.cli.callback_server import callback_server  # noqa: E402
from agentloom.cli.history import history  # noqa: E402
from agentloom.cli.info import info  # noqa: E402
from agentloom.cli.replay import replay  # noqa: E402
from agentloom.cli.resume import resume  # noqa: E402
from agentloom.cli.run import run  # noqa: E402
from agentloom.cli.runs import runs  # noqa: E402
from agentloom.cli.validate import validate  # noqa: E402
from agentloom.cli.verify import verify_determinism  # noqa: E402
from agentloom.cli.visualize import visualize  # noqa: E402


class LogLevel(StrEnum):
    """Supported ``--log-level`` values."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _print_version(show: bool) -> None:
    if show:
        typer.echo(f"agentloom {__version__}")
        raise typer.Exit()


def _configure_logging(level: LogLevel | None) -> None:
    """Send the ``agentloom`` logs to stderr as configured; with nothing set, leave them alone."""
    log_format = os.environ.get("AGENTLOOM_LOG_FORMAT") or None
    if log_format not in (None, "json", "text"):
        raise typer.BadParameter(f"AGENTLOOM_LOG_FORMAT must be json or text, not {log_format!r}.")
    if level is None and log_format is None:
        return

    from agentloom.observability.logging import setup_logging

    setup_logging(level=level or LogLevel.INFO, format=log_format or "json")


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True, help="Print the version."
    ),
    log_level: LogLevel | None = typer.Option(
        None,
        "--log-level",
        envvar="AGENTLOOM_LOG_LEVEL",
        case_sensitive=False,
        help="Level of the agentloom logs, on stderr.",
    ),
) -> None:
    _configure_logging(log_level)


app.command("run")(run)
app.command("replay")(replay)
app.command("resume")(resume)
app.command("runs")(runs)
app.command("history")(history)
app.command("validate")(validate)
app.command("verify-determinism")(verify_determinism)
app.command("visualize")(visualize)
app.command("info")(info)
app.command("callback-server")(callback_server)
app.add_typer(agent_app, name="agent")


if __name__ == "__main__":
    app()
