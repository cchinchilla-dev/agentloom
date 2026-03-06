"""Main CLI application for AgentLoom."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="agentloom",
    help="AgentLoom: Production-ready agentic workflow orchestrator.",
    no_args_is_help=True,
)

from agentloom.cli.run import run  # noqa: E402
from agentloom.cli.validate import validate  # noqa: E402

app.command("run")(run)
app.command("validate")(validate)

if __name__ == "__main__":
    app()
