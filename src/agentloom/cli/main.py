"""Main CLI application for AgentLoom."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="agentloom",
    help="AgentLoom: Production-ready agentic workflow orchestrator.",
    no_args_is_help=True,
)

from agentloom.cli.info import info  # noqa: E402
from agentloom.cli.resume import resume  # noqa: E402
from agentloom.cli.run import run  # noqa: E402
from agentloom.cli.runs import runs  # noqa: E402
from agentloom.cli.validate import validate  # noqa: E402
from agentloom.cli.visualize import visualize  # noqa: E402

app.command("run")(run)
app.command("resume")(resume)
app.command("runs")(runs)
app.command("validate")(validate)
app.command("visualize")(visualize)
app.command("info")(info)


if __name__ == "__main__":
    app()
