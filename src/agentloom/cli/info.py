"""CLI command: show system information."""

from __future__ import annotations

import os
import platform
import sys

import typer

from agentloom import __version__
from agentloom.compat import is_available, try_import


def info() -> None:
    """Show AgentLoom version, dependencies, and system information."""
    typer.echo(f"AgentLoom v{__version__}")
    typer.echo(f"{'=' * 40}")

    # Python info
    typer.echo(f"\nPython:    {sys.version.split()[0]}")
    typer.echo(f"Platform:  {platform.platform()}")
    typer.echo(f"Arch:      {platform.machine()}")

    # Core dependencies
    typer.echo("\nCore dependencies:")
    for pkg in ["pydantic", "httpx", "pyyaml", "typer", "anyio"]:
        _show_dep(pkg)

    # Optional dependencies
    typer.echo("\nOptional (observability):")
    otel = try_import("opentelemetry.sdk", extra="observability")
    prom = try_import("prometheus_client", extra="observability")
    typer.echo(f"  opentelemetry: {'installed' if is_available(otel) else 'not installed'}")
    typer.echo(f"  prometheus:    {'installed' if is_available(prom) else 'not installed'}")

    # Provider API keys
    typer.echo("\nProviders:")
    providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "Ollama": "OLLAMA_BASE_URL",
    }
    for name, env_var in providers.items():
        value = os.environ.get(env_var, "")
        if env_var == "OLLAMA_BASE_URL":
            status = value or "http://localhost:11434 (default)"
        elif value:
            status = f"configured ({value[:8]}...)"
        else:
            status = "not configured"
        typer.echo(f"  {name:12s} {status}")

    typer.echo("")


def _show_dep(package: str) -> None:
    """Show version of an installed package."""
    try:
        from importlib.metadata import version

        ver = version(package.replace("-", "_"))
        typer.echo(f"  {package:12s} {ver}")
    except Exception:
        typer.echo(f"  {package:12s} not found")
