"""Tests for CLI info command."""

from __future__ import annotations

from typer.testing import CliRunner

from agentloom.cli.main import app

runner = CliRunner()


class TestInfoCommand:
    def test_shows_version(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "AgentLoom v" in result.output

    def test_shows_python_version(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "Python:" in result.output

    def test_shows_core_dependencies(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "pydantic" in result.output
        assert "httpx" in result.output
        assert "anyio" in result.output

    def test_shows_observability_status(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "opentelemetry:" in result.output
        assert "prometheus:" in result.output

    def test_shows_provider_status(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "OpenAI" in result.output
        assert "Ollama" in result.output
