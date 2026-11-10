"""Tests for CLI info command."""

from __future__ import annotations

from typing import Any

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

    def test_shows_jsonschema_with_its_version(self) -> None:
        from importlib.metadata import version

        result = runner.invoke(app, ["info"])
        assert f"jsonschema   {version('jsonschema')}" in result.output

    def test_shows_observability_status(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "opentelemetry:" in result.output
        assert "prometheus:" in result.output

    def test_shows_provider_status(self) -> None:
        result = runner.invoke(app, ["info"])
        assert "OpenAI" in result.output
        assert "Ollama" in result.output

    def test_api_keys_show_whether_set_and_nothing_of_the_value(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijkl")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["info"])
        assert "sk-proj" not in result.output
        assert "OpenAI       configured\n" in result.output
        assert "Anthropic    not configured" in result.output

    def test_ollama_url_hides_its_credentials(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://user:pa55@ollama.example:11434/api")
        result = runner.invoke(app, ["info"])
        assert "pa55" not in result.output
        assert "user" not in result.output
        assert "https://***@ollama.example:11434/api" in result.output
