"""Tests for CLI resume command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentloom.cli.main import app

runner = CliRunner()


class TestResumeCommand:
    def test_missing_run_id(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["resume", "nonexistent", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "No checkpoint found" in (result.output + (result.stderr or ""))
