"""Tests for CLI runs command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentloom.checkpointing.base import CheckpointData
from agentloom.cli.main import app

runner = CliRunner()


def _write_checkpoint(cp_dir: Path, run_id: str, workflow_name: str = "wf") -> None:
    """Synchronous helper to write a checkpoint file for testing."""
    data = CheckpointData(
        workflow_name=workflow_name,
        run_id=run_id,
        workflow_definition={"name": workflow_name, "steps": []},
        state={},
        completed_steps=[],
        status="completed",
        created_at="2026-04-12T10:00:00+00:00",
        updated_at="2026-04-12T10:00:01+00:00",
    )
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / f"{run_id}.json").write_text(data.model_dump_json(indent=2))


class TestRunsCommand:
    def test_no_checkpoints(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["runs", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No checkpoint runs found" in result.output

    def test_lists_runs(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run1", "workflow-a")
        _write_checkpoint(tmp_path, "run2", "workflow-b")
        result = runner.invoke(app, ["runs", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "run1" in result.output
        assert "run2" in result.output
        assert "workflow-a" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run1", "my-wf")
        result = runner.invoke(app, ["runs", "--checkpoint-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["run_id"] == "run1"
