"""Tests for the root ``agentloom`` options and the ``--json`` contract of every command."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentloom import __version__
from agentloom.cli.main import app

runner = CliRunner()

OK_YAML = """\
name: ok
config: {provider: mock, model: m}
steps:
  - {id: answer, type: llm_call, prompt: hello, output: answer}
"""

FAILING_YAML = """\
name: bad
config: {provider: mock, model: m}
steps:
  - {id: use, type: tool, tool_name: no_such_tool, output: x}
"""

AGENT_YAML = """\
name: helper
system_prompt: You help.
"""

COMMANDS = [
    "run",
    "replay",
    "resume",
    "runs",
    "history",
    "validate",
    "verify-determinism",
    "visualize",
    "info",
    "callback-server",
]


@pytest.fixture
def agentloom_logger() -> Iterator[logging.Logger]:
    """The ``agentloom`` logger, put back as it was: ``setup_logging`` changes it process-wide."""
    logger = logging.getLogger("agentloom")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture(autouse=True)
def _no_ambient_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLOOM_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AGENTLOOM_LOG_FORMAT", raising=False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _pure_json(result: Any) -> Any:
    """The stdout of *result*, which must parse whole: any other line would break ``json.loads``."""
    return json.loads(result.stdout)


class TestRootOptions:
    def test_version_prints_the_version_and_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.stdout.strip() == f"agentloom {__version__}"

    def test_short_help_on_the_root(self) -> None:
        result = runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        assert "Usage" in result.stdout

    @pytest.mark.parametrize("command", COMMANDS)
    def test_short_help_on_each_command(self, command: str) -> None:
        result = runner.invoke(app, [command, "-h"])
        assert result.exit_code == 0, result.output
        assert "Usage" in result.stdout

    @pytest.mark.parametrize("argv", [["agent", "-h"], ["agent", "inspect", "-h"]])
    def test_short_help_on_the_nested_commands(self, argv: list[str]) -> None:
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        assert "Usage" in result.stdout


class TestLogging:
    def test_without_settings_the_cli_configures_no_logging(
        self, tmp_path: Path, agentloom_logger: logging.Logger
    ) -> None:
        agentloom_logger.handlers.clear()
        wf = _write(tmp_path, "wf.yaml", OK_YAML)
        result = runner.invoke(app, ["run", wf, "--lite"])
        assert result.exit_code == 0, result.output
        assert agentloom_logger.handlers == []

    def test_log_level_option_shows_the_lower_levels_on_stderr_only(
        self, tmp_path: Path, agentloom_logger: logging.Logger
    ) -> None:
        wf = _write(tmp_path, "wf.yaml", OK_YAML)
        result = runner.invoke(app, ["--log-level", "INFO", "run", wf, "--lite", "--json"])
        assert result.exit_code == 0, result.output
        assert "Starting workflow" in result.stderr
        assert _pure_json(result)["status"] == "success"

    def test_log_level_env_var_is_honoured(
        self, tmp_path: Path, agentloom_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_LOG_LEVEL", "DEBUG")
        wf = _write(tmp_path, "wf.yaml", OK_YAML)
        result = runner.invoke(app, ["run", wf, "--lite"])
        assert result.exit_code == 0, result.output
        assert agentloom_logger.level == logging.DEBUG

    def test_the_option_overrides_the_env_var(
        self, tmp_path: Path, agentloom_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_LOG_LEVEL", "DEBUG")
        wf = _write(tmp_path, "wf.yaml", OK_YAML)
        result = runner.invoke(app, ["--log-level", "error", "run", wf, "--lite"])
        assert result.exit_code == 0, result.output
        assert agentloom_logger.level == logging.ERROR

    def test_log_format_env_var_selects_text(
        self, tmp_path: Path, agentloom_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_LOG_FORMAT", "text")
        wf = _write(tmp_path, "wf.yaml", OK_YAML)
        result = runner.invoke(app, ["--log-level", "INFO", "run", wf, "--lite"])
        assert result.exit_code == 0, result.output
        assert "[INFO ] agentloom" in result.stderr

    def test_records_are_json_lines_by_default(
        self, tmp_path: Path, agentloom_logger: logging.Logger
    ) -> None:
        wf = _write(tmp_path, "wf.yaml", OK_YAML)
        result = runner.invoke(app, ["--log-level", "INFO", "run", wf, "--lite"])
        first = next(line for line in result.stderr.splitlines() if line.startswith("{"))
        assert json.loads(first)["level"] == "INFO"

    def test_an_unknown_level_is_a_usage_error(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["--log-level", "loud", "info"])
        assert result.exit_code == 2

    def test_an_unknown_level_in_the_environment_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_LOG_LEVEL", "loud")
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 2
        assert "AGENTLOOM_LOG_LEVEL" in result.stderr

    def test_an_unknown_format_in_the_environment_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_LOG_FORMAT", "yaml")
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 2
        assert "AGENTLOOM_LOG_FORMAT" in result.stderr


def _checkpoint(tmp_path: Path, wf: str) -> str:
    """Run *wf* with a checkpoint and return its run id, read from the JSON result."""
    result = runner.invoke(
        app,
        ["run", wf, "--lite", "--json", "--checkpoint", "--checkpoint-dir", str(tmp_path / "cp")],
    )
    run_id: str = _pure_json(result)["run_id"]
    return run_id


class TestJsonIsPure:
    """``--json`` leaves stdout parseable by ``json.loads``, whatever the run did."""

    def test_run_success(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["run", _write(tmp_path, "w.yaml", OK_YAML), "--lite", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert _pure_json(result)["status"] == "success"
        assert "Running workflow: ok" in result.stderr

    def test_run_failure(self, tmp_path: Path) -> None:
        wf = _write(tmp_path, "w.yaml", FAILING_YAML)
        result = runner.invoke(app, ["run", wf, "--lite", "--json"])
        assert result.exit_code == 1
        assert _pure_json(result)["status"] == "failed"

    def test_run_with_a_checkpoint_keeps_the_run_id_off_stdout(self, tmp_path: Path) -> None:
        wf = _write(tmp_path, "w.yaml", OK_YAML)
        result = runner.invoke(
            app,
            ["run", wf, "--lite", "--json", "--checkpoint", "--checkpoint-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        run_id = _pure_json(result)["run_id"]
        assert f"Run ID: {run_id}" in result.stderr

    def test_run_that_cannot_load_leaves_stdout_empty(self, tmp_path: Path) -> None:
        wf = _write(tmp_path, "w.yaml", "name: [broken\n")
        result = runner.invoke(app, ["run", wf, "--lite", "--json"])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Error loading workflow" in result.stderr

    def test_replay_success(self, tmp_path: Path) -> None:
        wf = _write(tmp_path, "w.yaml", OK_YAML)
        rec = _write(tmp_path, "r.json", json.dumps({"answer": {"content": "hi"}}))
        result = runner.invoke(app, ["replay", wf, "--recording", rec, "--json"])
        assert result.exit_code == 0, result.output
        assert _pure_json(result)["status"] == "success"

    def test_replay_failure(self, tmp_path: Path) -> None:
        wf = _write(tmp_path, "w.yaml", OK_YAML)
        rec = _write(tmp_path, "r.json", json.dumps({"other": {"content": "hi"}}))
        result = runner.invoke(app, ["replay", wf, "--recording", rec, "--json"])
        assert result.exit_code == 1
        assert _pure_json(result)["status"] == "failed"

    def test_resume_success(self, tmp_path: Path) -> None:
        run_id = _checkpoint(tmp_path, _write(tmp_path, "w.yaml", OK_YAML))
        result = runner.invoke(
            app, ["resume", run_id, "--checkpoint-dir", str(tmp_path / "cp"), "--lite", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert _pure_json(result)["status"] == "success"
        assert "Resuming workflow" in result.stderr

    def test_resume_failure(self, tmp_path: Path) -> None:
        run_id = _checkpoint(tmp_path, _write(tmp_path, "w.yaml", FAILING_YAML))
        result = runner.invoke(
            app, ["resume", run_id, "--checkpoint-dir", str(tmp_path / "cp"), "--lite", "--json"]
        )
        assert result.exit_code == 1
        assert _pure_json(result)["status"] == "failed"

    def test_resume_of_an_unknown_run_leaves_stdout_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["resume", "nope", "--checkpoint-dir", str(tmp_path), "--lite", "--json"]
        )
        assert result.exit_code == 1
        assert result.stdout == ""

    def test_runs_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["runs", "--checkpoint-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        assert _pure_json(result) == []

    def test_runs_with_entries(self, tmp_path: Path) -> None:
        run_id = _checkpoint(tmp_path, _write(tmp_path, "w.yaml", OK_YAML))
        result = runner.invoke(app, ["runs", "--checkpoint-dir", str(tmp_path / "cp"), "--json"])
        assert result.exit_code == 0
        assert [entry["run_id"] for entry in _pure_json(result)] == [run_id]

    def test_history_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        assert _pure_json(result) == []

    def test_history_with_records(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text(json.dumps({"run_id": "a", "workflow_name": "x"}))
        result = runner.invoke(app, ["history", "--runs-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        assert [record["run_id"] for record in _pure_json(result)] == ["a"]

    def test_history_with_a_bad_date_leaves_stdout_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["history", "--runs-dir", str(tmp_path), "--since", "soon", "--json"]
        )
        assert result.exit_code == 2
        assert result.stdout == ""

    def test_agent_inspect_success(self, tmp_path: Path) -> None:
        agent = _write(tmp_path, "agent.yaml", AGENT_YAML)
        result = runner.invoke(app, ["agent", "inspect", agent, "--json"])
        assert result.exit_code == 0, result.output
        assert _pure_json(result)["name"] == "helper"

    def test_agent_inspect_failure_leaves_stdout_empty(self, tmp_path: Path) -> None:
        agent = _write(tmp_path, "agent.yaml", "name: [broken\n")
        result = runner.invoke(app, ["agent", "inspect", agent, "--json"])
        assert result.exit_code == 1
        assert result.stdout == ""
