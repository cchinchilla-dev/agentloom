"""Tests for CLI replay command and YAML-configured MockProvider."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from agentloom.cli.main import app

runner = CliRunner()

SIMPLE_YAML = """\
name: replay-test
config:
  provider: mock
  model: mock-model
state:
  question: "What is Python?"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer: {state.question}"
    output: answer
"""

YAML_MOCK_CONFIG = """\
name: yaml-mock-test
config:
  provider: mock
  model: mock-model
  responses_file: "{responses_file}"
  latency_model: constant
  latency_ms: 0
state:
  question: "What is Python?"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer: {{state.question}}"
    output: answer
"""

RECORDING = {
    "answer": {
        "content": "42",
        "model": "mock-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "finish_reason": "stop",
    }
}


class TestReplayCommand:
    def test_replay_invokes_mock_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "rec.json"
            rec_path.write_text(json.dumps(RECORDING))
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(SIMPLE_YAML)
            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    ["replay", str(yaml_path), "--recording", str(rec_path)],
                )
        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "'answer'" in result.output

    def test_replay_requires_existing_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(SIMPLE_YAML)
            result = runner.invoke(
                app,
                ["replay", str(yaml_path), "--recording", "/tmp/does_not_exist_xyz.json"],
            )
        assert result.exit_code != 0


class TestYamlMockProvider:
    def test_workflow_config_defaults(self) -> None:
        from agentloom.core.models import WorkflowConfig

        cfg = WorkflowConfig()
        assert cfg.responses_file is None
        assert cfg.latency_model == "constant"
        assert cfg.latency_ms == 0.0

    def test_workflow_config_rejects_bad_latency_model(self) -> None:
        import pytest
        from pydantic import ValidationError

        from agentloom.core.models import WorkflowConfig

        with pytest.raises(ValidationError):
            WorkflowConfig(latency_model="bogus")  # type: ignore[arg-type]

    def test_yaml_latency_fields_reach_provider(self) -> None:
        from agentloom.cli.run import _run_async

        captured: dict[str, object] = {}

        class _Capture:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def supports_model(self, model: str) -> bool:
                return True

            async def complete(self, **kwargs: object) -> None:  # pragma: no cover
                raise RuntimeError("not invoked")

            async def close(self) -> None:
                return None

            name = "mock"

        yaml_body = """\
name: latency-test
config:
  provider: mock
  model: mock-model
  latency_model: normal
  latency_ms: 42
steps:
  - id: answer
    type: llm_call
    prompt: "hi"
    output: answer
"""
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(yaml_body)
            with (
                patch("agentloom.cli.run._setup_observer", return_value=None),
                patch("agentloom.providers.mock.MockProvider", _Capture),
            ):
                import contextlib

                import anyio

                with contextlib.suppress(Exception):
                    # engine will fail because _Capture.complete raises
                    anyio.run(
                        _run_async,
                        yaml_path,
                        [],
                        None,
                        None,
                        None,
                        True,
                        False,
                        False,
                        False,
                        ".agentloom/checkpoints",
                        None,
                        None,
                    )
        assert captured.get("latency_model") == "normal"
        assert captured.get("latency_ms") == 42.0

    def test_replay_with_state_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "rec.json"
            rec_path.write_text(json.dumps(RECORDING))
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(SIMPLE_YAML)
            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "replay",
                        str(yaml_path),
                        "--recording",
                        str(rec_path),
                        "--state",
                        "question=Override?",
                    ],
                )
        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_yaml_mock_provider_uses_responses_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "rec.json"
            rec_path.write_text(json.dumps(RECORDING))
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(YAML_MOCK_CONFIG.format(responses_file=str(rec_path)))
            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(app, ["run", str(yaml_path), "--lite"])
        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "'answer'" in result.output
