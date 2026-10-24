"""Tests for CLI replay command and YAML-configured MockProvider."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

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


TOOL_YAML = """\
name: tool-replay
config:
  provider: mock
  model: mock-model
steps:
  - id: ask
    type: llm_call
    prompt: "Fetch the page"
    tools:
      - name: http_request
        description: "Fetch a URL"
        parameters: {type: object, properties: {url: {type: string}}}
    output: answer
"""

FETCH = {"name": "http_request", "arguments": {"url": "https://example.com"}}


def _tool_turn(content: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    turn: dict[str, Any] = {
        "content": content,
        "model": "mock-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "cost_usd": 0.0,
        "finish_reason": "tool_calls" if calls else "stop",
    }
    if calls:
        turn["tool_calls"] = calls
    return turn


class TestReplayTools:
    def test_record_tools_then_replay_them_without_running_tools(self, tmp_path: Path) -> None:
        """``run --record-tools`` stores tool results; ``replay --tools replay`` serves them."""
        from agentloom.providers.mock import MockProvider

        src = tmp_path / "src.json"
        loop = [_tool_turn("", [{"id": "c1", **FETCH}]), _tool_turn("fetched", [])]
        src.write_text(json.dumps({"ask": loop}))
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(TOOL_YAML)
        recording = tmp_path / "rec.json"

        def _wire(gw: Any, default: str) -> None:
            gw.register(MockProvider(responses_file=src), priority=0)

        page = AsyncMock(return_value=("<html>", True))
        with (
            patch("agentloom.cli.run._setup_providers", side_effect=_wire),
            patch("agentloom.steps._tools._execute", page),
        ):
            args = ["run", str(workflow), "--record", str(recording), "--record-tools", "--lite"]
            recorded = runner.invoke(app, args)
        assert recorded.exit_code == 0, recorded.output
        tools = json.loads(recording.read_text())["_tools"]
        assert tools == {"ask": [{**FETCH, "result": "<html>", "success": True}]}

        never = AsyncMock(side_effect=AssertionError("a replayed tool ran"))
        with patch("agentloom.steps._tools._execute", never):
            args = ["replay", str(workflow), "--recording", str(recording), "--tools", "replay"]
            replayed = runner.invoke(app, args)
        assert replayed.exit_code == 0, replayed.output
        never.assert_not_called()
