"""Tests for CLI replay command and YAML-configured MockProvider."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from agentloom.cli.main import app

if TYPE_CHECKING:
    import pytest

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

    def test_yaml_latency_fields_reach_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agentloom.core.engine.compute_backoff", lambda *_args: 0.0)
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
            # In a double-quoted YAML string a Windows backslash starts an escape
            yaml_path.write_text(YAML_MOCK_CONFIG.format(responses_file=rec_path.as_posix()))
            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(app, ["run", str(yaml_path), "--lite"])
        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "'answer'" in result.output

    def test_a_responses_file_that_does_not_exist_fails_the_run(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "wf.yaml"
        yaml_path.write_text(YAML_MOCK_CONFIG.format(responses_file="nope.json"))
        result = runner.invoke(app, ["run", str(yaml_path), "--lite"])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "nope.json" in result.output
        assert "Mock response" not in result.output

    def test_a_malformed_responses_file_fails_without_a_traceback(self, tmp_path: Path) -> None:
        rec_path = tmp_path / "rec.json"
        rec_path.write_text("{not json")
        yaml_path = tmp_path / "wf.yaml"
        yaml_path.write_text(YAML_MOCK_CONFIG.format(responses_file=rec_path.as_posix()))
        result = runner.invoke(app, ["run", str(yaml_path), "--lite"])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "not valid JSON" in result.output


TOOL_YAML = """\
name: tool-replay
config:
  provider: openai
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

        def _wire(gw: Any, default: str, config_path: object = None) -> None:
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

    def test_a_masked_tool_argument_still_replays_strictly(self, tmp_path: Path) -> None:
        from agentloom.providers.mock import MockProvider

        call = {"id": "c1", "name": "http_request"}
        call["arguments"] = {"url": "https://example.com", "api_token": "dummy-secret"}
        src = tmp_path / "src.json"
        src.write_text(json.dumps({"ask": [_tool_turn("", [call]), _tool_turn("fetched", [])]}))
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(TOOL_YAML + 'state_schema:\n  "*token*": {redact: true}\n')
        recording = tmp_path / "rec.json"

        def _wire(gw: Any, default: str, config_path: object = None) -> None:
            gw.register(MockProvider(responses_file=src), priority=0)

        page = AsyncMock(return_value=('{"api_token": "dummy-secret", "ok": true}', True))
        with (
            patch("agentloom.cli.run._setup_providers", side_effect=_wire),
            patch("agentloom.steps._tools._execute", page),
        ):
            args = ["run", str(workflow), "--record", str(recording), "--record-tools", "--lite"]
            recorded = runner.invoke(app, args)
        assert recorded.exit_code == 0, recorded.output
        assert "dummy-secret" not in recording.read_text()
        assert page.call_args.args[0].arguments["api_token"] == "dummy-secret"

        never = AsyncMock(side_effect=AssertionError("a replayed tool ran"))
        with patch("agentloom.steps._tools._execute", never):
            args = ["replay", str(workflow), "--recording", str(recording), "--tools", "replay"]
            replayed = runner.invoke(app, args)
        assert replayed.exit_code == 0, replayed.output
        never.assert_not_called()


NO_MODEL_YAML = """\
name: no-model
steps:
  - {id: a, type: llm_call, prompt: hi, output: a}
"""

ANTHROPIC_REPLY = {
    "content": [{"type": "text", "text": "x"}],
    "model": "m",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

OPENAI_REPLY = {
    "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
    "model": "m",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class TestReplayProviderAndModel:
    @staticmethod
    def _recorded(tmp_path: Path, url: str, reply: dict[str, Any], *flags: str) -> list[str]:
        """Record a workflow without a ``model`` under *flags*; return replay's base arguments."""
        import httpx
        import respx

        workflow = tmp_path / "wf.yaml"
        workflow.write_text(NO_MODEL_YAML)
        recording = tmp_path / "rec.json"
        with respx.mock:
            respx.post(url).mock(return_value=httpx.Response(200, json=reply))
            result = runner.invoke(
                app, ["run", str(workflow), "--record", str(recording), "--lite", *flags]
            )
        assert result.exit_code == 0, result.output
        return ["replay", str(workflow), "-r", str(recording)]

    def test_a_recording_made_with_provider_replays_with_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        args = self._recorded(
            tmp_path,
            "https://api.anthropic.com/v1/messages",
            ANTHROPIC_REPLY,
            "--provider",
            "anthropic",
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        assert runner.invoke(app, args).exit_code == 1  # the default model isn't the one recorded
        result = runner.invoke(app, [*args, "--provider", "anthropic"])
        assert result.exit_code == 0, result.output

    def test_verify_determinism_takes_the_recorded_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        args = self._recorded(
            tmp_path,
            "https://api.anthropic.com/v1/messages",
            ANTHROPIC_REPLY,
            "--provider",
            "anthropic",
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        verify = ["verify-determinism", *args[1:]]
        assert runner.invoke(app, verify).exit_code == 1
        result = runner.invoke(app, [*verify, "--provider", "anthropic"])
        assert result.exit_code == 0, result.output

    def test_a_recording_that_cannot_be_replaced_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(NO_MODEL_YAML)
        taken = tmp_path / "a_directory"
        taken.mkdir()
        result = runner.invoke(app, ["run", str(workflow), "--record", str(taken), "--lite"])
        assert result.exit_code == 1
        assert "cannot replace" in result.output

    def test_verify_determinism_takes_the_recorded_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        args = self._recorded(
            tmp_path,
            "https://api.openai.com/v1/chat/completions",
            OPENAI_REPLY,
            "--model",
            "gpt-4.1",
        )
        monkeypatch.delenv("OPENAI_API_KEY")
        verify = ["verify-determinism", *args[1:]]
        assert runner.invoke(app, verify).exit_code == 1
        result = runner.invoke(app, [*verify, "--model", "gpt-4.1"])
        assert result.exit_code == 0, result.output

    def test_a_recording_made_with_model_replays_with_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        args = self._recorded(
            tmp_path,
            "https://api.openai.com/v1/chat/completions",
            OPENAI_REPLY,
            "--model",
            "gpt-4.1",
        )
        monkeypatch.delenv("OPENAI_API_KEY")
        result = runner.invoke(app, [*args, "--model", "gpt-4.1"])
        assert result.exit_code == 0, result.output
