"""Tests for CLI run command."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agentloom.cli.main import app
from agentloom.exceptions import ConfigError

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

SIMPLE_YAML = """\
name: cli-test
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


class TestRunCommand:
    def test_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["run", "/tmp/no_such_workflow_xyz.yaml"])
        assert result.exit_code != 0

    def test_state_override_format(self) -> None:
        """A ``--state`` override not in ``key=value`` form fails."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--state", "bad_format"])
        assert result.exit_code == 2
        assert "key=value" in (result.output + (result.stderr or ""))

    def test_generic_openai_key_variable_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset ``api_key_env`` variable stops the run with a clear error."""
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        endpoint = (
            "provider: generic_openai\n"
            "  base_url: http://vllm.local:8000/v1\n"
            "  api_key_env: VLLM_API_KEY"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML.replace("provider: mock", endpoint))
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--lite"])
        assert result.exit_code == 1
        assert "'VLLM_API_KEY' is not set" in (result.output + (result.stderr or ""))


class TestRunCheckpoint:
    def test_checkpoint_flag_creates_checkpoint(self) -> None:
        """``--checkpoint`` prints the run ID and writes one checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            with tempfile.TemporaryDirectory() as cp_dir:
                with patch("agentloom.cli.run._setup_providers") as mock_setup:
                    from tests.conftest import MockProvider

                    def _wire(gw: object, default: str, config_path: object = None) -> None:
                        from agentloom.providers.gateway import ProviderGateway

                        assert isinstance(gw, ProviderGateway)
                        gw.register(MockProvider(), priority=0)

                    mock_setup.side_effect = _wire

                    with patch("agentloom.cli.run._setup_observer", return_value=None):
                        result = runner.invoke(
                            app,
                            [
                                "run",
                                f.name,
                                "--checkpoint",
                                "--checkpoint-dir",
                                cp_dir,
                                "--lite",
                            ],
                        )

                assert result.exit_code == 0, f"stdout: {result.output}"
                assert "Run ID:" in result.output

                from pathlib import Path

                cp_files = list(Path(cp_dir).glob("*.json"))
                assert len(cp_files) == 1

    def test_no_run_id_echo_without_checkpoint_flag(self) -> None:
        """Without ``--checkpoint`` the run ID is not printed: nothing can resume it."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            with patch("agentloom.cli.run._setup_providers") as mock_setup:
                from tests.conftest import MockProvider

                def _wire(gw: object, default: str, config_path: object = None) -> None:
                    from agentloom.providers.gateway import ProviderGateway

                    assert isinstance(gw, ProviderGateway)
                    gw.register(MockProvider(), priority=0)

                mock_setup.side_effect = _wire

                with patch("agentloom.cli.run._setup_observer", return_value=None):
                    result = runner.invoke(app, ["run", f.name, "--lite"])

            assert result.exit_code == 0, f"stdout: {result.output}"
            assert "Run ID:" not in result.output


@pytest.fixture
def no_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset what provider discovery reads: ``AGENTLOOM_*``, ``*_API_KEY`` and ``*_BASE_URL``.

    Clearing the whole environment instead leaves OpenSSL unable to start on Windows.
    """
    for name in list(os.environ):
        if name.startswith("AGENTLOOM_") or name.endswith(("_API_KEY", "_BASE_URL")):
            monkeypatch.delenv(name)


class TestSetupProviders:
    @pytest.mark.usefixtures("no_provider_env")
    def test_ollama_registered_when_explicit_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ollama registers when the workflow names it as its primary provider."""
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "1")
        _setup_providers(gw, "ollama")
        assert len(gw._providers) >= 1
        assert any(e.provider.name == "ollama" for e in gw._providers)

    @pytest.mark.usefixtures("no_provider_env")
    def test_ollama_not_registered_by_default(self) -> None:
        """Without the opt-in flag the CLI does not register Ollama, as in the config layer."""
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        _setup_providers(gw, "openai")
        names = [e.provider.name for e in gw._providers]
        assert "ollama" not in names

    @pytest.mark.usefixtures("no_provider_env")
    def test_openai_registered_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _setup_providers(gw, "openai")
        names = [e.provider.name for e in gw._providers]
        assert "openai" in names
        # Ollama is opt-in: another provider's key does not bring it in.
        assert "ollama" not in names

    def test_anthropic_registered_when_key_set(self) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            _setup_providers(gw, "anthropic")
        names = [e.provider.name for e in gw._providers]
        assert "anthropic" in names

    def test_google_registered_when_key_set(self) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            _setup_providers(gw, "google")
        names = [e.provider.name for e in gw._providers]
        assert "google" in names

    @pytest.mark.usefixtures("no_provider_env")
    def test_priority_reflects_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With Ollama opted in as fallback, the default provider sorts ahead of it."""
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "1")
        _setup_providers(gw, "openai")
        openai_entry = next(e for e in gw._providers if e.provider.name == "openai")
        ollama_entry = next(e for e in gw._providers if e.provider.name == "ollama")
        assert openai_entry.priority < ollama_entry.priority


class TestSetupObserver:
    def test_lite_mode_returns_none(self) -> None:
        from agentloom.cli.run import _setup_observer

        assert _setup_observer(lite=True) is None

    def test_observer_created_when_otel_available(self) -> None:
        from agentloom.cli.run import _setup_observer

        _setup_observer(lite=False)
        # No assertion: the result depends on whether OTel is installed; neither path may raise.

    def test_observer_respects_otel_endpoint_env(self) -> None:
        from types import ModuleType

        from agentloom.cli.run import _setup_observer

        custom = "http://collector.internal:4317"
        fake_mod = ModuleType("fake_otel")
        with (
            patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": custom}),
            patch("agentloom.compat.try_import", return_value=fake_mod),
            patch("agentloom.observability.tracing.TracingManager") as mock_tm,
            patch("agentloom.observability.metrics.MetricsManager") as mock_mm,
        ):
            mock_tm.return_value = mock_tm
            mock_mm.return_value = mock_mm
            _setup_observer(lite=False)
            mock_tm.assert_called_once_with(endpoint=custom)
            mock_mm.assert_called_once_with(endpoint=custom)


class TestPrintResult:
    def test_prints_success(self) -> None:
        from agentloom.cli.run import _print_result
        from agentloom.core.results import StepResult, StepStatus, WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.SUCCESS,
            total_duration_ms=1234.5,
            total_tokens=100,
            total_cost_usd=0.005,
            step_results={
                "step1": StepResult(
                    step_id="step1", status=StepStatus.SUCCESS, duration_ms=500.0, cost_usd=0.005
                )
            },
            final_state={"answer": "42"},
        )
        _print_result(result)

    def test_prints_failure(self) -> None:
        from agentloom.cli.run import _print_result
        from agentloom.core.results import StepResult, StepStatus, WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.FAILED,
            error="something broke",
            step_results={
                "s1": StepResult(step_id="s1", status=StepStatus.FAILED, error="boom"),
                "s2": StepResult(step_id="s2", status=StepStatus.SKIPPED),
            },
            final_state={},
        )
        _print_result(result)

    def test_prints_paused(self) -> None:
        from agentloom.cli.run import _print_result
        from agentloom.core.results import StepResult, StepStatus, WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.PAUSED,
            error="Pause requested at step 'review'",
            step_results={
                "draft": StepResult(step_id="draft", status=StepStatus.SUCCESS, duration_ms=100.0),
                "review": StepResult(step_id="review", status=StepStatus.PAUSED),
            },
            final_state={"draft_output": "hello"},
        )
        _print_result(result)

    def test_a_timed_out_step_shows_as_timeout(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agentloom.cli.run import _print_result
        from agentloom.core.results import StepResult, StepStatus, WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.FAILED,
            step_results={
                "slow": StepResult(step_id="slow", status=StepStatus.TIMEOUT, duration_ms=5.0),
                "bad": StepResult(step_id="bad", status=StepStatus.FAILED),
            },
            final_state={},
        )
        _print_result(result)
        out = capsys.readouterr().out
        assert "[TIMEOUT] slow" in out
        assert "[FAIL] bad" in out


APPROVAL_YAML = """\
name: gated
config: {provider: mock, model: m}
steps:
  - {id: gate, type: approval_gate, output: decision}
"""


class TestApprovalHint:
    def test_without_a_checkpoint_no_resume_command_is_offered(self, tmp_path: Path) -> None:
        wf = tmp_path / "wf.yaml"
        wf.write_text(APPROVAL_YAML)
        result = runner.invoke(app, ["run", str(wf), "--lite"])
        assert result.exit_code == 0, result.output
        assert "[APPROVAL REQUIRED] Step 'gate'" in result.output
        assert "agentloom resume" not in result.output

    def test_with_a_checkpoint_the_commands_name_the_run(self, tmp_path: Path) -> None:
        wf = tmp_path / "wf.yaml"
        wf.write_text(APPROVAL_YAML)
        result = runner.invoke(
            app, ["run", str(wf), "--lite", "--checkpoint", "--checkpoint-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        run_id = result.output.split("Run ID: ")[1].split()[0]
        assert f"agentloom resume {run_id} --approve" in result.output
        assert f"agentloom resume {run_id} --reject" in result.output


class TestMockIsNotRateLimited:
    async def test_the_mock_serves_more_calls_than_a_providers_minute(self) -> None:
        import anyio

        from agentloom.cli.run import _setup_mock
        from agentloom.core.models import WorkflowConfig
        from agentloom.providers.gateway import ProviderGateway

        gateway = ProviderGateway()
        _setup_mock(gateway, WorkflowConfig(provider="mock", model="m"), None, "wf")
        with anyio.fail_after(5):
            for _ in range(120):
                await gateway.complete([{"role": "user", "content": "hi"}], model="m")

    @pytest.mark.parametrize("command", ["run", "replay", "verify-determinism"])
    def test_every_command_that_sets_up_a_recording_lifts_the_limit(
        self, command: str, tmp_path: Path
    ) -> None:
        import json

        from agentloom.providers.gateway import ProviderGateway

        wf = tmp_path / "wf.yaml"
        wf.write_text(SIMPLE_YAML)
        rec = tmp_path / "rec.json"
        rec.write_text(json.dumps({"answer": {"content": "42"}}))
        argv = {
            "run": ["run", str(wf), "--lite", "--mock-responses", str(rec)],
            "replay": ["replay", str(wf), "--recording", str(rec)],
            "verify-determinism": ["verify-determinism", str(wf), "--recording", str(rec)],
        }[command]

        original = ProviderGateway.register
        with patch.object(
            ProviderGateway, "register", autospec=True, side_effect=original
        ) as register:
            result = runner.invoke(app, argv)

        assert result.exit_code == 0, result.output
        assert register.call_args_list
        assert all(call.kwargs["max_rpm"] > 60 for call in register.call_args_list)


class TestRunRecordAndReplay:
    def test_record_and_mock_mutually_exclusive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            result = runner.invoke(
                app,
                [
                    "run",
                    f.name,
                    "--record",
                    "/tmp/rec.json",
                    "--mock-responses",
                    f.name,
                ],
            )
        assert result.exit_code != 0
        combined = result.output + (result.stderr or "")
        assert "mutually exclusive" in combined

    @pytest.mark.parametrize("flags", [[], ["--provider", "mock"]])
    def test_record_refuses_a_mock_provider(self, tmp_path: Path, flags: list[str]) -> None:
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(
            SIMPLE_YAML.replace("provider: mock", "provider: openai") if flags else SIMPLE_YAML
        )
        recording = tmp_path / "rec.json"
        result = runner.invoke(
            app, ["run", str(workflow), *flags, "--record", str(recording), "--lite"]
        )
        assert result.exit_code == 2
        assert "--record needs a live provider" in (result.output + (result.stderr or ""))
        assert not recording.exists()

    def test_record_tools_needs_record(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--record-tools"])
        assert result.exit_code == 2
        assert "--record-tools needs --record" in (result.output + (result.stderr or ""))

    def test_record_replaces_a_file_left_by_an_earlier_run(self, tmp_path: Path) -> None:
        import json

        from agentloom.providers.mock import MockProvider

        def _wire(gw: Any, default: str, config_path: object = None) -> None:
            gw.register(MockProvider(default_response="new"), priority=0)
            gw.register(MockProvider(default_response="unused"), priority=1)

        workflow = tmp_path / "wf.yaml"
        workflow.write_text(SIMPLE_YAML.replace("provider: mock", "provider: openai"))
        recording = tmp_path / "rec.json"
        recording.write_text(
            json.dumps(
                {
                    "_version": 2,
                    "gone": {"content": "from an earlier run"},
                    "_inter_agent_events": [],
                    "_tools": {"gone": []},
                }
            )
        )
        with patch("agentloom.cli.run._setup_providers", side_effect=_wire):
            result = runner.invoke(
                app, ["run", str(workflow), "--record", str(recording), "--lite"]
            )
        assert result.exit_code == 0, result.output
        data = json.loads(recording.read_text())
        assert data["answer"]["content"] == "new"
        assert "gone" not in data
        assert "_tools" not in data
        assert data["_inter_agent_events"]

    def test_a_run_that_never_starts_keeps_the_old_recording(self, tmp_path: Path) -> None:
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(SIMPLE_YAML.replace("provider: mock", "provider: openai"))
        recording = tmp_path / "rec.json"
        recording.write_text('{"answer": {"content": "kept"}}')
        with patch(
            "agentloom.cli.run._setup_providers", side_effect=ConfigError("bad config file")
        ):
            result = runner.invoke(
                app, ["run", str(workflow), "--record", str(recording), "--lite"]
            )
        assert result.exit_code == 1
        assert "kept" in recording.read_text()

    def test_mock_responses_file_must_exist(self, tmp_path: Path) -> None:
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(SIMPLE_YAML)
        missing = tmp_path / "nope.json"
        result = runner.invoke(
            app, ["run", str(workflow), "--mock-responses", str(missing), "--lite"]
        )
        assert result.exit_code == 2
        assert "Running workflow" not in result.output

    def test_mock_responses_invalid_file_exits_with_error(self) -> None:
        # A malformed recording is a startup error (exit 1), not a run on an empty fixture.
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "broken.json"
            rec_path.write_text("{not valid json at all")
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(SIMPLE_YAML)
            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "run",
                        str(yaml_path),
                        "--mock-responses",
                        str(rec_path),
                        "--lite",
                    ],
                )
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "not valid JSON" in combined or "Error" in combined

    def test_mock_responses_registers_mock_provider(self) -> None:
        import json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "fix.json"
            rec_path.write_text(
                json.dumps(
                    {
                        "answer": {
                            "content": "42",
                            "model": "mock-model",
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                            "cost_usd": 0.0,
                            "latency_ms": 0.0,
                            "finish_reason": "stop",
                        }
                    }
                )
            )
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(SIMPLE_YAML)
            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "run",
                        str(yaml_path),
                        "--mock-responses",
                        str(rec_path),
                        "--lite",
                    ],
                )
        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "'answer'" in result.output

    def test_record_wraps_registered_providers(self) -> None:
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            rec_path = Path(tmp) / "out.json"
            yaml_path = Path(tmp) / "wf.yaml"
            yaml_path.write_text(SIMPLE_YAML)

            from tests.conftest import MockProvider as TestMock

            captured: dict[str, object] = {}

            def _wire(gw: object, default: str, config_path: object = None) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                inner = TestMock()
                gw.register(inner, priority=0)
                captured["inner"] = inner

            with patch("agentloom.cli.run._setup_providers") as mock_setup:
                mock_setup.side_effect = _wire
                with patch("agentloom.cli.run._setup_observer", return_value=None):
                    result = runner.invoke(
                        app,
                        [
                            "run",
                            str(yaml_path),
                            "--provider",
                            "openai",
                            "--record",
                            str(rec_path),
                            "--lite",
                        ],
                    )

            assert result.exit_code == 0, f"stdout: {result.output}"
            assert rec_path.exists()
            import json as _json

            data = _json.loads(rec_path.read_text())
            # The recording carries a ``_version`` envelope beside the one captured call.
            entries = {k: v for k, v in data.items() if not k.startswith("_")}
            assert len(entries) == 1


class TestStateJSONParsing:
    """``--state`` decodes JSON-shaped values; plain strings and URLs pass through untouched."""

    def test_parses_json_array(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value("[1, 2, 3]") == [1, 2, 3]

    def test_parses_json_object(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value('{"name": "alice"}') == {"name": "alice"}

    def test_parses_json_number(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value("42") == 42
        assert _parse_state_value("-3.5") == -3.5

    def test_parses_json_bool_and_null(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value("true") is True
        assert _parse_state_value("false") is False
        assert _parse_state_value("null") is None

    def test_preserves_plain_string(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value("hello") == "hello"

    def test_preserves_url_with_query_string(self) -> None:
        # A URL holds ``=`` and ``&`` but is not JSON-shaped, so it stays a string.
        from agentloom.cli.run import _parse_state_value

        url = "https://example.com/x?a=b&c=d"
        assert _parse_state_value(url) == url

    def test_json_shaped_but_invalid_falls_back_to_string(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value("[not, valid") == "[not, valid"

    def test_empty_value_preserved(self) -> None:
        from agentloom.cli.run import _parse_state_value

        assert _parse_state_value("") == ""


LIVE_YAML = SIMPLE_YAML.replace("provider: mock", "provider: openai")


class TestConfigOption:
    """``--config``, ``AGENTLOOM_CONFIG`` and ``./agentloom.yaml`` reach provider setup."""

    @pytest.fixture(autouse=True)
    def _in_tmp(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENTLOOM_CONFIG", raising=False)

    def _run(self, tmp_path: Path, *flags: str) -> tuple[object, list[object]]:
        """Run the live workflow with mocked providers; return the result and the paths seen."""
        from tests.conftest import MockProvider

        workflow = tmp_path / "wf.yaml"
        workflow.write_text(LIVE_YAML)
        seen: list[object] = []

        def _wire(gw: Any, default: str, config_path: object = None) -> None:
            seen.append(config_path)
            gw.register(MockProvider(), priority=0)

        with (
            patch("agentloom.cli.run._setup_providers", side_effect=_wire),
            patch("agentloom.cli.run._setup_observer", return_value=None),
        ):
            result = runner.invoke(app, ["run", str(workflow), "--lite", *flags])
        return result, seen

    def test_flag_path_reaches_provider_setup(self, tmp_path: Path) -> None:
        config = tmp_path / "custom.yaml"
        config.write_text("providers: [{name: openai}]\n")
        result, seen = self._run(tmp_path, "--config", str(config))
        assert result.exit_code == 0, result.output
        assert seen == [config.resolve()]

    def test_working_directory_file_is_read_without_the_flag(self, tmp_path: Path) -> None:
        (tmp_path / "agentloom.yaml").write_text("providers: [{name: openai}]\n")
        result, seen = self._run(tmp_path)
        assert result.exit_code == 0, result.output
        assert seen == [(tmp_path / "agentloom.yaml").resolve()]

    def test_flag_overrides_the_working_directory_file(self, tmp_path: Path) -> None:
        (tmp_path / "agentloom.yaml").write_text("providers: [{name: openai}]\n")
        config = tmp_path / "custom.yaml"
        config.write_text("providers: [{name: openai}]\n")
        _, seen = self._run(tmp_path, "--config", str(config))
        assert seen == [config.resolve()]

    def test_missing_env_file_exits_in_one_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENTLOOM_CONFIG", str(tmp_path / "gone.yaml"))
        result, seen = self._run(tmp_path)
        assert result.exit_code == 1
        assert "not found" in result.output
        assert seen == []

    @pytest.mark.usefixtures("no_provider_env")
    def test_invalid_file_exits_in_one_line(self, tmp_path: Path) -> None:
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(LIVE_YAML)
        config = tmp_path / "agentloom.yaml"
        config.write_text("providers: [{name: openai, api_key: YOUR_KEY_HERE}]\n")
        with patch("agentloom.cli.run._setup_observer", return_value=None):
            result = runner.invoke(app, ["run", str(workflow), "--lite"])
        assert result.exit_code == 1
        assert "set OPENAI_API_KEY instead" in result.output
        assert "Traceback" not in result.output

    @pytest.mark.usefixtures("no_provider_env")
    def test_providers_block_registers_only_its_providers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        monkeypatch.setenv("OPENAI_API_KEY", "from-env-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env-anthropic")
        config = tmp_path / "agentloom.yaml"
        config.write_text("providers: [{name: anthropic, models: [claude-x]}]\n")
        gw = ProviderGateway()
        _setup_providers(gw, "anthropic", config)
        assert [e.provider.name for e in gw._providers] == ["anthropic"]
        assert gw._providers[0].provider.api_key == "from-env-anthropic"

    @pytest.mark.parametrize("command", ["resume", "replay"])
    def test_resume_and_replay_pass_the_path(
        self, tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "custom.yaml"
        config.write_text("providers: [{name: openai}]\n")
        seen: list[object] = []

        async def _capture(*args: Any, **kwargs: Any) -> None:
            seen.append(kwargs.get("config_path", args[-1] if args else None))

        if command == "resume":
            monkeypatch.setattr("agentloom.cli.resume._resume_async", _capture)
            argv = ["resume", "run-1", "--config", str(config)]
        else:
            workflow = tmp_path / "wf.yaml"
            workflow.write_text(SIMPLE_YAML)
            recording = tmp_path / "rec.json"
            recording.write_text("{}")
            monkeypatch.setattr("agentloom.cli.replay._run_async", _capture)
            argv = ["replay", str(workflow), "-r", str(recording), "--config", str(config)]
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        assert seen == [config.resolve()]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is left alone on Windows")
class TestSigtermAsInterrupt:
    """A run stops on SIGTERM as on Ctrl-C, which saves its checkpoint."""

    def test_the_handler_is_swapped_while_the_run_goes(self) -> None:
        from agentloom.cli.run import _forward_to_sigint, _sigterm_as_interrupt

        before = signal.getsignal(signal.SIGTERM)
        with _sigterm_as_interrupt():
            assert signal.getsignal(signal.SIGTERM) is _forward_to_sigint
        assert signal.getsignal(signal.SIGTERM) is before

    def test_windows_and_other_threads_keep_theirs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import threading

        from agentloom.cli.run import _sigterm_as_interrupt

        before = signal.getsignal(signal.SIGTERM)
        seen: list[object] = []

        def _inside() -> None:
            with _sigterm_as_interrupt():
                seen.append(signal.getsignal(signal.SIGTERM))

        thread = threading.Thread(target=_inside)
        thread.start()
        thread.join()
        monkeypatch.setattr(sys, "platform", "win32")
        _inside()
        assert seen == [before, before]

    def test_sigterm_is_raised_again_as_sigint(self) -> None:
        from agentloom.cli.run import _forward_to_sigint

        received: list[int] = []
        previous = signal.signal(signal.SIGINT, lambda signum, frame: received.append(signum))
        try:
            _forward_to_sigint(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGINT, previous)
        assert received == [signal.SIGINT]

    def test_a_sigint_left_ignored_gets_a_handler_for_the_run(self) -> None:
        from agentloom.cli.run import _sigterm_as_interrupt

        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with _sigterm_as_interrupt():
                assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
            assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        finally:
            signal.signal(signal.SIGINT, previous)

    def test_a_sigint_handler_of_the_caller_stays(self) -> None:
        from agentloom.cli.run import _sigterm_as_interrupt

        def _own(signum: int, frame: object) -> None: ...

        previous = signal.signal(signal.SIGINT, _own)
        try:
            with _sigterm_as_interrupt():
                assert signal.getsignal(signal.SIGINT) is _own
            assert signal.getsignal(signal.SIGINT) is _own
        finally:
            signal.signal(signal.SIGINT, previous)

    @pytest.mark.parametrize("command", ["run", "resume"])
    def test_run_and_resume_go_through_it(self, command: str, tmp_path: Path) -> None:
        from contextlib import contextmanager

        entered: list[bool] = []

        @contextmanager
        def _spy() -> Any:
            entered.append(True)
            yield

        workflow = tmp_path / "wf.yaml"
        workflow.write_text(LIVE_YAML)
        args = [str(workflow)] if command == "run" else ["missing-run"]
        flags = ["--checkpoint-dir", str(tmp_path)] if command == "resume" else []
        with patch("agentloom.cli.run._sigterm_as_interrupt", _spy):
            runner.invoke(app, [command, *args, "--lite", *flags])
        assert entered == [True]


NO_MODEL_YAML = """\
name: no-model
state:
  q: hi
steps:
  - id: answer
    type: llm_call
    prompt: "{state.q}"
    output: a
"""


class TestProviderOverrideAndModel:
    def _run(self, tmp_path: Path, text: str, *flags: str) -> tuple[Any, list[str]]:
        from tests.conftest import MockProvider

        path = tmp_path / "w.yaml"
        path.write_text(text, encoding="utf-8")
        provider = MockProvider()

        def _wire(gw: Any, default: str, config_path: object = None) -> None:
            gw.register(provider, priority=0)

        with patch("agentloom.cli.run._setup_providers", side_effect=_wire):
            result = runner.invoke(app, ["run", str(path), "--lite", *flags])
        return result, [c["model"] for c in provider.calls]

    def test_a_provider_flag_brings_its_default_model(self, tmp_path: Path) -> None:
        result, models = self._run(tmp_path, NO_MODEL_YAML, "--provider", "anthropic")
        assert result.exit_code == 0, result.output
        assert models == ["claude-haiku-4-5-20251001"]

    def test_a_model_in_the_file_stays(self, tmp_path: Path) -> None:
        text = NO_MODEL_YAML.replace("state:", "config:\n  model: gpt-4o\nstate:", 1)
        result, models = self._run(tmp_path, text, "--provider", "anthropic")
        assert result.exit_code == 0, result.output
        assert models == ["gpt-4o"]

    def test_a_provider_without_a_default_needs_a_model(self, tmp_path: Path) -> None:
        result, models = self._run(tmp_path, NO_MODEL_YAML, "--provider", "ollama")
        assert result.exit_code == 2
        assert "no default model" in (result.output + (result.stderr or ""))
        assert models == []

    def test_a_provider_without_a_default_takes_steps_that_name_theirs(
        self, tmp_path: Path
    ) -> None:
        text = NO_MODEL_YAML.replace("    output: a\n", "    output: a\n    model: phi4\n", 1)
        result, models = self._run(tmp_path, text, "--provider", "ollama")
        assert result.exit_code == 0, result.output
        assert models == ["phi4"]

    def test_the_model_flag_covers_a_provider_without_a_default(self, tmp_path: Path) -> None:
        result, models = self._run(
            tmp_path, NO_MODEL_YAML, "--provider", "ollama", "--model", "phi4"
        )
        assert result.exit_code == 0, result.output
        assert models == ["phi4"]


class TestSetupProvidersPrimaryAndAliases:
    @pytest.mark.usefixtures("no_provider_env")
    def test_the_workflow_provider_is_the_primary_of_a_config_file(self, tmp_path: Path) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        path = tmp_path / "agentloom.yaml"
        path.write_text(
            "providers:\n  - name: ollama\n    priority: 0\n  - name: openai\n    priority: 50\n",
            encoding="utf-8",
        )
        gw = ProviderGateway()
        _setup_providers(gw, "openai", path)

        assert [e.provider.name for e in gw._get_candidates("gpt-4o")] == ["openai", "ollama"]

    @pytest.mark.usefixtures("no_provider_env")
    def test_a_discovered_primary_goes_ahead_of_a_listed_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        gw = ProviderGateway()
        _setup_providers(gw, "ollama")

        assert [e.provider.name for e in gw._get_candidates("gpt-4o-mini")] == ["ollama", "openai"]

    @pytest.mark.usefixtures("no_provider_env")
    def test_a_primary_that_does_not_serve_the_name_stays_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        gw = ProviderGateway()
        _setup_providers(gw, "anthropic")

        assert [e.provider.name for e in gw._get_candidates("gpt-4o-mini")] == ["openai"]
        assert [e.provider.name for e in gw._get_candidates("gpt-5.5")] == ["openai"]

    @pytest.mark.usefixtures("no_provider_env")
    def test_aliases_reach_the_gateway(self, tmp_path: Path) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        path = tmp_path / "agentloom.yaml"
        path.write_text(
            "providers:\n  - name: ollama\n    model_aliases:\n      gpt-4o: phi4\n",
            encoding="utf-8",
        )
        gw = ProviderGateway()
        _setup_providers(gw, "ollama", path)

        assert gw._providers[0].model_id("gpt-4o") == "phi4"


class TestProviderFlagDropsBaseUrl:
    @staticmethod
    def _workflow(config: dict[str, Any]) -> Any:
        from agentloom.core.models import WorkflowDefinition

        llm_call = {"id": "s", "type": "llm_call", "prompt": "hi"}
        return WorkflowDefinition.model_validate(
            {"name": "w", "config": config, "steps": [llm_call]}
        )

    def test_it_warns_and_drops_it(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agentloom.cli.run import _switch_provider

        workflow = self._workflow(
            {"provider": "generic_openai", "base_url": "http://x", "model": "m"}
        )
        _switch_provider(workflow, "openai")

        err = capsys.readouterr().err
        assert "base_url" in err
        assert "openai" in err
        assert workflow.config.base_url is None

    def test_generic_openai_itself_does_not_warn(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agentloom.cli.run import _switch_provider

        workflow = self._workflow(
            {"provider": "generic_openai", "base_url": "http://x", "model": "m"}
        )
        _switch_provider(workflow, "generic_openai")

        assert capsys.readouterr().err == ""
        assert workflow.config.base_url == "http://x"

    def test_a_workflow_without_a_base_url_does_not_warn(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agentloom.cli.run import _switch_provider

        _switch_provider(self._workflow({"provider": "openai"}), "anthropic")

        assert capsys.readouterr().err == ""


FALLOVER_YAML = """\
name: fallover
config: {provider: openai, model: gpt-4o-mini}
steps:
  - id: ask
    type: llm_call
    prompt: read it
    tools:
      - name: file_read
        description: Read a file.
        parameters: {type: object, properties: {path: {type: string}}, required: [path]}
    output: answer
"""


class TestRecordAcrossProviders:
    def test_a_tool_loop_that_falls_over_keeps_its_turns_and_replays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        import httpx
        import respx

        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        monkeypatch.setenv("AGENTLOOM_OLLAMA_FALLBACK", "1")
        target = tmp_path / "data.txt"
        target.write_text("contents")
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(FALLOVER_YAML)
        recording = tmp_path / "rec.json"
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "file_read", "arguments": json.dumps({"path": str(target)})},
        }
        tool_turn = {
            "choices": [
                {"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}
            ],
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        final = {"model": "m", "message": {"role": "assistant", "content": "done"}, "done": True}
        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                side_effect=[httpx.Response(200, json=tool_turn), httpx.Response(500, text="down")]
            )
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(200, json=final)
            )
            recorded = runner.invoke(
                app, ["run", str(workflow), "--record", str(recording), "--lite"]
            )

        assert recorded.exit_code == 0, recorded.output
        turns = json.loads(recording.read_text())["ask"]
        assert [turn["provider"] for turn in turns] == ["openai", "ollama"]
        replayed = runner.invoke(app, ["replay", str(workflow), "-r", str(recording)])
        assert replayed.exit_code == 0, replayed.output
