"""Tests for CLI run command."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from typer.testing import CliRunner

from agentloom.cli.main import app

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
        """Invalid state format should fail."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--state", "bad_format"])
        assert result.exit_code != 0
        assert "key=value" in (result.output + (result.stderr or ""))


class TestRunCheckpoint:
    def test_checkpoint_flag_creates_checkpoint(self) -> None:
        """Running with --checkpoint should print a run ID and create a checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            with tempfile.TemporaryDirectory() as cp_dir:
                with patch("agentloom.cli.run._setup_providers") as mock_setup:
                    from tests.conftest import MockProvider

                    def _wire(gw: object, default: str) -> None:
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

                # Verify checkpoint file exists
                from pathlib import Path

                cp_files = list(Path(cp_dir).glob("*.json"))
                assert len(cp_files) == 1

    def test_no_run_id_echo_without_checkpoint_flag(self) -> None:
        """Without --checkpoint the CLI should not surface the Run ID to
        the user (the id is only actionable when a checkpointer is active —
        otherwise it's internal-only metadata used for trace correlation)."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SIMPLE_YAML)
            f.flush()
            with patch("agentloom.cli.run._setup_providers") as mock_setup:
                from tests.conftest import MockProvider

                def _wire(gw: object, default: str) -> None:
                    from agentloom.providers.gateway import ProviderGateway

                    assert isinstance(gw, ProviderGateway)
                    gw.register(MockProvider(), priority=0)

                mock_setup.side_effect = _wire

                with patch("agentloom.cli.run._setup_observer", return_value=None):
                    result = runner.invoke(app, ["run", f.name, "--lite"])

            assert result.exit_code == 0, f"stdout: {result.output}"
            assert "Run ID:" not in result.output


class TestSetupProviders:
    def test_ollama_always_registered(self) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        with patch.dict("os.environ", {}, clear=True):
            _setup_providers(gw, "ollama")
        assert len(gw._providers) >= 1
        assert any(e.provider.name == "ollama" for e in gw._providers)

    def test_openai_registered_when_key_set(self) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            _setup_providers(gw, "openai")
        names = [e.provider.name for e in gw._providers]
        assert "openai" in names
        assert "ollama" in names

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

    def test_priority_reflects_default(self) -> None:
        from agentloom.cli.run import _setup_providers
        from agentloom.providers.gateway import ProviderGateway

        gw = ProviderGateway()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
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
        # If OTel is installed, observer is created; if not, None
        # Either way, no error should occur

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
        # Should not raise
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
                    "/tmp/rec.json",
                ],
            )
        assert result.exit_code != 0
        combined = result.output + (result.stderr or "")
        assert "mutually exclusive" in combined

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

            def _wire(gw: object, default: str) -> None:
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
                            "--record",
                            str(rec_path),
                            "--lite",
                        ],
                    )

            assert result.exit_code == 0, f"stdout: {result.output}"
            assert rec_path.exists()
            import json as _json

            data = _json.loads(rec_path.read_text())
            # v2 recording file carries a _version envelope; there should be
            # exactly one captured call entry alongside it.
            entries = {k: v for k, v in data.items() if not k.startswith("_")}
            assert len(entries) == 1
