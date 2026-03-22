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
