"""Tests for CLI resume command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from agentloom.checkpointing.base import CheckpointData
from agentloom.cli.main import app
from agentloom.core.models import (
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)

runner = CliRunner()


def _make_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="resume-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.LLM_CALL,
                prompt="Process: {state.input}",
                output="result_a",
            ),
        ],
    )


def _write_checkpoint(
    cp_dir: Path,
    run_id: str,
    *,
    status: str = "failed",
    completed_steps: list[str] | None = None,
    paused_step_id: str | None = None,
    workflow: WorkflowDefinition | None = None,
    state: dict | None = None,
    step_results: dict | None = None,
) -> None:
    """Write a checkpoint file for testing."""
    wf = workflow or _make_workflow()
    data = CheckpointData(
        workflow_name=wf.name,
        run_id=run_id,
        workflow_definition=wf.model_dump(),
        state=state or {"input": "hello"},
        step_results=step_results or {},
        completed_steps=completed_steps or [],
        status=status,
        paused_step_id=paused_step_id,
        created_at="2026-04-12T18:00:00+00:00",
        updated_at="2026-04-12T18:00:01+00:00",
    )
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / f"{run_id}.json").write_text(data.model_dump_json(indent=2))


class TestResumeCommand:
    def test_missing_run_id(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["resume", "nonexistent", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "No checkpoint found" in (result.output + (result.stderr or ""))

    def test_resume_success(self, tmp_path: Path) -> None:
        """Resume a checkpoint and run to success using mock provider."""
        _write_checkpoint(tmp_path, "run-ok", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            # Wire up MockProvider via the gateway
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire_providers

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    ["resume", "run-ok", "--checkpoint-dir", str(tmp_path), "--lite"],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "Resuming workflow" in result.output
        assert "resume-test" in result.output

    def test_resume_json_output(self, tmp_path: Path) -> None:
        """Resume with --json flag produces JSON output."""
        _write_checkpoint(tmp_path, "run-json", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire_providers

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "resume",
                        "run-json",
                        "--checkpoint-dir",
                        str(tmp_path),
                        "--lite",
                        "--json",
                    ],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"
        # Output should contain valid JSON (after the "Resuming..." line)
        lines = result.output.strip().split("\n")
        # Find the JSON block (skip "Resuming workflow..." line)
        json_start = next(i for i, line in enumerate(lines) if line.strip().startswith("{"))
        json_text = "\n".join(lines[json_start:])
        parsed = json.loads(json_text)
        assert parsed["workflow_name"] == "resume-test"

    def test_resume_with_stream(self, tmp_path: Path) -> None:
        """Resume with --stream flag enables streaming callback."""
        _write_checkpoint(tmp_path, "run-stream", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire_providers

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "resume",
                        "run-stream",
                        "--checkpoint-dir",
                        str(tmp_path),
                        "--lite",
                        "--stream",
                    ],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_resume_with_provider_override(self, tmp_path: Path) -> None:
        """Resume with --provider and --model overrides."""
        _write_checkpoint(tmp_path, "run-override", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire_providers

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "resume",
                        "run-override",
                        "--checkpoint-dir",
                        str(tmp_path),
                        "--lite",
                        "--provider",
                        "mock",
                        "--model",
                        "mock-v2",
                    ],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_resume_failed_workflow_exits_nonzero(self, tmp_path: Path) -> None:
        """Resume a workflow that fails should exit with code 1."""
        _write_checkpoint(tmp_path, "run-fail", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            class FailingProvider(MockProvider):
                async def complete(self, *args, **kwargs):  # type: ignore[override]
                    raise RuntimeError("provider error")

            def _wire_providers(gw: object, default: str) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(FailingProvider(), priority=0)

            mock_setup.side_effect = _wire_providers

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    [
                        "resume",
                        "run-fail",
                        "--checkpoint-dir",
                        str(tmp_path),
                        "--lite",
                    ],
                )

        assert result.exit_code != 0


def _approval_workflow() -> WorkflowDefinition:
    """Workflow with an approval gate."""
    return WorkflowDefinition(
        name="approval-cli-test",
        config=WorkflowConfig(provider="mock", model="mock-model"),
        state={"input": "hello"},
        steps=[
            StepDefinition(
                id="draft",
                type=StepType.LLM_CALL,
                prompt="Draft: {state.input}",
                output="draft_text",
            ),
            StepDefinition(
                id="approve",
                type=StepType.APPROVAL_GATE,
                depends_on=["draft"],
                output="decision",
            ),
            StepDefinition(
                id="send",
                type=StepType.LLM_CALL,
                depends_on=["approve"],
                prompt="Send: {state.draft_text}",
                output="result",
            ),
        ],
    )


class TestResumeApproval:
    def test_resume_approve_flag(self, tmp_path: Path) -> None:
        wf = _approval_workflow()
        _write_checkpoint(
            tmp_path,
            "approve-ok",
            status="paused",
            paused_step_id="approve",
            workflow=wf,
            completed_steps=["draft"],
            state={"input": "hello", "draft_text": "Mock response"},
            step_results={
                "draft": {
                    "step_id": "draft",
                    "status": "success",
                    "output": "Mock response",
                    "duration_ms": 10.0,
                },
            },
        )

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
                        "resume",
                        "approve-ok",
                        "--checkpoint-dir",
                        str(tmp_path),
                        "--lite",
                        "--approve",
                    ],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "decision=approved" in result.output

    def test_resume_reject_flag(self, tmp_path: Path) -> None:
        wf = _approval_workflow()
        _write_checkpoint(
            tmp_path,
            "reject-ok",
            status="paused",
            paused_step_id="approve",
            workflow=wf,
            completed_steps=["draft"],
            state={"input": "hello", "draft_text": "Mock response"},
            step_results={
                "draft": {
                    "step_id": "draft",
                    "status": "success",
                    "output": "Mock response",
                    "duration_ms": 10.0,
                },
            },
        )

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
                        "resume",
                        "reject-ok",
                        "--checkpoint-dir",
                        str(tmp_path),
                        "--lite",
                        "--reject",
                    ],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "decision=rejected" in result.output

    def test_approve_reject_mutual_exclusion(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "both-flags", status="paused", paused_step_id="gate")

        result = runner.invoke(
            app,
            [
                "resume",
                "both-flags",
                "--checkpoint-dir",
                str(tmp_path),
                "--approve",
                "--reject",
            ],
        )

        assert result.exit_code != 0
        assert "Cannot use --approve and --reject together" in (
            result.output + (result.stderr or "")
        )

    def test_approve_on_non_paused_run_fails(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "not-paused", status="failed", paused_step_id=None)

        result = runner.invoke(
            app,
            ["resume", "not-paused", "--checkpoint-dir", str(tmp_path), "--approve"],
        )

        assert result.exit_code != 0
        assert "not paused at an approval gate" in (
            result.output + (result.stderr or "")
        )
