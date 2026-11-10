"""Tests for CLI resume command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from agentloom.checkpointing.base import CheckpointData
from agentloom.cli.main import app
from agentloom.core.models import (
    StateKeyConfig,
    StepDefinition,
    StepType,
    WorkflowConfig,
    WorkflowDefinition,
)

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def _make_workflow(provider: str = "openai") -> WorkflowDefinition:
    return WorkflowDefinition(
        name="resume-test",
        config=WorkflowConfig(provider=provider, model="mock-model"),
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
    spent_usd: float = 0.0,
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
        spent_usd=spent_usd,
        created_at="2026-04-12T18:00:00+00:00",
        updated_at="2026-04-12T18:00:01+00:00",
    )
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / f"{run_id}.json").write_text(data.model_dump_json(indent=2))


class TestResumeCommand:
    def test_resume_imports_the_projects_schemas(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """As ``run``: a response_schema model of the working directory imports."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(Path.cwd()))])
        runner.invoke(app, ["resume", "nonexistent", "--checkpoint-dir", str(tmp_path)])
        assert sys.path[0] == str(Path.cwd())

    def test_missing_run_id(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["resume", "nonexistent", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "No checkpoint found" in (result.output + (result.stderr or ""))

    def test_invalid_run_id_exits_without_a_traceback(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["resume", "../x", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "Invalid run_id" in (result.output + (result.stderr or ""))

    def test_corrupted_checkpoint_exits_without_a_traceback(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{not json")
        result = runner.invoke(app, ["resume", "broken", "--checkpoint-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "unreadable or corrupted" in (result.output + (result.stderr or ""))

    def test_a_mock_recording_that_is_gone_exits_without_a_traceback(self, tmp_path: Path) -> None:
        workflow = _make_workflow("mock")
        workflow.config.responses_file = str(tmp_path / "gone.json")
        _write_checkpoint(tmp_path, "run-mock", workflow=workflow)
        result = runner.invoke(
            app, ["resume", "run-mock", "--checkpoint-dir", str(tmp_path), "--lite"]
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "gone.json" in (result.output + (result.stderr or ""))

    def test_resume_success(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-ok", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str, config_path: object = None) -> None:
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
        _write_checkpoint(tmp_path, "run-json", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str, config_path: object = None) -> None:
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
        # The JSON block follows the "Resuming workflow..." line.
        lines = result.output.strip().split("\n")
        json_start = next(i for i, line in enumerate(lines) if line.strip().startswith("{"))
        json_text = "\n".join(lines[json_start:])
        parsed = json.loads(json_text)
        assert parsed["workflow_name"] == "resume-test"

    def test_resume_with_stream(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run-stream", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str, config_path: object = None) -> None:
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
        _write_checkpoint(tmp_path, "run-override", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            def _wire_providers(gw: object, default: str, config_path: object = None) -> None:
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
                        "anthropic",
                        "--model",
                        "mock-v2",
                    ],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_resume_failed_workflow_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agentloom.core.engine.compute_backoff", lambda *_args: 0.0)
        _write_checkpoint(tmp_path, "run-fail", status="failed")

        with patch("agentloom.cli.run._setup_providers") as mock_setup:
            from tests.conftest import MockProvider

            class FailingProvider(MockProvider):
                async def complete(self, *args, **kwargs):  # type: ignore[override]
                    raise RuntimeError("provider error")

            def _wire_providers(gw: object, default: str, config_path: object = None) -> None:
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

    def test_resume_runs_a_mock_workflow_offline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        _write_checkpoint(tmp_path, "run-mock", workflow=_make_workflow("mock"))
        argv = ["resume", "run-mock", "--checkpoint-dir", str(tmp_path), "--lite", "--json"]
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output[result.output.index("{") :])
        assert parsed["final_state"]["result_a"] == "Mock response"

    def test_resume_generic_openai_without_base_url(self, tmp_path: Path) -> None:
        """``--provider generic_openai`` on a workflow without ``base_url`` fails cleanly."""
        _write_checkpoint(tmp_path, "run-generic", status="failed")
        result = runner.invoke(
            app,
            [
                "resume",
                "run-generic",
                "--checkpoint-dir",
                str(tmp_path),
                "--provider",
                "generic_openai",
                "--lite",
            ],
        )
        assert result.exit_code == 1
        assert "needs config.base_url" in (result.output + (result.stderr or ""))


def _approval_workflow() -> WorkflowDefinition:
    """Workflow with an approval gate."""
    return WorkflowDefinition(
        name="approval-cli-test",
        config=WorkflowConfig(provider="openai", model="mock-model"),
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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
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

        assert result.exit_code == 2
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
        assert "not paused at an approval gate" in (result.output + (result.stderr or ""))

    def test_a_second_gate_prints_the_run_id_in_its_hint(self, tmp_path: Path) -> None:
        wf = _approval_workflow()
        wf.steps.insert(
            2,
            StepDefinition(
                id="approve_again",
                type=StepType.APPROVAL_GATE,
                depends_on=["approve"],
                output="second_decision",
            ),
        )
        wf.steps[3].depends_on = ["approve_again"]
        _write_checkpoint(
            tmp_path,
            "two-gates",
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

            def _wire(gw: object, default: str, config_path: object = None) -> None:
                from agentloom.providers.gateway import ProviderGateway

                assert isinstance(gw, ProviderGateway)
                gw.register(MockProvider(), priority=0)

            mock_setup.side_effect = _wire

            with patch("agentloom.cli.run._setup_observer", return_value=None):
                result = runner.invoke(
                    app,
                    ["resume", "two-gates", "--checkpoint-dir", str(tmp_path), "--approve"],
                )

        assert result.exit_code == 0, f"stdout: {result.output}"
        assert "Step 'approve_again' is waiting for a decision" in result.output
        assert "agentloom resume two-gates --approve" in result.output
        assert "<run_id>" not in result.output

    def test_resume_replays_the_recording_of_a_mock_responses_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        workflow = tmp_path / "wf.yaml"
        workflow.write_text(yaml.safe_dump(_approval_workflow().model_dump(mode="json")))
        (tmp_path / "rec.json").write_text(
            json.dumps({step: {"content": f"recorded {step}"} for step in ("draft", "send")})
        )
        checkpoints = tmp_path / "checkpoints"
        monkeypatch.chdir(tmp_path)
        run = runner.invoke(
            app,
            ["run", str(workflow), "--mock-responses", "rec.json", "--lite"]
            + ["--checkpoint", "--checkpoint-dir", str(checkpoints)],
        )
        assert run.exit_code == 0, run.output
        (saved,) = checkpoints.glob("*.json")
        monkeypatch.chdir(checkpoints)  # a relative recording path would not resolve here
        argv = ["resume", saved.stem, "--checkpoint-dir", str(checkpoints), "--approve"]
        result = runner.invoke(app, [*argv, "--lite", "--json"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output[result.output.index("{") :])
        assert parsed["final_state"]["result"] == "recorded send"


class TestResumeState:
    """``resume --state`` overrides the checkpoint's state, parsed as ``run --state`` is."""

    _SENTINEL = "<REDACTED:sha256=0123456789abcdef>"

    def _resume(self, tmp_path: Path, *argv: str) -> tuple[object, list[dict]]:
        """Resume ``run1`` against a provider that keeps what it was sent."""
        from tests.conftest import MockProvider

        provider = MockProvider()

        def _wire(gw: object, default: str, config_path: object = None) -> None:
            gw.register(provider, priority=0)  # type: ignore[attr-defined]

        with (
            patch("agentloom.cli.run._setup_providers", side_effect=_wire),
            patch("agentloom.cli.run._setup_observer", return_value=None),
        ):
            result = runner.invoke(
                app, ["resume", "run1", "--checkpoint-dir", str(tmp_path), "--lite", *argv]
            )
        return result, provider.calls

    def test_puts_back_a_redacted_key(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run1", state={"input": self._SENTINEL})

        result, calls = self._resume(tmp_path, "--state", "input=dummy-secret")

        assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
        assert "Process: dummy-secret" in calls[0]["messages"][-1]["content"]
        assert "redacted state keys" not in result.stderr  # type: ignore[attr-defined]

    def test_without_it_the_step_reads_the_sentinel(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run1", state={"input": self._SENTINEL})

        result, calls = self._resume(tmp_path)

        assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
        assert self._SENTINEL in calls[0]["messages"][-1]["content"]

    def test_the_value_put_back_is_redacted_again_on_disk(self, tmp_path: Path) -> None:
        workflow = _make_workflow()
        workflow.state_schema = {"input": StateKeyConfig(redact=True)}
        _write_checkpoint(tmp_path, "run1", workflow=workflow, state={"input": self._SENTINEL})

        result, _ = self._resume(tmp_path, "--state", "input=dummy-secret")

        assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
        assert "dummy-secret" not in (tmp_path / "run1.json").read_text()

    def test_values_are_parsed_as_in_run(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run1")

        result, _ = self._resume(tmp_path, "--state", "items=[1, 2]", "--json")

        assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
        assert json.loads(result.stdout)["final_state"]["items"] == [1, 2]  # type: ignore[attr-defined]

    def test_the_key_is_taken_as_written(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "run1")

        result, _ = self._resume(tmp_path, "--state", "a.b=1", "--json")

        assert json.loads(result.stdout)["final_state"]["a.b"] == 1  # type: ignore[attr-defined]

    def test_without_an_equals_sign_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["resume", "nope", "--checkpoint-dir", str(tmp_path), "--state", "bad"]
        )
        assert result.exit_code == 2
        assert "key=value" in result.stderr


class TestResumeBudget:
    def _over_budget(self, tmp_path: Path) -> None:
        workflow = _make_workflow("mock")
        workflow.config.budget_usd = 0.5
        _write_checkpoint(
            tmp_path, "run1", workflow=workflow, status="budget_exceeded", spent_usd=0.5
        )

    def _resume(self, tmp_path: Path, *argv: str) -> Any:
        return runner.invoke(
            app, ["resume", "run1", "--checkpoint-dir", str(tmp_path), "--lite", "--json", *argv]
        )

    def test_a_run_over_its_budget_stops_again_without_it(self, tmp_path: Path) -> None:
        self._over_budget(tmp_path)
        result = self._resume(tmp_path)
        assert result.exit_code == 1
        assert json.loads(result.stdout)["status"] == "budget_exceeded"

    def test_a_higher_limit_lets_it_go_on(self, tmp_path: Path) -> None:
        self._over_budget(tmp_path)
        result = self._resume(tmp_path, "--budget", "5")
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["status"] == "success"
        saved = json.loads((tmp_path / "run1.json").read_text())
        assert saved["workflow_definition"]["config"]["budget_usd"] == 5.0

    def test_the_spend_so_far_counts_against_it(self, tmp_path: Path) -> None:
        self._over_budget(tmp_path)
        result = self._resume(tmp_path, "-b", "0.4")
        assert result.exit_code == 1
        assert json.loads(result.stdout)["status"] == "budget_exceeded"


GATED_GENERIC_YAML = """\
name: gated
config: {provider: generic_openai, model: local, base_url: "http://127.0.0.1:9/v1"}
steps:
  - {id: gate, type: approval_gate, output: decision}
  - {id: answer, type: llm_call, prompt: hi, output: a, depends_on: [gate]}
"""

GATED_YAML = """\
name: gated
config: {{{config}}}
steps:
  - {{id: gate, type: approval_gate, output: decision}}
  - {{id: answer, type: llm_call, prompt: hi, output: a, depends_on: [gate]}}
"""


def _paused_run(tmp_path: Path, yaml_text: str, *flags: str) -> str:
    """Run a workflow that stops at its gate, and return the run id."""
    workflow = tmp_path / "wf.yaml"
    workflow.write_text(yaml_text)
    args = ["run", str(workflow), "--checkpoint", "--checkpoint-dir", str(tmp_path), "--lite"]
    result = runner.invoke(app, [*args, "--json", *flags])
    assert result.exit_code == 0, result.output
    return str(json.loads(result.stdout)["run_id"])


class TestResumeAfterLeavingGenericOpenai:
    def test_a_run_moved_with_provider_resumes(self, tmp_path: Path) -> None:
        run_id = _paused_run(tmp_path, GATED_GENERIC_YAML, "--provider", "mock")
        args = ["resume", run_id, "--approve", "--checkpoint-dir", str(tmp_path), "--lite"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output

    def test_a_run_on_a_recording_resumes(self, tmp_path: Path) -> None:
        recording = tmp_path / "rec.json"
        recording.write_text('{"_version": 2, "answer": {"content": "x"}}')
        run_id = _paused_run(tmp_path, GATED_GENERIC_YAML, "--mock-responses", str(recording))
        args = ["resume", run_id, "--approve", "--checkpoint-dir", str(tmp_path), "--lite"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output


class TestResumeProviderFlag:
    @staticmethod
    def _resume_on(tmp_path: Path, run_id: str, url: str, reply: dict[str, Any], flag: str) -> Any:
        import httpx
        import respx

        args = ["resume", run_id, "--approve", "--checkpoint-dir", str(tmp_path), "--lite"]
        with respx.mock:
            route = respx.post(url).mock(return_value=httpx.Response(200, json=reply))
            result = runner.invoke(app, [*args, "--provider", flag])
        assert result.exit_code == 0, result.output
        return json.loads(route.calls.last.request.content)

    def test_a_model_the_file_did_not_set_follows_the_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
        run_id = _paused_run(tmp_path, GATED_YAML.format(config="provider: mock"))
        reply = {
            "content": [{"type": "text", "text": "x"}],
            "model": "m",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        sent = self._resume_on(
            tmp_path, run_id, "https://api.anthropic.com/v1/messages", reply, "anthropic"
        )
        assert sent["model"] == "claude-haiku-4-5-20251001"

    def test_a_model_the_file_set_stays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
        run_id = _paused_run(tmp_path, GATED_YAML.format(config="provider: mock, model: gpt-4o"))
        reply = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "model": "m",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        sent = self._resume_on(
            tmp_path, run_id, "https://api.openai.com/v1/chat/completions", reply, "openai"
        )
        assert sent["model"] == "gpt-4o"


class TestSubworkflowCheckpoints:
    def test_its_own_checkpoint_is_refused_naming_the_root_run(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, "abc123.child", status="paused", paused_step_id="step_a")
        args = ["resume", "abc123.child", "--checkpoint-dir", str(tmp_path), "--lite"]
        result = runner.invoke(app, args)
        assert result.exit_code == 2
        assert "'abc123'" in result.stderr

    def test_a_dotted_id_with_no_checkpoint_is_unknown(self, tmp_path: Path) -> None:
        args = ["resume", "abc123.child", "--checkpoint-dir", str(tmp_path), "--lite"]
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert "No checkpoint found" in result.stderr
