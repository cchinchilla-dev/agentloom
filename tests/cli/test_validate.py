"""Tests for the CLI validate command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentloom.cli.main import app

runner = CliRunner()


class TestValidateCommand:
    """Test the 'agentloom validate' CLI command."""

    def test_validate_valid_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
name: valid-workflow
version: "1.0"
config:
  provider: mock
  model: mock-model
state:
  question: "test"
steps:
  - id: answer
    type: llm_call
    prompt: "Answer: {state.question}"
    output: answer
"""
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(yaml_content)

        result = runner.invoke(app, ["validate", str(yaml_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()
        assert "valid-workflow" in result.output

    def test_validate_valid_yaml_shows_details(self, tmp_path: Path) -> None:
        yaml_content = """
name: detail-test
version: "2.0"
config:
  provider: openai
  model: gpt-4o-mini
steps:
  - id: step_a
    type: llm_call
    prompt: "A"
  - id: step_b
    type: llm_call
    prompt: "B"
    depends_on: [step_a]
"""
        yaml_file = tmp_path / "detail.yaml"
        yaml_file.write_text(yaml_content)

        result = runner.invoke(app, ["validate", str(yaml_file)])
        assert result.exit_code == 0
        assert "Steps:" in result.output
        assert "Layers:" in result.output

    def test_validate_invalid_yaml_missing_ref(self, tmp_path: Path) -> None:
        yaml_content = """
name: invalid-ref
steps:
  - id: a
    type: llm_call
    prompt: "a"
    depends_on: [nonexistent]
"""
        yaml_file = tmp_path / "invalid_ref.yaml"
        yaml_file.write_text(yaml_content)

        result = runner.invoke(app, ["validate", str(yaml_file)])
        assert result.exit_code != 0

    def test_validate_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        bad_yaml = "name: test\nsteps:\n  - id: [invalid"
        yaml_file = tmp_path / "bad_syntax.yaml"
        yaml_file.write_text(bad_yaml)

        result = runner.invoke(app, ["validate", str(yaml_file)])
        assert result.exit_code != 0

    def test_validate_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["validate", "/nonexistent/path/workflow.yaml"])
        assert result.exit_code != 0

    def test_validate_workflow_with_parallel_steps(self, tmp_path: Path) -> None:
        yaml_content = """
name: parallel-test
config:
  provider: mock
  model: mock-model
steps:
  - id: a
    type: llm_call
    prompt: "A"
  - id: b
    type: llm_call
    prompt: "B"
  - id: c
    type: llm_call
    prompt: "C"
    depends_on: [a, b]
"""
        yaml_file = tmp_path / "parallel.yaml"
        yaml_file.write_text(yaml_content)

        result = runner.invoke(app, ["validate", str(yaml_file)])
        assert result.exit_code == 0
        assert "parallel" in result.output.lower()

    def test_validate_missing_steps_field(self, tmp_path: Path) -> None:
        yaml_content = """
name: no-steps
config:
  provider: mock
"""
        yaml_file = tmp_path / "no_steps.yaml"
        yaml_file.write_text(yaml_content)

        result = runner.invoke(app, ["validate", str(yaml_file)])
        assert result.exit_code != 0
