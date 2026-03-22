"""Tests for CLI visualize command."""

from __future__ import annotations

import tempfile

from typer.testing import CliRunner

from agentloom.cli.main import app

runner = CliRunner()

SAMPLE_YAML = """\
name: viz-test
steps:
  - id: a
    type: llm_call
    prompt: "a"
  - id: b
    type: llm_call
    prompt: "b"
  - id: merge
    type: llm_call
    prompt: "merge"
    depends_on: [a, b]
"""


class TestVisualizeCommand:
    def test_ascii_output(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SAMPLE_YAML)
            f.flush()
            result = runner.invoke(app, ["visualize", f.name])
        assert result.exit_code == 0
        assert "viz-test" in result.output
        assert "[LLM: a]" in result.output
        assert "[LLM: merge]" in result.output

    def test_mermaid_output(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SAMPLE_YAML)
            f.flush()
            result = runner.invoke(app, ["visualize", f.name, "--format", "mermaid"])
        assert result.exit_code == 0
        assert "```mermaid" in result.output
        assert "graph TD" in result.output
        assert "a --> merge" in result.output

    def test_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["visualize", "/tmp/no_such_file_1234.yaml"])
        assert result.exit_code != 0

    def test_router_shapes_in_mermaid(self) -> None:
        yaml = """\
name: router-viz
steps:
  - id: classify
    type: llm_call
    prompt: "classify"
  - id: route
    type: router
    depends_on: [classify]
    conditions:
      - expression: "state.x == 1"
        target: handler
    default: fallback
  - id: handler
    type: llm_call
    prompt: "h"
    depends_on: [route]
  - id: fallback
    type: llm_call
    prompt: "f"
    depends_on: [route]
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            result = runner.invoke(app, ["visualize", f.name, "--format", "mermaid"])
        assert result.exit_code == 0
        assert "route{route}" in result.output  # diamond shape for router
