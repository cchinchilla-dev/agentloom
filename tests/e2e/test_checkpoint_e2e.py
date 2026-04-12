"""End-to-end test for the checkpoint lifecycle against a live Ollama instance.

Exercises the full flow: ``agentloom run --checkpoint`` → verify checkpoint
file → ``agentloom runs`` → ``agentloom resume``.

Requires a running Ollama server with the test model pulled.
Excluded from normal test runs via the ``e2e`` marker.

To run locally::

    ollama pull qwen2.5:0.5b
    uv run pytest -m e2e -k checkpoint -v
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

MODEL = "qwen2.5:0.5b"

TWO_STEP_YAML = f"""\
name: e2e-checkpoint
config:
  provider: ollama
  model: {MODEL}
  max_retries: 2
state:
  question: "What is 2+2? Reply with just the number."
steps:
  - id: think
    type: llm_call
    prompt: "Think step by step: {{{{state.question}}}}"
    output: thought
  - id: answer
    type: llm_call
    depends_on: [think]
    prompt: "Given: {{{{state.thought}}}}\\nAnswer in one word: {{{{state.question}}}}"
    output: answer
"""


def _run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run ``agentloom`` CLI with inherited env (includes OLLAMA_BASE_URL)."""
    return subprocess.run(
        ["uv", "run", "agentloom", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )


class TestCheckpointLifecycle:
    """Full run → list → resume cycle through the CLI with live Ollama."""

    def test_run_list_resume(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(TWO_STEP_YAML)
            f.flush()
            workflow_file = f.name

        with tempfile.TemporaryDirectory() as cp_dir:
            # ── Step 1: run with checkpoint ──────────────────────────
            result = _run_cli(
                "run",
                workflow_file,
                "--checkpoint",
                "--checkpoint-dir",
                cp_dir,
                "--lite",
            )
            assert result.returncode == 0, (
                f"run failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "Run ID:" in result.stdout

            # Extract run ID
            match = re.search(r"Run ID:\s+(\S+)", result.stdout)
            assert match, f"Could not find Run ID in: {result.stdout}"
            run_id = match.group(1)

            # Verify checkpoint file on disk
            cp_files = list(Path(cp_dir).glob("*.json"))
            assert len(cp_files) == 1
            assert cp_files[0].stem == run_id

            # Verify checkpoint content
            checkpoint = json.loads(cp_files[0].read_text())
            assert checkpoint["workflow_name"] == "e2e-checkpoint"
            assert checkpoint["status"] == "success"
            assert "think" in checkpoint["completed_steps"]
            assert "answer" in checkpoint["completed_steps"]
            assert "thought" in checkpoint["state"]
            assert "answer" in checkpoint["state"]

            # ── Step 2: list runs ────────────────────────────────────
            list_result = _run_cli("runs", "--checkpoint-dir", cp_dir)
            assert list_result.returncode == 0, f"runs failed: {list_result.stderr}"
            assert run_id in list_result.stdout
            assert "e2e-checkpoint" in list_result.stdout

            # JSON output
            list_json = _run_cli("runs", "--checkpoint-dir", cp_dir, "--json")
            assert list_json.returncode == 0
            runs_data = json.loads(list_json.stdout)
            assert len(runs_data) == 1
            assert runs_data[0]["run_id"] == run_id
            assert runs_data[0]["status"] == "success"

            # ── Step 3: resume (all steps done → no-op success) ──────
            resume_result = _run_cli(
                "resume",
                run_id,
                "--checkpoint-dir",
                cp_dir,
                "--lite",
            )
            assert resume_result.returncode == 0, (
                f"resume failed:\nstdout: {resume_result.stdout}\nstderr: {resume_result.stderr}"
            )
            assert "Resuming workflow" in resume_result.stdout
            assert "e2e-checkpoint" in resume_result.stdout
