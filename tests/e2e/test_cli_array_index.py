"""CLI functional test for array index support in state paths."""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.e2e

EXAMPLE = "examples/27_array_index.yaml"


def test_validate_array_index_workflow() -> None:
    result = subprocess.run(
        ["agentloom", "validate", EXAMPLE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"validate failed: {result.stderr}"
