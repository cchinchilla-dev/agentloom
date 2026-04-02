"""CLI functional test for array index support in state paths.

Validates that the CLI can parse and validate a workflow using array
index syntax without errors.  Marked ``e2e`` so it is excluded from
normal ``pytest`` runs.
"""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.e2e

EXAMPLE = "examples/27_array_index.yaml"


def test_validate_array_index_workflow() -> None:
    """``agentloom validate`` should succeed for a workflow with array state."""
    result = subprocess.run(
        ["agentloom", "validate", EXAMPLE],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"validate failed: {result.stderr}"
