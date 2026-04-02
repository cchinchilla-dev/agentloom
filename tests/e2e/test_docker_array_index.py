"""Docker functional test for array index support.

Builds the dev image and runs the array-index unit tests inside it.
Skipped when Docker is not available.  Marked ``e2e``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.e2e

DOCKER = shutil.which("docker")


@pytest.mark.skipif(DOCKER is None, reason="docker not available")
def test_array_index_tests_in_docker() -> None:
    """Unit tests for array index paths pass inside the dev container."""
    # Build the dev image
    build = subprocess.run(
        ["docker", "build", "--target", "dev", "-t", "agentloom:dev-test", "."],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, f"docker build failed: {build.stderr}"

    # Run the state array-index tests inside the container
    run = subprocess.run(
        [
            "docker", "run", "--rm", "agentloom:dev-test",
            "pytest", "tests/core/test_state.py::TestArrayIndexPaths", "-v",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, f"tests failed in docker: {run.stdout}\n{run.stderr}"
