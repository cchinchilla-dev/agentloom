"""Check the example Jobs and the Argo CD Application for what a cluster would refuse."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
_EXAMPLE_JOBS = sorted((_DEPLOY / "k8s" / "examples").glob("*.yaml"))


def _jobs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d and d.get("kind") == "Job"]


@pytest.mark.parametrize("path", _EXAMPLE_JOBS, ids=lambda p: p.stem)
def test_example_jobs_pass_the_restricted_pod_security_standard(path: Path) -> None:
    """The ``agentloom`` namespace enforces ``restricted``: a pod without these is refused."""
    jobs = _jobs(path)
    assert jobs
    for job in jobs:
        pod = job["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
        for container in pod["containers"]:
            assert container["securityContext"]["allowPrivilegeEscalation"] is False
            assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


def test_argocd_application_gives_the_chart_a_workflow() -> None:
    """The chart refuses to render without a workflow definition or ConfigMap."""
    app = yaml.safe_load((_DEPLOY / "argocd" / "application.yaml").read_text())
    workflow = app["spec"]["source"]["helm"]["valuesObject"]["workflow"]
    assert workflow.get("existingConfigMap") or workflow.get("definition", "").strip()
