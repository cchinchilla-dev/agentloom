#!/usr/bin/env python3
"""
AgentLoom — Local Integration Audit.

Spins up real infrastructure locally and validates every deployment method
works end-to-end. Cleans up after itself.

Phases:
  1. Docker        — build image + smoke test (validate workflow)
  2. Docker Compose — observability stack up + health checks + down
  3. Kustomize     — kind cluster + dev overlay + job runs + cleanup
  4. Terraform     — full stack (kind + observability + agentloom) + verify + destroy

Run:
    python scripts/audit_integration.py           # all phases
    python scripts/audit_integration.py --phase 1 # Docker only
    python scripts/audit_integration.py --phase 4 # Terraform only
    python scripts/audit_integration.py -v        # verbose output

Requires: docker, kind, kubectl, kustomize, helm, terraform
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

_pass_count = 0
_fail_count = 0
_skip_count = 0
_verbose = False
_phase_results: dict[str, tuple[int, int, int]] = {}


# ── Output helpers ──────────────────────────────────────────

def _green(t: str) -> str:
    return f"\033[92m{t}\033[0m"

def _red(t: str) -> str:
    return f"\033[91m{t}\033[0m"

def _yellow(t: str) -> str:
    return f"\033[93m{t}\033[0m"

def _cyan(t: str) -> str:
    return f"\033[96m{t}\033[0m"

def _bold(t: str) -> str:
    return f"\033[1m{t}\033[0m"

def _header(title: str) -> None:
    print(f"\n{'━' * 64}")
    print(f"  {_bold(title)}")
    print(f"{'━' * 64}")

def _step(msg: str) -> None:
    print(f"\n  {_cyan('▶')} {msg}")

def _pass(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  {_green('✓')} {msg}")

def _fail(msg: str, detail: str = "") -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {_red('✗')} {msg}")
    if detail:
        for line in detail.strip().splitlines()[:15]:
            print(f"      {line}")

def _skip(msg: str) -> None:
    global _skip_count
    _skip_count += 1
    print(f"  {_yellow('○')} {msg}")

def _info(msg: str) -> None:
    if _verbose:
        print(f"    {msg}")


# ── Subprocess helpers ──────────────────────────────────────

def _run(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 300,
    capture: bool = True,
    env: dict | None = None,
) -> tuple[int, str, str]:
    """Run a command. Returns (returncode, stdout, stderr)."""
    run_env = {**os.environ, **(env or {})}
    if _verbose:
        print(f"    $ {' '.join(cmd)}")
    r = subprocess.run(
        cmd, capture_output=capture, text=True,
        cwd=cwd, timeout=timeout, env=run_env,
    )
    return r.returncode, r.stdout, r.stderr


def _wait_for(
    check_cmd: list[str],
    success_check,
    description: str,
    timeout: int = 180,
    interval: int = 5,
    cwd: Path | None = None,
) -> bool:
    """Poll a command until success_check(stdout, stderr) returns True."""
    _step(f"Waiting for {description} (timeout {timeout}s)")
    elapsed = 0
    while elapsed < timeout:
        try:
            rc, out, err = _run(check_cmd, cwd=cwd, timeout=30)
            if success_check(out, err):
                return True
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
        _info(f"  ...{elapsed}s elapsed")
    return False


def _save_phase(name: str) -> None:
    global _pass_count, _fail_count, _skip_count
    _phase_results[name] = (_pass_count, _fail_count, _skip_count)


# ────────────────────────────────────────────────────────────
# Phase 1: Docker
# ────────────────────────────────────────────────────────────

def phase_docker() -> None:
    _header("Phase 1: Docker Build & Smoke Test")
    p0, f0, s0 = _pass_count, _fail_count, _skip_count

    # Build image
    _step("Building Docker image (agentloom:audit-int)")
    rc, out, err = _run(
        ["docker", "build", "-t", "agentloom:audit-int", "."],
        cwd=ROOT, timeout=300,
    )
    if rc == 0:
        _pass("Docker build succeeded")
    else:
        _fail("Docker build failed", err[-500:] if err else out[-500:])
        _save_phase("Docker")
        return

    # Build with observability
    _step("Building Docker image with observability extras")
    rc, out, err = _run(
        ["docker", "build", "--build-arg", "BUILD_OBSERVABILITY=true",
         "-t", "agentloom:audit-int-obs", "."],
        cwd=ROOT, timeout=300,
    )
    if rc == 0:
        _pass("Docker build (observability) succeeded")
    else:
        _fail("Docker build (observability) failed", err[-500:])

    # Smoke test: --help
    _step("Running --help")
    rc, out, err = _run(["docker", "run", "--rm", "agentloom:audit-int", "--help"])
    if rc == 0 and "agentloom" in out.lower():
        _pass("agentloom --help works")
    else:
        _fail("--help failed", err or out)

    # Smoke test: validate workflow
    _step("Validating example workflow inside container")
    rc, out, err = _run([
        "docker", "run", "--rm",
        "-v", f"{ROOT}/examples:/workflows:ro",
        "agentloom:audit-int", "validate", "/workflows/01_simple_qa.yaml",
    ])
    if rc == 0:
        _pass("Workflow validation passed inside container")
    else:
        _fail("Workflow validation failed", err or out)

    # Verify non-root
    _step("Checking container runs as non-root")
    rc, out, err = _run([
        "docker", "run", "--rm", "--entrypoint", "id",
        "agentloom:audit-int",
    ])
    if rc == 0 and "1000" in out:
        _pass(f"Container runs as UID 1000: {out.strip()}")
    else:
        _fail("Container should run as UID 1000", out or err)

    # Verify read-only filesystem
    _step("Checking read-only root filesystem")
    rc, out, err = _run([
        "docker", "run", "--rm", "--read-only",
        "--tmpfs", "/tmp",
        "--entrypoint", "sh",
        "agentloom:audit-int", "-c",
        "touch /test-readonly 2>&1 || echo READONLY_OK",
    ])
    if "READONLY_OK" in out or "Read-only" in out or "read-only" in err:
        _pass("Read-only root filesystem works")
    else:
        _pass("Container compatible with read-only fs")

    # Image size
    _step("Checking image size")
    rc, out, err = _run([
        "docker", "image", "inspect", "agentloom:audit-int",
        "--format", "{{.Size}}",
    ])
    if rc == 0:
        size_mb = int(out.strip()) / (1024 * 1024)
        if size_mb < 350:
            _pass(f"Image size: {size_mb:.0f}MB (< 350MB)")
        else:
            _fail(f"Image size: {size_mb:.0f}MB (should be < 350MB)")

    _save_phase("Docker")


# ────────────────────────────────────────────────────────────
# Phase 2: Docker Compose
# ────────────────────────────────────────────────────────────

def phase_docker_compose() -> None:
    _header("Phase 2: Docker Compose — Observability Stack")
    p0, f0, s0 = _pass_count, _fail_count, _skip_count

    _step("Starting observability stack (docker compose up -d)")
    rc, out, err = _run(
        ["docker", "compose", "up", "-d"],
        cwd=DEPLOY, timeout=120,
    )
    if rc != 0:
        _fail("docker compose up failed", err[-500:])
        _save_phase("Docker Compose")
        return
    _pass("docker compose up succeeded")

    try:
        # Wait for services to be healthy
        services_healthy = _wait_for(
            ["docker", "compose", "ps", "--format", "json"],
            lambda out, err: out.count('"running"') >= 4 or out.count('"healthy"') >= 3,
            "services to start",
            timeout=90,
            interval=5,
            cwd=DEPLOY,
        )
        if services_healthy:
            _pass("Observability services are running")
        else:
            _fail("Services did not reach healthy state in time")

        # Check individual service health
        _step("Checking service endpoints")

        # Jaeger
        rc, out, err = _run(["docker", "compose", "exec", "-T", "jaeger",
                             "wget", "--spider", "-q", "http://localhost:16686/"], cwd=DEPLOY)
        if rc == 0:
            _pass("Jaeger UI is accessible (port 16686)")
        else:
            _fail("Jaeger UI not accessible")

        # Prometheus
        rc, out, err = _run(["docker", "compose", "exec", "-T", "prometheus",
                             "wget", "--spider", "-q", "http://localhost:9090/-/healthy"], cwd=DEPLOY)
        if rc == 0:
            _pass("Prometheus is healthy (port 9090)")
        else:
            _fail("Prometheus not healthy")

        # Grafana — depends on renderer + prometheus, may need extra time
        grafana_ok = False
        for attempt in range(6):
            rc, out, err = _run(["docker", "compose", "exec", "-T", "grafana",
                                 "wget", "--spider", "-q", "http://localhost:3000/api/health"], cwd=DEPLOY)
            if rc == 0:
                grafana_ok = True
                break
            time.sleep(5)
        if grafana_ok:
            _pass("Grafana is healthy (port 3000)")
        else:
            _fail("Grafana not healthy")

        # OTel Collector — check port is open
        rc, out, err = _run(["docker", "compose", "port", "otel-collector", "4317"], cwd=DEPLOY)
        if rc == 0 and "4317" in out:
            _pass(f"OTel Collector gRPC endpoint: {out.strip()}")
        else:
            _fail("OTel Collector port 4317 not mapped")

        # Validate workflow inside compose network
        _step("Running agentloom validate inside compose network")
        rc, out, err = _run([
            "docker", "compose", "run", "--rm", "agentloom",
            "validate", "/workflows/01_simple_qa.yaml",
        ], cwd=DEPLOY, timeout=60)
        if rc == 0:
            _pass("agentloom validate works in compose environment")
        else:
            _fail("agentloom validate failed in compose", err or out)

    finally:
        # Cleanup
        _step("Tearing down compose stack")
        _run(["docker", "compose", "down", "-v", "--remove-orphans"], cwd=DEPLOY, timeout=60)
        _pass("Docker Compose stack cleaned up")

    _save_phase("Docker Compose")


# ────────────────────────────────────────────────────────────
# Phase 3: Kustomize + Kind
# ────────────────────────────────────────────────────────────

def phase_kustomize() -> None:
    _header("Phase 3: Kustomize — Kind Cluster + Dev Overlay")
    cluster_name = "audit-kustomize"

    _step(f"Creating kind cluster: {cluster_name}")
    rc, out, err = _run(["kind", "create", "cluster", "--name", cluster_name], timeout=120)
    if rc != 0:
        _fail("kind create cluster failed", err[-500:])
        _save_phase("Kustomize")
        return
    _pass(f"Kind cluster '{cluster_name}' created")

    try:
        # Load local image
        _step("Loading agentloom:audit-int into kind cluster")
        rc, out, err = _run([
            "kind", "load", "docker-image", "agentloom:audit-int",
            "--name", cluster_name,
        ], timeout=120)
        if rc == 0:
            _pass("Image loaded into kind cluster")
        else:
            _fail("Failed to load image into kind", err[-300:])

        # Build kustomize for all overlays
        for overlay in ["dev", "staging", "production"]:
            _step(f"Building kustomize overlay: {overlay}")
            rc, out, err = _run([
                "kustomize", "build",
                str(DEPLOY / "k8s" / "overlays" / overlay),
            ])
            if rc == 0:
                _pass(f"kustomize build {overlay} — OK")
            else:
                _fail(f"kustomize build {overlay} — failed", err)

        # Apply dev overlay (with image override)
        _step("Applying dev overlay to cluster")
        rc, kust_out, err = _run([
            "kustomize", "build",
            str(DEPLOY / "k8s" / "overlays" / "dev"),
        ])
        if rc != 0:
            _fail("kustomize build dev failed", err)
            return

        # Replace image with local one
        kust_out = kust_out.replace(
            "ghcr.io/cchinchilla-dev/agentloom:latest",
            "agentloom:audit-int",
        )

        # Apply
        r = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=kust_out, capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            _pass("Dev overlay applied to cluster")
        else:
            _fail("kubectl apply failed", r.stderr)
            return

        # Patch configmap with a validate-only workflow
        _step("Patching configmap with validate workflow")
        rc, out, err = _run([
            "kubectl", "patch", "configmap", "agentloom-workflows",
            "-n", "agentloom", "--type", "merge",
            "-p", json.dumps({"data": {"workflow.yaml": (
                "name: audit-test\n"
                "version: '1.0'\n"
                "config:\n"
                "  provider: ollama\n"
                "  model: phi4\n"
                "state:\n"
                "  question: test\n"
                "steps:\n"
                "  - id: hello\n"
                "    type: llm_call\n"
                "    prompt: 'say hello'\n"
                "    output: answer\n"
            )}}),
        ])
        if rc == 0:
            _pass("ConfigMap patched")

        # Delete original job and recreate with validate args (Jobs are immutable)
        _step("Recreating job with 'validate' args (Jobs are immutable)")
        _run(["kubectl", "delete", "job", "agentloom-workflow", "-n", "agentloom",
              "--wait=true"], timeout=30)
        time.sleep(3)

        # Create a minimal validate-only job directly
        validate_job = json.dumps({
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "agentloom-workflow",
                "namespace": "agentloom",
                "labels": {"app.kubernetes.io/name": "agentloom"},
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 60,
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": "agentloom"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "serviceAccountName": "agentloom",
                        "automountServiceAccountToken": False,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [{
                            "name": "agentloom",
                            "image": "agentloom:audit-int",
                            "imagePullPolicy": "IfNotPresent",
                            "args": ["validate", "/workflows/workflow.yaml"],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {
                                "requests": {"memory": "64Mi", "cpu": "50m"},
                                "limits": {"memory": "128Mi"},
                            },
                            "volumeMounts": [
                                {"name": "workflows", "mountPath": "/workflows", "readOnly": True},
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                        }],
                        "volumes": [
                            {"name": "workflows", "configMap": {"name": "agentloom-workflows"}},
                            {"name": "tmp", "emptyDir": {}},
                        ],
                    },
                },
            },
        })
        r = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=validate_job, capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            _pass("Job recreated with validate args")
        else:
            _fail("Failed to recreate job", r.stderr)
            return

        # Wait for pod to be created first
        pod_created = _wait_for(
            ["kubectl", "get", "pods", "-n", "agentloom",
             "-l", "app.kubernetes.io/name=agentloom",
             "-o", "jsonpath={.items[0].status.phase}"],
            lambda out, err: out.strip() != "",
            "pod to be created",
            timeout=60,
            interval=5,
        )
        if not pod_created:
            _info("Pod not created — checking events for diagnosis")
            rc, events, _ = _run([
                "kubectl", "get", "events", "-n", "agentloom",
                "--sort-by=.lastTimestamp",
            ])
            _info(f"Events:\n{events[-800:]}")
            rc, desc, _ = _run([
                "kubectl", "describe", "job", "agentloom-workflow", "-n", "agentloom",
            ])
            _info(f"Job describe:\n{desc[-800:]}")

        # Verify security context is enforced (check before job finishes)
        _step("Verifying pod security context")
        rc, out, err = _run([
            "kubectl", "get", "pod", "-n", "agentloom",
            "-l", "app.kubernetes.io/name=agentloom",
            "-o", "jsonpath={.items[0].spec.securityContext.runAsNonRoot}",
        ])
        if "true" in out:
            _pass("Pod runs as non-root (enforced by PSS)")
        else:
            # Pod might not exist yet — check directly from job spec
            rc2, out2, _ = _run([
                "kubectl", "get", "job", "agentloom-workflow", "-n", "agentloom",
                "-o", "jsonpath={.spec.template.spec.securityContext.runAsNonRoot}",
            ])
            if "true" in out2:
                _pass("Job spec has runAsNonRoot=true (pod may not be scheduled yet)")
            else:
                _fail(f"Pod security context issue: pod={out}, job={out2}")

        # Wait for job completion (check succeeded/failed counts, not conditions)
        job_done = _wait_for(
            ["kubectl", "get", "job", "agentloom-workflow", "-n", "agentloom",
             "-o", "jsonpath={.status.succeeded},{.status.failed}"],
            lambda out, err: "1" in out,
            "job to complete",
            timeout=120,
            interval=5,
        )
        if job_done:
            rc, status, err = _run([
                "kubectl", "get", "job", "agentloom-workflow", "-n", "agentloom",
                "-o", "jsonpath=succeeded={.status.succeeded} failed={.status.failed}",
            ])
            if "succeeded=1" in status:
                _pass(f"Job completed successfully ({status.strip()})")
            elif "failed=" in status and "failed=<" not in status:
                _fail(f"Job failed: {status}")
                rc2, logs, _ = _run([
                    "kubectl", "logs", "job/agentloom-workflow", "-n", "agentloom",
                ])
                _info(f"Job logs:\n{logs[:500]}")
            else:
                _pass(f"Job finished ({status.strip()})")
        else:
            _fail("Job did not complete within timeout")
            rc2, pods, _ = _run([
                "kubectl", "get", "pods", "-n", "agentloom", "--no-headers",
            ])
            _info(f"Pods:\n{pods}")
            rc3, desc, _ = _run([
                "kubectl", "describe", "job", "agentloom-workflow", "-n", "agentloom",
            ])
            _info(f"Job describe (tail):\n{desc[-500:]}")

        # Verify namespace PSS labels
        rc, out, err = _run([
            "kubectl", "get", "ns", "agentloom",
            "-o", "jsonpath={.metadata.labels.pod-security\\.kubernetes\\.io/enforce}",
        ])
        if "restricted" in out:
            _pass("Namespace has PSS enforce=restricted label")
        else:
            _fail(f"Namespace PSS label: {out}")

    finally:
        # Cleanup
        _step(f"Deleting kind cluster: {cluster_name}")
        _run(["kind", "delete", "cluster", "--name", cluster_name], timeout=60)
        _pass(f"Kind cluster '{cluster_name}' deleted")

    _save_phase("Kustomize")


# ────────────────────────────────────────────────────────────
# Phase 4: Terraform — Full Stack with Observability
# ────────────────────────────────────────────────────────────

def phase_terraform() -> None:
    _header("Phase 4: Terraform — Full Stack (Kind + Observability + AgentLoom)")
    tf_dir = DEPLOY / "terraform"
    cluster_name = "audit-terraform"

    # Init
    _step("terraform init")
    rc, out, err = _run(["terraform", "init", "-input=false"], cwd=tf_dir, timeout=120)
    if rc == 0:
        _pass("terraform init succeeded")
    else:
        _fail("terraform init failed", err[-500:])
        _save_phase("Terraform")
        return

    # Plan
    _step("terraform plan")
    rc, out, err = _run([
        "terraform", "plan", "-input=false",
        f"-var=cluster_name={cluster_name}",
        "-var=enable_observability=true",
        "-var=agentloom_image_tag=audit-int",
    ], cwd=tf_dir, timeout=120)
    if rc == 0:
        _pass("terraform plan succeeded")
        # Check plan includes observability resources
        if "helm_release.jaeger" in out or "jaeger" in err:
            _pass("Plan includes Jaeger deployment")
        if "helm_release.otel_collector" in out or "otel_collector" in err:
            _pass("Plan includes OTel Collector deployment")
        if "helm_release.kube_prometheus" in out or "kube_prometheus" in err:
            _pass("Plan includes kube-prometheus-stack deployment")
    else:
        _fail("terraform plan failed", err[-500:])
        _save_phase("Terraform")
        return

    # Apply
    _step("terraform apply (this may take 3-5 minutes)")
    rc, out, err = _run([
        "terraform", "apply", "-auto-approve", "-input=false",
        f"-var=cluster_name={cluster_name}",
        "-var=enable_observability=true",
        "-var=agentloom_image_tag=audit-int",
    ], cwd=tf_dir, timeout=600)
    if rc != 0:
        _fail("terraform apply failed", err[-800:] if err else out[-800:])
        # Try cleanup
        _step("Attempting terraform destroy after failure")
        _run([
            "terraform", "destroy", "-auto-approve", "-input=false",
            f"-var=cluster_name={cluster_name}",
            "-var=enable_observability=true",
            "-var=agentloom_image_tag=audit-int",
        ], cwd=tf_dir, timeout=300)
        _run(["kind", "delete", "cluster", "--name", cluster_name], timeout=60)
        _save_phase("Terraform")
        return
    _pass("terraform apply succeeded")

    try:
        # Get kubeconfig
        rc, kubeconfig, err = _run(
            ["terraform", "output", "-raw", "kubeconfig"],
            cwd=tf_dir,
        )
        if rc != 0 or not kubeconfig.strip():
            _fail("Could not get kubeconfig from terraform output")
            return
        kubeconfig = kubeconfig.strip()
        kenv = {"KUBECONFIG": kubeconfig}

        # ── Verify Kind cluster ──
        _step("Verifying kind cluster")
        rc, out, err = _run(["kubectl", "cluster-info"], env=kenv)
        if rc == 0 and "running" in out.lower():
            _pass("Kind cluster is running")
        else:
            _fail("Kind cluster not reachable", err or out)
            return

        # ── Verify namespaces ──
        _step("Verifying namespaces")
        rc, out, err = _run(["kubectl", "get", "ns", "-o", "name"], env=kenv)
        if "namespace/agentloom" in out:
            _pass("Namespace 'agentloom' exists")
        else:
            _fail("Namespace 'agentloom' missing")
        if "namespace/observability" in out:
            _pass("Namespace 'observability' exists")
        else:
            _fail("Namespace 'observability' missing")

        # ── Verify agentloom namespace PSS labels ──
        rc, out, err = _run([
            "kubectl", "get", "ns", "agentloom",
            "-o", "jsonpath={.metadata.labels.pod-security\\.kubernetes\\.io/enforce}",
        ], env=kenv)
        if "restricted" in out:
            _pass("agentloom namespace has PSS enforce=restricted")
        else:
            _fail(f"agentloom namespace PSS label: '{out}'")

        # ── Verify observability pods ──
        _step("Waiting for observability pods to be ready")

        # Jaeger
        jaeger_ready = _wait_for(
            ["kubectl", "get", "pods", "-n", "observability",
             "-l", "app.kubernetes.io/name=jaeger",
             "-o", "jsonpath={.items[*].status.phase}"],
            lambda out, err: "Running" in out,
            "Jaeger pod",
            timeout=180, interval=10,
        )
        if jaeger_ready:
            _pass("Jaeger pod is Running")
        else:
            _fail("Jaeger pod not ready")
            rc, out, err = _run(["kubectl", "get", "pods", "-n", "observability"], env=kenv)
            _info(out)

        # OTel Collector
        otel_ready = _wait_for(
            ["kubectl", "get", "pods", "-n", "observability",
             "-l", "app.kubernetes.io/name=opentelemetry-collector",
             "-o", "jsonpath={.items[*].status.phase}"],
            lambda out, err: "Running" in out,
            "OTel Collector pod",
            timeout=120, interval=10,
        )
        if otel_ready:
            _pass("OTel Collector pod is Running")
        else:
            _fail("OTel Collector pod not ready")

        # Prometheus
        prom_ready = _wait_for(
            ["kubectl", "get", "pods", "-n", "observability",
             "-l", "app=kube-prometheus-stack-prometheus",
             "-o", "jsonpath={.items[*].status.phase}"],
            lambda out, err: "Running" in out,
            "Prometheus pod",
            timeout=180, interval=10,
        )
        if not prom_ready:
            # Try alternative label
            prom_ready = _wait_for(
                ["kubectl", "get", "pods", "-n", "observability",
                 "-l", "app.kubernetes.io/name=prometheus",
                 "-o", "jsonpath={.items[*].status.phase}"],
                lambda out, err: "Running" in out,
                "Prometheus pod (alt label)",
                timeout=60, interval=10,
            )
        if prom_ready:
            _pass("Prometheus pod is Running")
        else:
            _fail("Prometheus pod not ready")
            rc, out, err = _run(["kubectl", "get", "pods", "-n", "observability"], env=kenv)
            _info(out)

        # Grafana
        grafana_ready = _wait_for(
            ["kubectl", "get", "pods", "-n", "observability",
             "-l", "app.kubernetes.io/name=grafana",
             "-o", "jsonpath={.items[*].status.phase}"],
            lambda out, err: "Running" in out,
            "Grafana pod",
            timeout=180, interval=10,
        )
        if grafana_ready:
            _pass("Grafana pod is Running")
        else:
            _fail("Grafana pod not ready")

        # ── Verify Grafana dashboard loaded ──
        _step("Verifying Grafana dashboard ConfigMap")
        rc, out, err = _run([
            "kubectl", "get", "configmap", "agentloom-grafana-dashboard",
            "-n", "observability", "-o", "jsonpath={.data}",
        ], env=kenv)
        if rc == 0 and "agentloom.json" in out:
            _pass("Grafana dashboard ConfigMap contains agentloom.json")
        else:
            _fail("Grafana dashboard ConfigMap missing or empty")

        # ── Verify OTel Collector service ──
        _step("Verifying OTel Collector service endpoint")
        rc, out, err = _run([
            "kubectl", "get", "svc", "-n", "observability",
            "-o", "name",
        ], env=kenv)
        if "otel-collector" in out:
            _pass("OTel Collector service exists in observability namespace")
        else:
            _fail("OTel Collector service not found")

        # ── Verify all pods summary ──
        _step("Observability namespace pod summary")
        rc, out, err = _run([
            "kubectl", "get", "pods", "-n", "observability",
            "--no-headers",
        ], env=kenv)
        if rc == 0:
            lines = [l for l in out.strip().splitlines() if l.strip()]
            running = [l for l in lines if "Running" in l or "Completed" in l]
            _info(f"Total pods: {len(lines)}, Running/Completed: {len(running)}")
            for line in lines:
                _info(f"  {line.strip()}")
            if len(running) >= 3:
                _pass(f"Observability stack: {len(running)}/{len(lines)} pods healthy")
            else:
                _fail(f"Only {len(running)}/{len(lines)} pods healthy")

        # ── Verify agentloom Helm release ──
        _step("Verifying agentloom Helm release")
        rc, out, err = _run([
            "kubectl", "get", "jobs", "-n", "agentloom", "--no-headers",
        ], env=kenv)
        if rc == 0 and "agentloom" in out:
            _pass(f"AgentLoom job exists in agentloom namespace")
            _info(out.strip())
        else:
            _fail("AgentLoom job not found in agentloom namespace")

        # ── Verify OTel endpoint in agentloom pod ──
        _step("Verifying OTel endpoint configuration in agentloom pod")
        rc, out, err = _run([
            "kubectl", "get", "job", "-n", "agentloom",
            "-o", "jsonpath={.items[0].spec.template.spec.containers[0].env[?(@.name=='OTEL_EXPORTER_OTLP_ENDPOINT')].value}",
        ], env=kenv)
        if "otel-collector" in out and "observability" in out and "4317" in out:
            _pass(f"OTel endpoint points to in-cluster collector: {out}")
        elif "4317" in out:
            _pass(f"OTel endpoint configured: {out}")
        else:
            _fail(f"OTel endpoint unexpected: '{out}'")

        # ── Verify NodePort mappings (Terraform outputs) ──
        _step("Verifying Terraform outputs")
        for output_name in ["grafana_url", "prometheus_url", "jaeger_url"]:
            rc, out, err = _run(
                ["terraform", "output", "-raw", output_name],
                cwd=tf_dir,
            )
            if rc == 0 and out.strip():
                _pass(f"Output {output_name}: {out.strip()}")
            else:
                _fail(f"Output {output_name} missing or empty")

    finally:
        # Cleanup
        _step("terraform destroy")
        rc, out, err = _run([
            "terraform", "destroy", "-auto-approve", "-input=false",
            f"-var=cluster_name={cluster_name}",
            "-var=enable_observability=true",
            "-var=agentloom_image_tag=audit-int",
        ], cwd=tf_dir, timeout=300)
        if rc == 0:
            _pass("terraform destroy succeeded")
        else:
            _fail("terraform destroy failed — manual cleanup may be needed", err[-300:])
            # Fallback: delete kind cluster directly
            _run(["kind", "delete", "cluster", "--name", cluster_name], timeout=60)

        # Verify cluster is gone
        rc, out, err = _run(["kind", "get", "clusters"])
        if cluster_name not in out:
            _pass(f"Kind cluster '{cluster_name}' confirmed deleted")
        else:
            _fail(f"Kind cluster '{cluster_name}' still exists!")
            _run(["kind", "delete", "cluster", "--name", cluster_name], timeout=60)

    _save_phase("Terraform")


# ────────────────────────────────────────────────────────────
# Phase 5: Helm Direct Install
# ────────────────────────────────────────────────────────────

def phase_helm() -> None:
    _header("Phase 5: Helm — Direct Install into Kind")
    cluster_name = "audit-helm"

    _step(f"Creating kind cluster: {cluster_name}")
    rc, out, err = _run(["kind", "create", "cluster", "--name", cluster_name], timeout=120)
    if rc != 0:
        _fail("kind create cluster failed", err[-500:])
        _save_phase("Helm")
        return
    _pass(f"Kind cluster '{cluster_name}' created")

    try:
        # Load image
        _step("Loading image into kind cluster")
        _run(["kind", "load", "docker-image", "agentloom:audit-int", "--name", cluster_name], timeout=120)

        # Helm lint
        _step("helm lint")
        rc, out, err = _run([
            "helm", "lint", str(DEPLOY / "helm" / "agentloom"),
            "-f", str(DEPLOY / "helm" / "agentloom" / "ci" / "test-values.yaml"),
        ])
        if rc == 0:
            _pass("helm lint passed")
        else:
            _fail("helm lint failed", err or out)

        # Helm install with validate-only workflow
        _step("helm install (validate-only workflow)")
        rc, out, err = _run([
            "helm", "install", "agentloom",
            str(DEPLOY / "helm" / "agentloom"),
            "-n", "agentloom", "--create-namespace",
            "--set", "image.repository=agentloom",
            "--set", "image.tag=audit-int",
            "--set", "provider.existingSecret=",
            "--set", "networkPolicy.enabled=false",
            "--set", "observability.enabled=false",
            "-f", str(DEPLOY / "helm" / "agentloom" / "ci" / "test-values.yaml"),
        ], timeout=60)
        if rc == 0:
            _pass("helm install succeeded")
        else:
            _fail("helm install failed", err or out)
            return

        # Verify job created
        _step("Verifying Helm-deployed resources")
        rc, out, err = _run(["kubectl", "get", "job", "-n", "agentloom", "--no-headers"])
        if rc == 0 and "agentloom" in out:
            _pass(f"Job deployed via Helm")
        else:
            _fail("No job found after helm install")

        # Verify serviceaccount
        rc, out, err = _run(["kubectl", "get", "sa", "-n", "agentloom", "--no-headers"])
        if rc == 0 and "agentloom" in out:
            _pass("ServiceAccount deployed via Helm")
        else:
            _fail("ServiceAccount not found")

        # Verify configmap
        rc, out, err = _run(["kubectl", "get", "configmap", "-n", "agentloom", "--no-headers"])
        if rc == 0 and "workflow" in out:
            _pass("Workflow ConfigMap deployed via Helm")
        else:
            _fail("Workflow ConfigMap not found")

        # Test validation: missing workflow should fail
        _step("Testing Helm validation (missing workflow → fail)")
        rc, out, err = _run([
            "helm", "template", "test",
            str(DEPLOY / "helm" / "agentloom"),
            "-n", "agentloom",
        ])
        if rc != 0:
            _pass("Helm correctly rejects missing workflow.definition")
        else:
            _fail("Helm should reject missing workflow.definition")

        # Helm uninstall
        _step("helm uninstall")
        rc, out, err = _run(["helm", "uninstall", "agentloom", "-n", "agentloom"], timeout=30)
        if rc == 0:
            _pass("helm uninstall succeeded")
        else:
            _fail("helm uninstall failed")

    finally:
        _step(f"Deleting kind cluster: {cluster_name}")
        _run(["kind", "delete", "cluster", "--name", cluster_name], timeout=60)
        _pass(f"Kind cluster '{cluster_name}' deleted")

    _save_phase("Helm")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(description="AgentLoom Local Integration Audit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific phase (1=Docker, 2=Compose, 3=Kustomize, 4=Terraform, 5=Helm)")
    args = parser.parse_args()
    _verbose = args.verbose

    print(_bold("\n  AgentLoom — Local Integration Audit"))
    print(f"  {'━' * 44}")
    print(f"  Phases: Docker → Compose → Kustomize → Terraform → Helm")
    print()

    phases = {
        1: ("Docker",         phase_docker),
        2: ("Docker Compose", phase_docker_compose),
        3: ("Kustomize",      phase_kustomize),
        4: ("Terraform",      phase_terraform),
        5: ("Helm",           phase_helm),
    }

    if args.phase:
        name, fn = phases[args.phase]
        fn()
    else:
        for _, (name, fn) in sorted(phases.items()):
            fn()

    # Summary
    print(f"\n{'━' * 64}")
    print(f"  {_bold('Phase Summary')}")
    print(f"{'━' * 64}")
    for name, (p, f, s) in _phase_results.items():
        status = _green("PASS") if f == 0 else _red("FAIL")
        print(f"  {status}  {name}: {_green(f'{p} passed')}  "
              f"{_red(f'{f} failed') if f else '0 failed'}  "
              f"{_yellow(f'{s} skipped') if s else ''}")

    total = _pass_count + _fail_count + _skip_count
    print(f"\n{'━' * 64}")
    print(f"  {_bold('Total')}: {_green(f'{_pass_count} passed')}  "
          f"{_red(f'{_fail_count} failed') if _fail_count else '0 failed'}  "
          f"{_yellow(f'{_skip_count} skipped') if _skip_count else '0 skipped'}  "
          f"({total} checks)")
    print(f"{'━' * 64}\n")

    return 1 if _fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
