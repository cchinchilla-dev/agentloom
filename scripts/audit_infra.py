#!/usr/bin/env python3
"""
Infrastructure Audit for AgentLoom.

Validates the entire deploy/ stack without requiring a running cluster.
Two phases:
  1. Static analysis  — pure Python, reads YAML/HCL/Dockerfile, checks policies.
  2. Tool validation   — calls kustomize, helm, terraform (skipped if not installed).

Run:
    python scripts/audit_infra.py          # from repo root
    python scripts/audit_infra.py -v       # verbose (show check names on pass)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import yaml  # PyYAML — available in dev environment
except ImportError:
    yaml = None  # type: ignore[assignment]

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

_pass_count = 0
_fail_count = 0
_skip_count = 0
_verbose = False


def _green(t: str) -> str:
    return f"\033[92m{t}\033[0m"


def _red(t: str) -> str:
    return f"\033[91m{t}\033[0m"


def _yellow(t: str) -> str:
    return f"\033[93m{t}\033[0m"


def _bold(t: str) -> str:
    return f"\033[1m{t}\033[0m"


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {_bold(title)}")
    print(f"{'─' * 60}")


def _pass(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    if _verbose:
        print(f"  {_green('✓')} {msg}")


def _fail(msg: str, detail: str = "") -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {_red('✗')} {msg}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"      {line}")


def _skip(msg: str) -> None:
    global _skip_count
    _skip_count += 1
    print(f"  {_yellow('○')} {msg}")


def _load_yaml(path: Path) -> dict | list | None:
    """Load a YAML file. Returns None on parse error."""
    if yaml is None:
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _load_yaml_all(path: Path) -> list:
    """Load a multi-document YAML file."""
    if yaml is None:
        return []
    with open(path) as f:
        return list(yaml.safe_load_all(f))


def _file_exists(path: Path) -> bool:
    return path.is_file()


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


# ────────────────────────────────────────────────────────────
# 1. Dockerfile
# ────────────────────────────────────────────────────────────

def audit_dockerfile() -> None:
    _header("Dockerfile")
    path = ROOT / "Dockerfile"
    if not _file_exists(path):
        _fail("Dockerfile not found")
        return
    content = path.read_text()

    # Multi-stage
    stages = re.findall(r"^FROM\s+.+\s+AS\s+(\w+)", content, re.MULTILINE)
    if len(stages) >= 3:
        _pass(f"Multi-stage build: {len(stages)} stages ({', '.join(stages)})")
    else:
        _fail(f"Expected ≥3 stages, found {len(stages)}: {stages}")

    # Non-root user
    if re.search(r"^USER\s+agentloom", content, re.MULTILINE):
        _pass("Non-root USER directive (agentloom)")
    else:
        _fail("Missing USER agentloom directive in production stage")

    # UID 1000
    if "useradd --uid 1000" in content or "uid=1000" in content:
        _pass("UID 1000 for agentloom user")
    else:
        _fail("agentloom user does not have UID 1000")

    # ENTRYPOINT
    if re.search(r'ENTRYPOINT\s+\["agentloom"\]', content):
        _pass("ENTRYPOINT is agentloom CLI")
    else:
        _fail("Production ENTRYPOINT should be agentloom CLI")

    # BUILD_OBSERVABILITY build arg
    if "BUILD_OBSERVABILITY" in content:
        _pass("BUILD_OBSERVABILITY build arg present")
    else:
        _fail("Missing BUILD_OBSERVABILITY build arg")

    # No secrets baked in
    for pattern in ["API_KEY", "SECRET", "PASSWORD"]:
        if re.search(rf"^ENV\s+.*{pattern}", content, re.MULTILINE):
            _fail(f"Potential secret baked into image: ENV containing {pattern}")
            break
    else:
        _pass("No secrets baked into image (ENV)")

    # OTEL endpoint env var
    if "OTEL_EXPORTER_OTLP_ENDPOINT" in content:
        _pass("OTEL_EXPORTER_OTLP_ENDPOINT env var set")
    else:
        _fail("Missing OTEL_EXPORTER_OTLP_ENDPOINT default")

    # python:3.12-slim base (not alpine)
    if "python:3.12-slim" in content:
        _pass("Base image is python:3.12-slim (not alpine)")
    elif "alpine" in content.lower():
        _fail("Alpine base — pydantic/httpx compilation issues")
    else:
        _pass("Base image is not alpine")


# ────────────────────────────────────────────────────────────
# 2. Docker Compose
# ────────────────────────────────────────────────────────────

def audit_docker_compose() -> None:
    _header("Docker Compose")
    path = DEPLOY / "docker-compose.yml"
    if not _file_exists(path):
        _fail("docker-compose.yml not found")
        return
    if yaml is None:
        _skip("PyYAML not installed — skipping Docker Compose audit")
        return
    data = _load_yaml(path)
    if not data:
        _fail("Failed to parse docker-compose.yml")
        return

    services = data.get("services", {})
    expected = {"agentloom", "otel-collector", "jaeger", "prometheus", "grafana"}
    found = set(services.keys())
    missing = expected - found
    if not missing:
        _pass(f"All expected services present ({', '.join(sorted(found & expected))})")
    else:
        _fail(f"Missing services: {missing}")

    # agentloom profile=run (doesn't auto-start)
    al = services.get("agentloom", {})
    profiles = al.get("profiles", [])
    if "run" in profiles:
        _pass("agentloom service uses profiles: [run] (on-demand)")
    else:
        _fail("agentloom service should use profiles: [run] to avoid auto-start")

    # Shared network
    networks = data.get("networks", {})
    if "agentloom" in networks:
        _pass("Shared 'agentloom' network defined")
    else:
        _fail("Missing shared 'agentloom' network")

    # Health checks on infra services
    for svc_name in ["jaeger", "prometheus", "grafana"]:
        svc = services.get(svc_name, {})
        if "healthcheck" in svc:
            _pass(f"{svc_name} has healthcheck")
        else:
            _fail(f"{svc_name} missing healthcheck")

    # OTel endpoint pointing to collector
    env = al.get("environment", [])
    otel_env = [e for e in env if "OTEL_EXPORTER_OTLP_ENDPOINT" in str(e)]
    if otel_env:
        _pass("agentloom points to OTel Collector endpoint")
    else:
        _fail("agentloom missing OTEL_EXPORTER_OTLP_ENDPOINT")


# ────────────────────────────────────────────────────────────
# 3. K8s Base Manifests
# ────────────────────────────────────────────────────────────

def _check_pod_security(spec: dict, context: str) -> None:
    """Validate security context on a pod spec."""
    sc = spec.get("securityContext", {})

    if sc.get("runAsNonRoot") is True:
        _pass(f"{context}: runAsNonRoot=true")
    else:
        _fail(f"{context}: runAsNonRoot not set")

    if sc.get("runAsUser") == 1000:
        _pass(f"{context}: runAsUser=1000")
    else:
        _fail(f"{context}: runAsUser should be 1000")

    seccomp = sc.get("seccompProfile", {})
    if seccomp.get("type") == "RuntimeDefault":
        _pass(f"{context}: seccomp RuntimeDefault")
    else:
        _fail(f"{context}: seccomp profile should be RuntimeDefault")

    if spec.get("automountServiceAccountToken") is False:
        _pass(f"{context}: automountServiceAccountToken=false")
    else:
        _fail(f"{context}: automountServiceAccountToken should be false")

    # Container-level security
    containers = spec.get("containers", [])
    for c in containers:
        csc = c.get("securityContext", {})
        name = c.get("name", "?")
        if csc.get("allowPrivilegeEscalation") is False:
            _pass(f"{context}/{name}: allowPrivilegeEscalation=false")
        else:
            _fail(f"{context}/{name}: allowPrivilegeEscalation should be false")

        if csc.get("readOnlyRootFilesystem") is True:
            _pass(f"{context}/{name}: readOnlyRootFilesystem=true")
        else:
            _fail(f"{context}/{name}: readOnlyRootFilesystem should be true")

        caps = csc.get("capabilities", {})
        if "ALL" in caps.get("drop", []):
            _pass(f"{context}/{name}: capabilities drop ALL")
        else:
            _fail(f"{context}/{name}: should drop ALL capabilities")

    # No CPU limits (intentional — avoids CFS throttling)
    for c in containers:
        limits = c.get("resources", {}).get("limits", {})
        if "cpu" not in limits:
            _pass(f"{context}/{c.get('name')}: no CPU limit (CFS throttling prevention)")
        else:
            _fail(f"{context}/{c.get('name')}: CPU limit set — causes CFS throttling on I/O-bound LLM calls")

    # /tmp emptyDir volume
    volumes = spec.get("volumes", [])
    tmp_vols = [v for v in volumes if v.get("name") == "tmp" and "emptyDir" in v]
    if tmp_vols:
        _pass(f"{context}: /tmp emptyDir volume for read-only rootfs")
    else:
        _fail(f"{context}: missing /tmp emptyDir (required with readOnlyRootFilesystem)")


def audit_k8s_base() -> None:
    _header("Kubernetes Base Manifests")
    base = DEPLOY / "k8s" / "base"
    if not base.is_dir():
        _fail("deploy/k8s/base/ directory not found")
        return
    if yaml is None:
        _skip("PyYAML not installed — skipping K8s audit")
        return

    # Namespace PSS labels
    ns = _load_yaml(base / "namespace.yaml")
    if ns:
        labels = ns.get("metadata", {}).get("labels", {})
        for level in ["enforce", "audit", "warn"]:
            key = f"pod-security.kubernetes.io/{level}"
            if labels.get(key) == "restricted":
                _pass(f"Namespace PSS {level}=restricted")
            else:
                _fail(f"Namespace missing PSS label {key}=restricted")

    # ServiceAccount
    sa = _load_yaml(base / "serviceaccount.yaml")
    if sa:
        if sa.get("automountServiceAccountToken") is False:
            _pass("ServiceAccount automountServiceAccountToken=false")
        else:
            _fail("ServiceAccount should have automountServiceAccountToken=false")

    # Job
    job = _load_yaml(base / "job.yaml")
    if job:
        spec = job.get("spec", {})
        tmpl = spec.get("template", {}).get("spec", {})
        _check_pod_security(tmpl, "Job")

        if tmpl.get("restartPolicy") == "Never":
            _pass("Job restartPolicy=Never")
        else:
            _fail("Job restartPolicy should be Never")

        if spec.get("backoffLimit", 0) > 0:
            _pass(f"Job backoffLimit={spec['backoffLimit']}")
        else:
            _fail("Job backoffLimit should be > 0")

        if spec.get("ttlSecondsAfterFinished", 0) > 0:
            _pass(f"Job ttlSecondsAfterFinished={spec['ttlSecondsAfterFinished']}")
        else:
            _fail("Job ttlSecondsAfterFinished should be set for cleanup")

    # CronJob
    cj = _load_yaml(base / "cronjob.yaml")
    if cj:
        cj_spec = cj.get("spec", {})
        if cj_spec.get("concurrencyPolicy") == "Forbid":
            _pass("CronJob concurrencyPolicy=Forbid")
        else:
            _fail("CronJob concurrencyPolicy should be Forbid")

        jt = cj_spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec", {})
        _check_pod_security(jt, "CronJob")

    # Secret — should have empty data
    secret = _load_yaml(base / "secret.yaml")
    if secret:
        data = secret.get("data", None)
        if data == {} or data is None:
            _pass("Secret has empty data (populated at deploy time)")
        else:
            _fail("Secret should ship with empty data — never bake API keys")

    # NetworkPolicy
    np = _load_yaml(base / "networkpolicy.yaml")
    if np:
        spec = np.get("spec", {})
        ptypes = spec.get("policyTypes", [])
        if "Ingress" in ptypes and "Egress" in ptypes:
            _pass("NetworkPolicy covers both Ingress and Egress")
        else:
            _fail(f"NetworkPolicy policyTypes should include Ingress+Egress, got {ptypes}")

        if spec.get("ingress") == []:
            _pass("NetworkPolicy denies all ingress (empty list)")
        else:
            _fail("NetworkPolicy should deny all ingress for batch workloads")

        egress = spec.get("egress", [])
        egress_ports = set()
        for rule in egress:
            for p in rule.get("ports", []):
                egress_ports.add(p.get("port"))
        required_ports = {53, 443, 4317}
        if required_ports.issubset(egress_ports):
            _pass(f"NetworkPolicy allows required egress ports: DNS(53), HTTPS(443), OTel(4317)")
        else:
            _fail(f"NetworkPolicy missing ports: {required_ports - egress_ports}")

    # Kustomization references
    kust = _load_yaml(base / "kustomization.yaml")
    if kust:
        resources = kust.get("resources", [])
        for r in resources:
            if _file_exists(base / r):
                _pass(f"kustomization.yaml → {r} exists")
            else:
                _fail(f"kustomization.yaml references {r} but file not found")


# ────────────────────────────────────────────────────────────
# 4. Kustomize Overlays
# ────────────────────────────────────────────────────────────

def audit_kustomize_overlays() -> None:
    _header("Kustomize Overlays")
    overlays_dir = DEPLOY / "k8s" / "overlays"
    if not overlays_dir.is_dir():
        _fail("deploy/k8s/overlays/ not found")
        return
    if yaml is None:
        _skip("PyYAML not installed")
        return

    expected_overlays = ["dev", "staging", "production"]
    for name in expected_overlays:
        overlay = overlays_dir / name
        if not overlay.is_dir():
            _fail(f"Overlay '{name}' directory missing")
            continue
        _pass(f"Overlay '{name}' exists")

        kust = _load_yaml(overlay / "kustomization.yaml")
        if not kust:
            _fail(f"{name}/kustomization.yaml not parseable")
            continue

        # Must reference base
        resources = kust.get("resources", [])
        if "../../base" in resources:
            _pass(f"{name}: references ../../base")
        else:
            _fail(f"{name}: should reference ../../base")

        # Patch files exist
        patches = kust.get("patches", [])
        for p in patches:
            if isinstance(p, dict) and "path" in p:
                pf = overlay / p["path"]
                if pf.is_file():
                    _pass(f"{name}: patch file {p['path']} exists")
                else:
                    _fail(f"{name}: patch file {p['path']} not found")

    # Dev should disable NetworkPolicy
    dev_kust = _load_yaml(overlays_dir / "dev" / "kustomization.yaml")
    if dev_kust:
        patches = dev_kust.get("patches", [])
        np_deleted = any(
            isinstance(p, dict) and "$patch: delete" in str(p.get("patch", ""))
            and "NetworkPolicy" in str(p.get("target", ""))
            for p in patches
        )
        if np_deleted:
            _pass("dev: NetworkPolicy deleted (fast iteration)")
        else:
            _fail("dev: should delete NetworkPolicy for fast iteration")

    # Production should have pinned image (not latest)
    prod_kust = _load_yaml(overlays_dir / "production" / "kustomization.yaml")
    if prod_kust:
        images = prod_kust.get("images", [])
        for img in images:
            tag = img.get("newTag", "")
            if tag and tag != "latest":
                _pass(f"production: pinned image tag ({tag})")
            elif tag == "latest":
                _fail("production: should not use 'latest' tag")


# ────────────────────────────────────────────────────────────
# 5. Helm Chart
# ────────────────────────────────────────────────────────────

def audit_helm_chart() -> None:
    _header("Helm Chart")
    chart_dir = DEPLOY / "helm" / "agentloom"
    if not chart_dir.is_dir():
        _fail("deploy/helm/agentloom/ not found")
        return
    if yaml is None:
        _skip("PyYAML not installed")
        return

    # Chart.yaml
    chart = _load_yaml(chart_dir / "Chart.yaml")
    if chart:
        if chart.get("apiVersion") == "v2":
            _pass("Chart apiVersion: v2")
        else:
            _fail("Chart should use apiVersion: v2")

        if chart.get("type") == "application":
            _pass("Chart type: application")
        else:
            _fail("Chart type should be 'application'")

        if chart.get("appVersion"):
            _pass(f"appVersion set: {chart['appVersion']}")
        else:
            _fail("appVersion not set")

    # values.yaml — security defaults
    values = _load_yaml(chart_dir / "values.yaml")
    if values:
        sc = values.get("securityContext", {})
        if sc.get("runAsNonRoot") is True:
            _pass("values.yaml: securityContext.runAsNonRoot=true")
        else:
            _fail("values.yaml: securityContext.runAsNonRoot should default to true")

        if sc.get("runAsUser") == 1000:
            _pass("values.yaml: securityContext.runAsUser=1000")
        else:
            _fail("values.yaml: securityContext.runAsUser should default to 1000")

        csc = values.get("containerSecurityContext", {})
        if csc.get("readOnlyRootFilesystem") is True:
            _pass("values.yaml: readOnlyRootFilesystem=true")
        else:
            _fail("values.yaml: readOnlyRootFilesystem should default to true")

        if csc.get("allowPrivilegeEscalation") is False:
            _pass("values.yaml: allowPrivilegeEscalation=false")
        else:
            _fail("values.yaml: allowPrivilegeEscalation should default to false")

        # No CPU limit in defaults
        limits = values.get("resources", {}).get("limits", {})
        if "cpu" not in limits:
            _pass("values.yaml: no default CPU limit")
        else:
            _fail("values.yaml: should not set default CPU limit (CFS throttling)")

        # automountServiceAccountToken
        sa = values.get("serviceAccount", {})
        if sa.get("automountServiceAccountToken") is False:
            _pass("values.yaml: serviceAccount.automountServiceAccountToken=false")
        else:
            _fail("values.yaml: serviceAccount.automountServiceAccountToken should default to false")

        # NetworkPolicy enabled by default
        np = values.get("networkPolicy", {})
        if np.get("enabled") is True:
            _pass("values.yaml: networkPolicy.enabled=true by default")
        else:
            _fail("values.yaml: networkPolicy should be enabled by default")

        # Observability enabled by default
        obs = values.get("observability", {})
        if obs.get("enabled") is True:
            _pass("values.yaml: observability.enabled=true by default")
        else:
            _fail("values.yaml: observability should be enabled by default")

        # Provider keys — existingSecret preferred
        prov = values.get("provider", {})
        if prov.get("existingSecret") == "":
            _pass("values.yaml: provider.existingSecret empty (user must set)")
        else:
            _fail("values.yaml: provider.existingSecret should default to empty")

    # Required templates exist
    templates_dir = chart_dir / "templates"
    required_templates = [
        "_helpers.tpl", "validate.yaml", "job.yaml", "cronjob.yaml",
        "configmap.yaml", "secret.yaml", "serviceaccount.yaml",
        "namespace.yaml", "networkpolicy.yaml", "resourcequota.yaml", "NOTES.txt",
    ]
    for t in required_templates:
        if _file_exists(templates_dir / t):
            _pass(f"Template {t} exists")
        else:
            _fail(f"Template {t} missing")

    # Validate template uses fail function
    validate = (templates_dir / "validate.yaml").read_text() if _file_exists(templates_dir / "validate.yaml") else ""
    if "fail" in validate:
        _pass("validate.yaml uses fail function for input validation")
    else:
        _fail("validate.yaml should use fail for render-time validation")

    # _helpers.tpl has podSpec helper
    helpers = (templates_dir / "_helpers.tpl").read_text() if _file_exists(templates_dir / "_helpers.tpl") else ""
    if "agentloom.podSpec" in helpers:
        _pass("_helpers.tpl defines shared podSpec (DRY between Job and CronJob)")
    else:
        _fail("_helpers.tpl should define agentloom.podSpec helper")

    # Job template uses podSpec
    job_tpl = (templates_dir / "job.yaml").read_text() if _file_exists(templates_dir / "job.yaml") else ""
    cj_tpl = (templates_dir / "cronjob.yaml").read_text() if _file_exists(templates_dir / "cronjob.yaml") else ""
    if "agentloom.podSpec" in job_tpl:
        _pass("job.yaml uses shared podSpec helper")
    else:
        _fail("job.yaml should include agentloom.podSpec")
    if "agentloom.podSpec" in cj_tpl:
        _pass("cronjob.yaml uses shared podSpec helper")
    else:
        _fail("cronjob.yaml should include agentloom.podSpec")

    # CI test values exist
    if _file_exists(chart_dir / "ci" / "test-values.yaml"):
        _pass("ci/test-values.yaml exists for CI lint/template")
    else:
        _fail("ci/test-values.yaml missing — Helm CI validation will fail")


# ────────────────────────────────────────────────────────────
# 6. Terraform
# ────────────────────────────────────────────────────────────

def audit_terraform() -> None:
    _header("Terraform")
    tf_dir = DEPLOY / "terraform"
    if not tf_dir.is_dir():
        _fail("deploy/terraform/ not found")
        return

    required_files = ["versions.tf", "variables.tf", "main.tf", "outputs.tf",
                      "terraform.tfvars.example"]
    for f in required_files:
        if _file_exists(tf_dir / f):
            _pass(f"{f} exists")
        else:
            _fail(f"{f} missing")

    # .gitignore for terraform
    gitignore = tf_dir / ".gitignore"
    if _file_exists(gitignore):
        gi_content = gitignore.read_text()
        for pattern in [".terraform/", "*.tfstate", "*.tfvars"]:
            if pattern in gi_content:
                _pass(f".gitignore includes {pattern}")
            else:
                _fail(f".gitignore should include {pattern}")
    else:
        _fail("deploy/terraform/.gitignore not found")

    # Analyze main.tf for observability stack
    main_tf = (tf_dir / "main.tf").read_text() if _file_exists(tf_dir / "main.tf") else ""

    # Conditional observability resources
    if 'var.enable_observability' in main_tf:
        _pass("Observability is conditional on enable_observability variable")
    else:
        _fail("main.tf should use var.enable_observability for conditional deployment")

    obs_components = {
        "Jaeger": "helm_release" in main_tf and "jaeger" in main_tf,
        "OTel Collector": "otel_collector" in main_tf or "otel-collector" in main_tf,
        "kube-prometheus-stack": "kube_prometheus" in main_tf or "kube-prometheus-stack" in main_tf,
        "Grafana dashboard ConfigMap": "grafana_dashboard" in main_tf or "agentloom-grafana-dashboard" in main_tf,
    }
    for component, present in obs_components.items():
        if present:
            _pass(f"Observability: {component} defined in main.tf")
        else:
            _fail(f"Observability: {component} missing from main.tf")

    # Kind cluster
    if "kind_cluster" in main_tf:
        _pass("Kind cluster resource defined")
    else:
        _fail("Kind cluster resource missing")

    # Port mappings for local access
    for port_name, port in [("Grafana", "3000"), ("Prometheus", "9090"), ("Jaeger", "16686")]:
        if port in main_tf:
            _pass(f"Port mapping for {port_name} (:{port})")
        else:
            _fail(f"Port mapping for {port_name} (:{port}) missing")

    # PSS namespace labels
    if "pod-security.kubernetes.io/enforce" in main_tf:
        _pass("Namespace has PSS enforce label")
    else:
        _fail("Namespace should have PSS enforce=restricted label")

    # Provider keys as sensitive variable
    vars_tf = (tf_dir / "variables.tf").read_text() if _file_exists(tf_dir / "variables.tf") else ""
    if "sensitive" in vars_tf and "provider_api_keys" in vars_tf:
        _pass("provider_api_keys variable is sensitive")
    else:
        _fail("provider_api_keys should be marked sensitive")

    # Variable validation blocks
    if "validation" in vars_tf:
        _pass("Variables have validation blocks")
    else:
        _fail("Variables should have validation blocks")

    # Outputs
    outputs_tf = (tf_dir / "outputs.tf").read_text() if _file_exists(tf_dir / "outputs.tf") else ""
    for output in ["grafana_url", "prometheus_url", "jaeger_url", "kubeconfig"]:
        if output in outputs_tf:
            _pass(f"Output: {output}")
        else:
            _fail(f"Output missing: {output}")

    # No residual files
    residual = list(tf_dir.glob("*.tfstate*")) + list(tf_dir.glob("*-config"))
    if not residual:
        _pass("No residual state/config files in terraform/")
    else:
        _fail(f"Residual files in terraform/: {[f.name for f in residual]}")


# ────────────────────────────────────────────────────────────
# 7. ArgoCD
# ────────────────────────────────────────────────────────────

def audit_argocd() -> None:
    _header("ArgoCD")
    path = DEPLOY / "argocd" / "application.yaml"
    if not _file_exists(path):
        _fail("deploy/argocd/application.yaml not found")
        return
    if yaml is None:
        _skip("PyYAML not installed")
        return

    app = _load_yaml(path)
    if not app:
        _fail("Failed to parse application.yaml")
        return

    spec = app.get("spec", {})

    # Sync policy
    sync = spec.get("syncPolicy", {})
    auto = sync.get("automated", {})
    if auto.get("prune") is True and auto.get("selfHeal") is True:
        _pass("Automated sync with prune + selfHeal")
    else:
        _fail("syncPolicy should have automated prune and selfHeal")

    # Sync options
    opts = sync.get("syncOptions", [])
    for expected in ["CreateNamespace=true", "Replace=true"]:
        if expected in opts:
            _pass(f"syncOption: {expected}")
        else:
            _fail(f"syncOption missing: {expected}")

    # ignoreDifferences for Job immutability
    diffs = spec.get("ignoreDifferences", [])
    job_ignore = any(d.get("kind") == "Job" for d in diffs)
    if job_ignore:
        _pass("ignoreDifferences configured for Job (immutability)")
    else:
        _fail("Should ignore Job selector/label differences (K8s immutability)")

    # Retry policy
    retry = sync.get("retry", {})
    if retry.get("limit", 0) > 0:
        _pass(f"Retry policy: limit={retry['limit']}")
    else:
        _fail("Retry policy should have limit > 0")

    # Source points to Helm chart
    source = spec.get("source", {})
    if "deploy/helm/agentloom" in source.get("path", ""):
        _pass("Source points to deploy/helm/agentloom")
    else:
        _fail("Source path should be deploy/helm/agentloom")


# ────────────────────────────────────────────────────────────
# 8. CI/CD Workflows
# ────────────────────────────────────────────────────────────

def audit_ci_workflows() -> None:
    _header("CI/CD Workflows")
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        _fail(".github/workflows/ not found")
        return
    if yaml is None:
        _skip("PyYAML not installed")
        return

    # Required workflows
    expected = {"ci.yml", "docker.yml", "release.yml"}
    found = {f.name for f in workflows.glob("*.yml")}
    for w in expected:
        if w in found:
            _pass(f"Workflow {w} exists")
        else:
            _fail(f"Workflow {w} missing")

    # SHA pinning — check all uses: lines
    for wf_file in workflows.glob("*.yml"):
        content = wf_file.read_text()
        uses_lines = re.findall(r"uses:\s+(.+)", content)
        unpinned = []
        for use in uses_lines:
            use = use.strip()
            # Skip local actions (./), docker://, and comments
            if use.startswith("./") or use.startswith("docker://"):
                continue
            # Should have @<sha> not @v3 etc.
            match = re.match(r"[\w\-]+/[\w\-]+@(.+?)(?:\s|$)", use)
            if match:
                ref = match.group(1).strip()
                # SHA is 40 hex chars; tags are short
                if not re.match(r"^[0-9a-f]{40}$", ref):
                    unpinned.append(f"{use}")

        if not unpinned:
            _pass(f"{wf_file.name}: all actions pinned to SHAs")
        else:
            _fail(f"{wf_file.name}: unpinned actions", "\n".join(unpinned))

    # CI workflow should validate manifests
    ci_content = (workflows / "ci.yml").read_text() if _file_exists(workflows / "ci.yml") else ""
    if "kubeconform" in ci_content or "kustomize" in ci_content:
        _pass("ci.yml validates K8s manifests")
    else:
        _fail("ci.yml should validate K8s manifests")

    if "helm lint" in ci_content or "helm" in ci_content:
        _pass("ci.yml validates Helm chart")
    else:
        _fail("ci.yml should validate Helm chart")

    # Docker workflow
    docker_content = (workflows / "docker.yml").read_text() if _file_exists(workflows / "docker.yml") else ""
    if "ghcr.io" in docker_content:
        _pass("docker.yml pushes to GHCR")
    else:
        _fail("docker.yml should push to GHCR")

    if "smoke test" in docker_content.lower() or "validate" in docker_content:
        _pass("docker.yml includes smoke test")
    else:
        _fail("docker.yml should include smoke test")


# ────────────────────────────────────────────────────────────
# 9. Documentation Consistency
# ────────────────────────────────────────────────────────────

def audit_docs() -> None:
    _header("Documentation Consistency")
    readme = (ROOT / "README.md").read_text() if _file_exists(ROOT / "README.md") else ""
    infra = (DEPLOY / "INFRASTRUCTURE.md").read_text() if _file_exists(DEPLOY / "INFRASTRUCTURE.md") else ""

    if not readme:
        _fail("README.md not found")
        return

    # README references key deployment methods
    for method in ["Docker", "Kustomize", "Helm", "Terraform", "ArgoCD"]:
        if method in readme:
            _pass(f"README mentions {method}")
        else:
            _fail(f"README should mention {method}")

    # README links to INFRASTRUCTURE.md
    if "INFRASTRUCTURE.md" in readme:
        _pass("README links to INFRASTRUCTURE.md")
    else:
        _fail("README should link to INFRASTRUCTURE.md")

    # INFRASTRUCTURE.md exists and is substantial
    if len(infra) > 2000:
        _pass(f"INFRASTRUCTURE.md is substantial ({len(infra.splitlines())} lines)")
    else:
        _fail("INFRASTRUCTURE.md should be a comprehensive guide")

    # INFRASTRUCTURE.md documents observability stack
    if "enable_observability" in infra:
        _pass("INFRASTRUCTURE.md documents enable_observability toggle")
    else:
        _fail("INFRASTRUCTURE.md should document enable_observability")

    for component in ["OTel Collector", "Jaeger", "Prometheus", "Grafana"]:
        if component in infra:
            _pass(f"INFRASTRUCTURE.md documents {component}")
        else:
            _fail(f"INFRASTRUCTURE.md should document {component}")

    # INFRASTRUCTURE.md documents security
    for concept in ["runAsNonRoot", "readOnlyRootFilesystem", "seccompProfile",
                     "NetworkPolicy", "automountServiceAccountToken"]:
        if concept in infra:
            _pass(f"INFRASTRUCTURE.md documents {concept}")
        else:
            _fail(f"INFRASTRUCTURE.md should document {concept}")


# ────────────────────────────────────────────────────────────
# 10. Cross-Reference Integrity
# ────────────────────────────────────────────────────────────

def audit_cross_references() -> None:
    _header("Cross-Reference Integrity")

    # Helm chart image matches K8s base
    if yaml is None:
        _skip("PyYAML not installed")
        return

    values = _load_yaml(DEPLOY / "helm" / "agentloom" / "values.yaml")
    job_yaml = _load_yaml(DEPLOY / "k8s" / "base" / "job.yaml")

    if values and job_yaml:
        helm_repo = values.get("image", {}).get("repository", "")
        k8s_image = job_yaml.get("spec", {}).get("template", {}).get("spec", {}).get(
            "containers", [{}])[0].get("image", "")
        if helm_repo and helm_repo in k8s_image:
            _pass(f"Image repository consistent: {helm_repo}")
        else:
            _fail(f"Image mismatch — Helm: {helm_repo}, K8s: {k8s_image}")

    # Helm OTel endpoint matches Docker Compose
    compose = _load_yaml(DEPLOY / "docker-compose.yml")
    if values and compose:
        helm_otel = values.get("observability", {}).get("otelEndpoint", "")
        if "4317" in helm_otel:
            _pass(f"Helm OTel endpoint uses port 4317")
        else:
            _fail(f"Helm OTel endpoint should use port 4317")

    # Terraform references Helm chart
    main_tf = (DEPLOY / "terraform" / "main.tf").read_text() if _file_exists(DEPLOY / "terraform" / "main.tf") else ""
    if "helm/agentloom" in main_tf:
        _pass("Terraform references local Helm chart")
    else:
        _fail("Terraform should reference deploy/helm/agentloom chart")

    # Terraform references Grafana dashboard JSON
    if "agentloom.json" in main_tf or "grafana/dashboards" in main_tf:
        _pass("Terraform references Grafana dashboard JSON")
    else:
        _fail("Terraform should load deploy/grafana/dashboards/agentloom.json")

    # Dashboard JSON exists
    dashboard = DEPLOY / "grafana" / "dashboards" / "agentloom.json"
    if _file_exists(dashboard):
        _pass("Grafana dashboard JSON exists")
        try:
            json.loads(dashboard.read_text())
            _pass("Grafana dashboard JSON is valid")
        except json.JSONDecodeError as e:
            _fail(f"Grafana dashboard JSON is invalid: {e}")
    else:
        _fail("deploy/grafana/dashboards/agentloom.json not found")

    # ArgoCD points to correct chart path
    argo = _load_yaml(DEPLOY / "argocd" / "application.yaml")
    if argo:
        path = argo.get("spec", {}).get("source", {}).get("path", "")
        if path == "deploy/helm/agentloom":
            _pass("ArgoCD source path matches Helm chart location")
        else:
            _fail(f"ArgoCD source path '{path}' should be 'deploy/helm/agentloom'")


# ────────────────────────────────────────────────────────────
# 11. Tool-Based Validation (requires kustomize, helm, terraform)
# ────────────────────────────────────────────────────────────

def audit_kustomize_build() -> None:
    _header("Kustomize Build (tool)")
    if not _has_tool("kustomize"):
        _skip("kustomize not installed — skipping build validation")
        return

    for overlay in ["dev", "staging", "production"]:
        path = DEPLOY / "k8s" / "overlays" / overlay
        rc, out, err = _run(["kustomize", "build", str(path)])
        if rc == 0:
            _pass(f"kustomize build {overlay} — OK")
            # Optionally validate with kubeconform
            if _has_tool("kubeconform"):
                rc2, out2, err2 = _run(
                    ["kubeconform", "-strict", "-summary"],
                    cwd=path,
                )
                # kubeconform reads from stdin, so pipe
                r = subprocess.run(
                    ["kubeconform", "-strict", "-summary"],
                    input=out, capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    _pass(f"kubeconform {overlay} — valid")
                else:
                    _fail(f"kubeconform {overlay} — errors", r.stderr or r.stdout)
        else:
            _fail(f"kustomize build {overlay} — failed", err)


def audit_helm_render() -> None:
    _header("Helm Validation (tool)")
    if not _has_tool("helm"):
        _skip("helm not installed — skipping Helm validation")
        return

    chart = DEPLOY / "helm" / "agentloom"
    ci_values = chart / "ci" / "test-values.yaml"

    # Lint
    rc, out, err = _run(["helm", "lint", str(chart), "-f", str(ci_values)])
    if rc == 0:
        _pass("helm lint — OK")
    else:
        _fail("helm lint — failed", err or out)

    # Template render
    rc, out, err = _run([
        "helm", "template", "test", str(chart),
        "-f", str(ci_values), "-n", "agentloom",
    ])
    if rc == 0:
        _pass("helm template — renders successfully")

        # Validate rendered output with kubeconform
        if _has_tool("kubeconform"):
            r = subprocess.run(
                ["kubeconform", "-strict", "-summary"],
                input=out, capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                _pass("kubeconform on Helm output — valid")
            else:
                _fail("kubeconform on Helm output — errors", r.stderr or r.stdout)
    else:
        _fail("helm template — failed", err)

    # Validation: missing workflow definition should fail
    rc, out, err = _run([
        "helm", "template", "test", str(chart), "-n", "agentloom",
    ])
    if rc != 0 and "workflow" in (err + out).lower():
        _pass("helm template without workflow.definition — fails with clear message")
    elif rc != 0:
        _pass("helm template without workflow.definition — fails (validation works)")
    else:
        _fail("helm template without workflow.definition should fail")


def audit_terraform_validate() -> None:
    _header("Terraform Validation (tool)")
    tf_dir = DEPLOY / "terraform"
    if not _has_tool("terraform"):
        _skip("terraform not installed — skipping Terraform validation")
        return

    # Check if .terraform exists (initialized)
    if not (tf_dir / ".terraform").is_dir():
        # Try to init
        rc, out, err = _run(["terraform", "init", "-backend=false"], cwd=tf_dir, timeout=120)
        if rc != 0:
            _fail("terraform init failed", err)
            return
        _pass("terraform init — OK")
    else:
        _pass("terraform already initialized")

    # Validate
    rc, out, err = _run(["terraform", "validate"], cwd=tf_dir, timeout=60)
    if rc == 0:
        _pass("terraform validate — OK")
    else:
        _fail("terraform validate — failed", err or out)

    # Format check
    rc, out, err = _run(["terraform", "fmt", "-check", "-recursive"], cwd=tf_dir, timeout=30)
    if rc == 0:
        _pass("terraform fmt — all files formatted")
    else:
        _fail("terraform fmt — formatting issues", out)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(description="AgentLoom Infrastructure Audit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show passing checks")
    parser.add_argument("--skip-tools", action="store_true", help="Skip tool-based validation")
    args = parser.parse_args()
    _verbose = args.verbose

    print(_bold("\n  AgentLoom Infrastructure Audit"))
    print(f"  {'=' * 40}")

    # Phase 1: Static analysis (pure Python)
    audit_dockerfile()
    audit_docker_compose()
    audit_k8s_base()
    audit_kustomize_overlays()
    audit_helm_chart()
    audit_terraform()
    audit_argocd()
    audit_ci_workflows()
    audit_docs()
    audit_cross_references()

    # Phase 2: Tool validation
    if not args.skip_tools:
        audit_kustomize_build()
        audit_helm_render()
        audit_terraform_validate()

    # Summary
    total = _pass_count + _fail_count + _skip_count
    print(f"\n{'=' * 60}")
    print(f"  {_bold('Results')}: {_green(f'{_pass_count} passed')}  "
          f"{_red(f'{_fail_count} failed') if _fail_count else f'{_fail_count} failed'}  "
          f"{_yellow(f'{_skip_count} skipped') if _skip_count else f'{_skip_count} skipped'}  "
          f"({total} total)")
    print(f"{'=' * 60}\n")

    return 1 if _fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
