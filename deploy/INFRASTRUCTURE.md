# Infrastructure

AgentLoom runs as a **batch Job/CronJob** in Kubernetes, not as a long-running server.
There is no Deployment, Service, Ingress, or HPA — the CLI processes a workflow YAML and exits.

## Table of Contents

- [Architecture](#architecture)
- [Deployment Methods](#deployment-methods)
  - [Docker](#docker)
  - [Docker Compose](#docker-compose)
  - [Kubernetes (Kustomize)](#kubernetes-kustomize)
  - [Helm](#helm)
  - [Terraform](#terraform)
  - [ArgoCD (GitOps)](#argocd-gitops)
- [CI/CD Pipeline](#cicd-pipeline)
- [Environment Variables](#environment-variables)
- [Kustomize Overlays](#kustomize-overlays)
- [Helm Chart Reference](#helm-chart-reference)
- [Security](#security)
- [Directory Structure](#directory-structure)

## Architecture

```
Testing library (Python)
  -> creates K8s Job with workflow YAML
  -> agentloom runs isolated in pod, emits traces + metrics
  -> library queries Jaeger/Prometheus to evaluate
  -> quality gate: pass/fail based on hybrid metrics
```

### Component flow

```
+---------------+     +------------------+     +--------------+
| K8s Job       |---->| OTel Collector   |---->| Jaeger       |
| (agentloom)   |     | (gRPC :4317)     |     | (traces)     |
+---------------+     +--------+---------+     +--------------+
                               |
                               +-------------->+--------------+
                                               | Prometheus   |
                                               | (metrics)    |
                                               +------+-------+
                                                      |
                                               +------v-------+
                                               | Grafana      |
                                               | (dashboards) |
                                               +--------------+
```

### Why K8s Jobs, not Deployments?

AgentLoom is a CLI that processes a workflow and exits. It has no HTTP API, no persistent
connections, and no reason to stay running. A Kubernetes Job is the correct primitive:

- **Finite execution**: the pod runs, completes, and is cleaned up via `ttlSecondsAfterFinished`.
- **Automatic retries**: `backoffLimit` handles transient failures without external orchestration.
- **Scheduled runs**: CronJobs handle periodic execution (e.g., nightly batch processing).
- **Resource isolation**: each workflow run gets its own pod with dedicated resource limits.

## Deployment Methods

### Docker

The simplest way to run agentloom. The multi-stage Dockerfile produces a minimal
image (~120MB) with a non-root user.

```bash
# Build
docker build -t agentloom .

# Run a workflow
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v ./examples:/workflows:ro \
  agentloom run /workflows/01_simple_qa.yaml

# Validate without running
docker run --rm \
  -v ./examples:/workflows:ro \
  agentloom validate /workflows/01_simple_qa.yaml

# Override provider and model
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v ./examples:/workflows:ro \
  agentloom run /workflows/01_simple_qa.yaml --provider openai --model gpt-4o-mini
```

Build with observability extras:

```bash
docker build --build-arg BUILD_OBSERVABILITY=true -t agentloom:obs .
```

### Docker Compose

Runs agentloom alongside the full observability stack (Prometheus, Grafana, Jaeger,
OTel Collector).

```bash
cd deploy

# Start observability stack
docker compose up -d

# Run a workflow (on-demand, not part of `up`)
docker compose run --rm agentloom run /workflows/01_simple_qa.yaml

# Stop everything
docker compose down
```

Access the dashboards:
- **Grafana**: http://localhost:3000 (admin / admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

### Kubernetes (Kustomize)

Raw K8s manifests organized with Kustomize overlays. Best for teams that prefer
plain YAML without Helm templating.

```bash
# Create a local cluster
kind create cluster

# Load local image (optional, if not pulling from GHCR)
docker build -t agentloom:local .
kind load docker-image agentloom:local

# Deploy dev overlay
kubectl apply -k deploy/k8s/overlays/dev

# Check status
kubectl get jobs -n agentloom
kubectl logs job/agentloom-workflow -n agentloom

# Clean up
kubectl delete -k deploy/k8s/overlays/dev
```

**Important**: The base secret (`deploy/k8s/base/secret.yaml`) ships with `data: {}`
by design. Populate it before deploying:

```bash
kubectl create secret generic agentloom-provider-keys \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  -n agentloom --dry-run=client -o yaml | kubectl apply -f -
```

See [Kustomize Overlays](#kustomize-overlays) for per-environment configuration.

### Helm

The recommended deployment method. Packages all K8s resources into a configurable chart
with built-in validation.

```bash
# Lint
helm lint deploy/helm/agentloom

# Dry-run
helm template agentloom deploy/helm/agentloom \
  --set workflow.definition="$(cat examples/01_simple_qa.yaml)" \
  --set provider.existingSecret=my-secret

# Install
helm install agentloom deploy/helm/agentloom \
  -n agentloom --create-namespace \
  --set workflow.definition="$(cat examples/01_simple_qa.yaml)" \
  --set provider.existingSecret=my-secret

# Upgrade (note: Jobs are immutable — uninstall first or use ArgoCD with Replace=true)
helm uninstall agentloom -n agentloom
helm install agentloom deploy/helm/agentloom -n agentloom --set ...

# CronJob mode (scheduled execution)
helm install agentloom deploy/helm/agentloom \
  -n agentloom --create-namespace \
  --set schedule.enabled=true \
  --set schedule.cron="0 */6 * * *" \
  --set workflow.definition="$(cat examples/01_simple_qa.yaml)"
```

The chart validates inputs at render time:
- Missing `workflow.definition` **and** `workflow.existingConfigMap` → fails with a clear message.
- Both set simultaneously → fails (pick one).

See [Helm Chart Reference](#helm-chart-reference) for the full `values.yaml` documentation.

### Terraform

Provisions a complete local development environment in one command: kind cluster +
agentloom + observability stack (OTel Collector, Prometheus, Grafana, Jaeger).

```bash
cd deploy/terraform

# Configure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# Deploy everything
terraform init
terraform plan
terraform apply

# Access
export KUBECONFIG=$(terraform output -raw kubeconfig)
kubectl get jobs -n agentloom
kubectl logs job/agentloom -n agentloom

# Tear down
terraform destroy
```

When `enable_observability = true` (default), Terraform deploys:
- **OTel Collector**: receives traces and metrics from agentloom pods (gRPC :4317).
- **Jaeger**: trace storage and UI (http://localhost:16686).
- **Prometheus**: metrics scraping from OTel Collector (http://localhost:9090).
- **Grafana**: dashboards with the agentloom dashboard pre-loaded (http://localhost:3000, admin/admin).

Set `enable_observability = false` for a lightweight setup without the observability stack.

### ArgoCD (GitOps)

For continuous deployment from the Git repository. ArgoCD watches the Helm chart in
`deploy/helm/agentloom` and syncs changes automatically.

```bash
# Install ArgoCD (if not already running)
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Deploy the agentloom Application
kubectl apply -f deploy/argocd/application.yaml
```

The Application CRD configures:
- **Automated sync** with prune and self-heal.
- **Replace=true** for immutable resources (K8s Jobs cannot be patched in-place).
- **ignoreDifferences** on Job selector/labels to prevent perpetual out-of-sync.
- **Retry policy**: 5 attempts with exponential backoff (5s → 3m max).
- **CreateNamespace**: the `agentloom` namespace is created if it does not exist.

> **Note**: Secrets are not managed by ArgoCD. Create them manually or use an external
> secret management tool (External Secrets Operator, Sealed Secrets, SOPS).

## CI/CD Pipeline

### Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Push / PR to `main` | Lint, type check, test (458 tests), Docker build, K8s validation, Helm lint |
| `docker.yml` | Push to `main` / tags `v*` | Build multi-arch image, push to GHCR, smoke test on PRs |
| `release.yml` | Tags `v*` | Build wheel, create GitHub release, publish to PyPI |
| `pr-labeler.yml` | PR opened/updated | Auto-label PRs based on file paths |
| `labels.yml` | Push to `main` (labels.json) | Sync GitHub labels from config |
| `stale.yml` | Weekly (Monday 9am UTC) | Mark/close stale issues |

### Image tags

The Docker workflow produces these tags:

| Event | Tag pattern | Example |
|---|---|---|
| Push to `main` | `main-<sha>` | `main-a1b2c3d` |
| Tag `v1.2.3` | `1.2.3`, `1.2`, `latest` | `1.2.3` |
| Pull request | `pr-<number>` | `pr-42` |

Images are published to `ghcr.io/cchinchilla-dev/agentloom`.

### Supply chain security

All GitHub Actions are pinned to commit SHAs (not mutable tags) to prevent
supply chain attacks. Example:

```yaml
# Instead of: uses: actions/checkout@v4
uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5  # v4
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GOOGLE_API_KEY` | Google AI API key | — |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTel Collector gRPC endpoint | `http://localhost:4317` |

In Kubernetes, these are injected via `envFrom` referencing a Secret. Never bake API
keys into images or commit them to source control.

## Kustomize Overlays

Three overlays are provided for different environments:

| Overlay | Image tag | Memory (req/limit) | CPU request | NetworkPolicy | activeDeadlineSeconds |
|---|---|---|---|---|---|
| **dev** | `latest` | 64Mi / 128Mi | 50m | Disabled | — |
| **staging** | CI SHA | 128Mi / 256Mi | 100m | Enabled | — |
| **production** | Pinned version | 256Mi / 512Mi | 200m | Enabled (strict, no Ollama) | 600 |

Key differences:
- **dev**: Minimal resources, no NetworkPolicy, `latest` tag for fast iteration.
- **staging**: Moderate resources, NetworkPolicy enabled (allows Ollama egress), matches CI image.
- **production**: High resources, strict NetworkPolicy (no Ollama — only HTTPS, DNS, OTel), `activeDeadlineSeconds` enforces a hard timeout, image pinned to a specific version.

No overlay sets CPU limits — this is intentional. CPU limits cause CFS throttling during
LLM API calls (which are I/O-bound, not CPU-bound). CPU requests are set for scheduling guarantees.

## Helm Chart Reference

### Key values

| Parameter | Description | Default |
|---|---|---|
| `image.repository` | Container image repository | `ghcr.io/cchinchilla-dev/agentloom` |
| `image.tag` | Image tag (defaults to `appVersion`) | `""` |
| `workflow.definition` | Inline workflow YAML | `""` |
| `workflow.existingConfigMap` | Reference to existing ConfigMap | `""` |
| `workflow.args` | Extra CLI arguments | `[]` |
| `schedule.enabled` | `false` = Job, `true` = CronJob | `false` |
| `schedule.cron` | CronJob schedule | `0 */6 * * *` |
| `schedule.timeZone` | CronJob timezone | `UTC` |
| `provider.existingSecret` | Pre-created Secret with API keys | `""` |
| `provider.openaiApiKey` | OpenAI key (not recommended, use secret) | `""` |
| `observability.enabled` | Enable OTel endpoint env var | `true` |
| `observability.otelEndpoint` | OTel Collector endpoint | `http://otel-collector:4317` |
| `job.backoffLimit` | Retry attempts on failure | `2` |
| `job.ttlSecondsAfterFinished` | Cleanup delay after completion | `3600` |
| `job.activeDeadlineSeconds` | Hard timeout | `null` |
| `serviceAccount.create` | Create a ServiceAccount | `true` |
| `serviceAccount.automountServiceAccountToken` | Mount K8s API token | `false` |
| `namespace.create` | Create namespace with PSS labels | `false` |
| `networkPolicy.enabled` | Enable NetworkPolicy | `true` |
| `networkPolicy.allowOllama` | Allow egress to Ollama (port 11434) | `true` |
| `resourceQuota.enabled` | Enable namespace ResourceQuota | `false` |

### Validation

The chart includes a validation template (`templates/validate.yaml`) that fails at
render time if:
- Neither `workflow.definition` nor `workflow.existingConfigMap` is set.
- Both are set simultaneously.

This prevents deploying pods that would crash trying to read an empty workflow file.

### Jobs and immutability

Kubernetes Jobs are **immutable after creation**. This means:
- `helm upgrade` cannot modify a running or completed Job.
- Use `helm uninstall` + `helm install` for changes, or use ArgoCD with `Replace=true`.
- The ArgoCD Application CRD is configured with `Replace=true` and `ignoreDifferences`
  to handle this automatically.

## Security

All workloads enforce the Kubernetes **Pod Security Standards "restricted" profile**:

- **Non-root container**: runs as `agentloom` (UID 1000) — enforced via both Dockerfile
  `USER` directive and K8s `securityContext.runAsNonRoot`.
- **Read-only root filesystem**: `readOnlyRootFilesystem: true` with a writable `/tmp`
  via `emptyDir` volume.
- **No privilege escalation**: `allowPrivilegeEscalation: false`, all Linux capabilities
  dropped (`drop: ["ALL"]`).
- **seccompProfile**: `RuntimeDefault` blocks dangerous syscalls.
- **No API server access**: `automountServiceAccountToken: false` — pods cannot access
  the Kubernetes API.
- **NetworkPolicy**: restricts pod egress to DNS (kube-system only), HTTPS (443), and
  OTel (4317). Production overlay removes Ollama (11434). All ingress is denied.
- **Secrets management**: provider API keys are stored in K8s Secrets, never baked into
  images. Use `existingSecret` or external tools (External Secrets Operator, Sealed
  Secrets, SOPS) for production.
- **ResourceQuota**: optional namespace-level limits to prevent resource exhaustion.
- **No CPU limits**: CPU limits removed to avoid CFS throttling during I/O-bound LLM
  API calls. CPU requests are set for scheduling.
- **PSS namespace labels**: the `agentloom` namespace is labeled with
  `pod-security.kubernetes.io/enforce: restricted` (both Kustomize and Helm).
- **GitHub Actions pinned to SHAs**: all CI/CD actions reference commit SHAs instead
  of mutable tags to prevent supply chain attacks.

## Directory Structure

```
deploy/
  docker-compose.yml              # Local observability stack
  otel-collector-config.yaml      # OTel Collector pipeline config
  prometheus.yml                  # Prometheus scrape targets
  INFRASTRUCTURE.md               # This file
  DASHBOARD.md                    # Grafana dashboard documentation
  grafana/                        # Grafana provisioning + dashboards
    dashboards/agentloom.json
    provisioning/
      dashboards/dashboards.yaml
      datasources/datasources.yaml
  k8s/                            # Raw Kubernetes manifests
    base/                         # Kustomize base (shared resources)
      namespace.yaml              # Namespace with PSS labels
      serviceaccount.yaml         # SA with automountToken=false
      configmap.yaml              # Workflow YAML mount
      secret.yaml                 # API keys placeholder (data: {})
      job.yaml                    # One-shot workflow execution
      cronjob.yaml                # Scheduled workflow execution
      networkpolicy.yaml          # Egress restrictions
      resourcequota.yaml          # Namespace resource limits
      kustomization.yaml          # Kustomize entrypoint
    overlays/
      dev/                        # Low resources, no netpol
      staging/                    # Moderate resources, netpol enabled
      production/                 # High resources, strict netpol, deadlines
  helm/
    agentloom/                    # Helm chart
      Chart.yaml                  # Chart metadata (v0.1.0, appVersion 0.1.2)
      values.yaml                 # Default configuration
      .helmignore                 # Files excluded from packaging
      ci/test-values.yaml         # Values for CI lint/template
      templates/
        _helpers.tpl              # Shared template helpers (podSpec)
        validate.yaml             # Fail-fast input validation
        job.yaml                  # Job template
        cronjob.yaml              # CronJob template
        configmap.yaml            # Workflow ConfigMap
        secret.yaml               # Provider API keys Secret
        serviceaccount.yaml       # ServiceAccount
        namespace.yaml            # Namespace (optional)
        networkpolicy.yaml        # NetworkPolicy
        resourcequota.yaml        # ResourceQuota (optional)
        NOTES.txt                 # Post-install instructions
  terraform/                      # Infrastructure as Code
    versions.tf                   # Provider requirements
    variables.tf                  # Input variables with validation
    main.tf                       # Kind cluster + namespace + Helm release
    outputs.tf                    # Cluster info and URLs
    terraform.tfvars.example      # Example variable values
    .gitignore                    # Exclude .terraform/, state files
    .terraform.lock.hcl           # Provider lock (committed)
  argocd/
    application.yaml              # ArgoCD Application CRD
```
