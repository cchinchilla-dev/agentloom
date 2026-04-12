# Deployment

AgentLoom runs anywhere — from a single Docker container to a fully orchestrated Kubernetes cluster with GitOps and observability. The CLI processes a workflow and exits; there is no long-running server.

!!! tip "Why Jobs, not Deployments?"
    AgentLoom is a CLI that exits after processing. Kubernetes **Jobs** provide finite execution, automatic retries via `backoffLimit`, scheduled runs via CronJobs, and clean resource isolation per workflow.

## Docker

The multi-stage Dockerfile produces a minimal image (~120MB) with a non-root user and read-only filesystem.

```bash
# Build
docker build -t agentloom .

# Run a workflow
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v ./examples:/workflows:ro \
  agentloom run /workflows/01_simple_qa.yaml

# Override provider and model
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v ./examples:/workflows:ro \
  agentloom run /workflows/01_simple_qa.yaml --provider openai --model gpt-4o-mini

# Build with observability extras
docker build --build-arg BUILD_OBSERVABILITY=true -t agentloom:obs .
```

## Docker Compose

Runs AgentLoom alongside the full observability stack:

```bash
cd deploy

# Start observability stack
docker compose up -d

# Run a workflow (on-demand, not part of `up`)
docker compose run --rm agentloom run /workflows/01_simple_qa.yaml

# Stop everything
docker compose down
```

| Service | Port | Purpose |
|---------|------|---------|
| Grafana | 3000 | Dashboard visualization (admin/admin) |
| Prometheus | 9090 | Metrics storage and PromQL queries |
| Jaeger | 16686 | Distributed tracing UI |
| OTel Collector | 4317 | Receives traces and metrics via gRPC |

## Kubernetes (Kustomize)

Plain YAML manifests organized with Kustomize overlays. Three environments provided:

| Overlay | Image Tag | Memory (req/limit) | CPU Request | NetworkPolicy | Deadline |
|---------|-----------|---------------------|-------------|---------------|----------|
| **dev** | `latest` | 64Mi / 128Mi | 50m | Disabled | — |
| **staging** | CI SHA | 128Mi / 256Mi | 100m | Enabled | — |
| **production** | Pinned | 256Mi / 512Mi | 200m | Strict (no Ollama) | 600s |

```bash
# Deploy dev overlay
kubectl apply -k deploy/k8s/overlays/dev
kubectl logs job/agentloom-workflow -n agentloom

# Clean up
kubectl delete -k deploy/k8s/overlays/dev
```

!!! warning "Secrets"
    The base secret ships with `data: {}` by design. Populate it before deploying:
    ```bash
    kubectl create secret generic agentloom-provider-keys \
      --from-literal=OPENAI_API_KEY=sk-... \
      --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
      -n agentloom --dry-run=client -o yaml | kubectl apply -f -
    ```

!!! info "No CPU limits"
    No overlay sets CPU limits — this is intentional. CPU limits cause CFS throttling during LLM API calls (which are I/O-bound, not CPU-bound). CPU requests are set for scheduling guarantees.

## Helm

Recommended for parameterized deployments. The chart validates inputs at render time — deploying without a workflow definition fails at render time, not at runtime.

=== "One-shot Job"

    ```bash
    helm install agentloom deploy/helm/agentloom \
      -n agentloom --create-namespace \
      --set workflow.definition="$(cat examples/01_simple_qa.yaml)" \
      --set provider.existingSecret=my-secret
    ```

=== "CronJob (scheduled)"

    ```bash
    helm install agentloom deploy/helm/agentloom \
      -n agentloom --create-namespace \
      --set schedule.enabled=true \
      --set schedule.cron="0 */6 * * *" \
      --set workflow.definition="$(cat examples/01_simple_qa.yaml)"
    ```

??? abstract "Helm chart reference"

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `image.repository` | `ghcr.io/cchinchilla-dev/agentloom` | Container image repository |
    | `image.tag` | `""` | Image tag (defaults to `appVersion`) |
    | `workflow.definition` | `""` | Inline workflow YAML |
    | `workflow.existingConfigMap` | `""` | Reference to existing ConfigMap |
    | `workflow.args` | `[]` | Extra CLI arguments |
    | `schedule.enabled` | `false` | `false` = Job, `true` = CronJob |
    | `schedule.cron` | `0 */6 * * *` | CronJob schedule |
    | `schedule.timeZone` | `UTC` | CronJob timezone |
    | `provider.existingSecret` | `""` | Pre-created Secret with API keys |
    | `provider.openaiApiKey` | `""` | OpenAI key (not recommended) |
    | `observability.enabled` | `true` | Enable OTel endpoint env var |
    | `observability.otelEndpoint` | `http://otel-collector:4317` | OTel Collector endpoint |
    | `job.backoffLimit` | `2` | Retry attempts on failure |
    | `job.ttlSecondsAfterFinished` | `3600` | Cleanup delay after completion |
    | `job.activeDeadlineSeconds` | `null` | Hard timeout |
    | `serviceAccount.create` | `true` | Create a ServiceAccount |
    | `serviceAccount.automountServiceAccountToken` | `false` | Mount K8s API token |
    | `namespace.create` | `false` | Create namespace with PSS labels |
    | `networkPolicy.enabled` | `true` | Enable NetworkPolicy |
    | `networkPolicy.allowOllama` | `true` | Allow egress to Ollama (port 11434) |
    | `resourceQuota.enabled` | `false` | Enable namespace ResourceQuota |

    **Validation:** The chart fails at render time if neither `workflow.definition` nor `workflow.existingConfigMap` is set, or if both are set simultaneously.

    **Jobs and immutability:** Kubernetes Jobs are immutable after creation. Use `helm uninstall` + `helm install` for changes, or use ArgoCD with `Replace=true`.

## Terraform

Provisions a complete local development environment in one command: kind cluster + AgentLoom + observability stack.

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
```

When `enable_observability = true` (default), Terraform deploys the full stack:

| Component | URL | Purpose |
|-----------|-----|---------|
| OTel Collector | gRPC :4317 | Receives traces and metrics |
| Jaeger | http://localhost:16686 | Trace storage and UI |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

Set `enable_observability = false` for a lightweight setup without the observability stack.

## ArgoCD

GitOps deployment with automated sync, self-heal, and retry policies. ArgoCD watches the Helm chart in the repository and syncs changes automatically.

```bash
kubectl apply -f deploy/argocd/application.yaml
```

The Application CRD configures:

- **Automated sync** with prune and self-heal
- **Replace=true** for immutable resources (K8s Jobs cannot be patched in-place)
- **ignoreDifferences** on Job selector/labels to prevent perpetual out-of-sync
- **Retry policy**: 5 attempts with exponential backoff (5s -> 3m max)
- **CreateNamespace**: the `agentloom` namespace is created if it does not exist

!!! note "Secrets"
    Secrets are not managed by ArgoCD. Create them manually or use External Secrets Operator / Sealed Secrets / SOPS.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GOOGLE_API_KEY` | Google AI API key | — |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTel Collector gRPC endpoint | `http://localhost:4317` |

In Kubernetes, these are injected via `envFrom` referencing a Secret. Never bake API keys into images or commit them to source control.

## CI/CD pipeline

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Push / PR to `main` | Lint, type check, test, Docker build, K8s validation, Helm lint |
| `docker.yml` | Push to `main` / tags `v*` | Build multi-arch image, push to GHCR |
| `release.yml` | Tags `v*` | Build wheel, GitHub release, publish to PyPI |
| `docs.yml` | Push to `main` (docs/mkdocs.yml) | Build and deploy documentation to GitHub Pages |
| `e2e-ollama.yml` | Weekly / `release/**` / `e2e` label | Ollama integration tests against live Docker instance |
| `pr-labeler.yml` | PR opened/updated | Auto-label PRs based on file paths |
| `labels.yml` | Push to `main` (labels.json) | Sync GitHub labels from config |
| `stale.yml` | Weekly (Monday 9am UTC) | Mark/close stale issues |

### Image tags

| Event | Tag pattern | Example |
|-------|-------------|---------|
| Push to `main` | `main-<sha>` | `main-a1b2c3d` |
| Tag `v1.2.3` | `1.2.3`, `1.2`, `latest` | `1.2.3` |
| Pull request | `pr-<number>` | `pr-42` |

Images are published to `ghcr.io/cchinchilla-dev/agentloom`.

## Security

All workloads enforce the Kubernetes **Pod Security Standards "restricted" profile**:

- :material-account-lock: **Non-root container** — runs as `agentloom` (UID 1000), enforced via Dockerfile and `securityContext.runAsNonRoot`
- :material-file-lock: **Read-only root filesystem** — `readOnlyRootFilesystem: true` with writable `/tmp` via `emptyDir`
- :material-shield-lock: **No privilege escalation** — `allowPrivilegeEscalation: false`, all Linux capabilities dropped
- :material-lock-outline: **seccomp** — `RuntimeDefault` profile blocks dangerous syscalls
- :material-key-remove: **No API server access** — `automountServiceAccountToken: false`
- :material-web-off: **NetworkPolicy** — restricts egress to DNS (kube-system), HTTPS (443), and OTel (4317). Production removes Ollama (11434). All ingress denied
- :material-shield-key: **Secrets management** — API keys stored in K8s Secrets, never baked into images
- :material-pin: **Supply chain security** — all GitHub Actions pinned to commit SHAs, not mutable tags
- :material-gauge: **ResourceQuota** — optional namespace-level limits to prevent resource exhaustion
- :material-label: **PSS namespace labels** — `pod-security.kubernetes.io/enforce: restricted`
