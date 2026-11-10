---
title: "Deployment"
description: "Run AgentLoom in Docker, Docker Compose, Kubernetes with Kustomize or Helm, Terraform and Argo CD, and what each manifest sets up."
order: 17
section: "Operations"
---

# Deployment

AgentLoom is a command that runs a workflow and exits, so it deploys as a container that runs once: a Kubernetes Job, or a CronJob for scheduled runs. The repository ships a Docker image, a Compose stack with the observability tools, Kustomize manifests, a Helm chart, a Terraform setup for a local cluster and an Argo CD Application.

## Docker

```bash
docker build -t agentloom .
docker run --rm agentloom run 40_fault_replay.yaml --lite
docker run --rm -e OPENAI_API_KEY agentloom run 01_simple_qa.yaml --provider openai --model gpt-4o-mini
```

The image's entrypoint is `agentloom`, and its working directory, `/workflows`, holds the examples, with their recordings under `/workflows/recordings`. Examples 35 to 41 run there offline; `/workflows/examples` points back at the directory, for the ones that name their files under `examples/`.

| Target | What it is |
|--------|------------|
| `production`, the default | The package with the `observability` extra: a container that sets `OTEL_EXPORTER_OTLP_ENDPOINT` exports traces and metrics. |
| `production-lite` | The package without the extra. Build it with `--target production-lite`. |
| `dev` | The development dependencies, the sources and the tests; it runs `pytest`. Build it with `--target dev`. |

Both production images run as the user `agentloom` (UID 1000), on `python:3.14-slim`, and have no pip: an image built `FROM` them that installs packages brings its own installer, such as `uv` (`COPY --from=ghcr.io/astral-sh/uv`). They expose no port and set `OTEL_EXPORTER_OTLP_ENDPOINT` to `http://otel-collector:4317`.

- **Your own workflows** go in a mount below `/workflows`, such as `-v ./workflows:/workflows/custom:ro` and `run custom/report.yaml`. A mount over `/workflows` itself hides the bundled recordings, and the mock examples then stop with an error. Paths in a workflow, such as `responses_file`, a subworkflow or an agent file, are relative to `/workflows`.
- **Files a run writes** need a writable directory, and `/workflows` belongs to root. Without one, the run history is lost without a warning, and `--checkpoint` logs that it can't save, so a run paused at an approval gate can't be resumed. Mount a volume the UID 1000 user can write, and point `AGENTLOOM_RUNS_DIR` and `--checkpoint-dir` at it.
- **Ollama on the host** isn't at `localhost` from inside the container: set `OLLAMA_BASE_URL`, as the Compose file does with `http://host.docker.internal:11434`.

### Published images

Images are published to `ghcr.io/cchinchilla-dev/agentloom`:

| Event | Tags | Platforms |
|-------|------|-----------|
| Release `v1.2.3` | `1.2.3`, `1.2`, and `latest` when it is the newest release | `linux/amd64`, `linux/arm64` |
| Push to `main` | `main-<short sha>` | `linux/amd64` |
| Pull request | `pr-<number>`, built and tested but not pushed | `linux/amd64` |

## Docker Compose

`deploy/docker-compose.yml` runs the observability stack, and AgentLoom on demand:

```bash
cd deploy
docker compose up -d
docker compose run --rm agentloom run /workflows/01_simple_qa.yaml
docker compose down
```

`up` starts the stack only; the `agentloom` service runs with `run`, reads keys from `../.env` when the file exists, and reaches Ollama on the host.

| Service | Ports | What it does |
|---------|-------|--------------|
| OTel Collector | 4317 (gRPC), 4318 (HTTP), 8889 (metrics for Prometheus) | Receives traces and metrics; sends traces to Jaeger. |
| Jaeger | 16686 (UI), 14268 | Stores and shows traces. |
| Prometheus | 9090 | Scrapes the collector every 5 seconds and keeps 7 days. |
| Grafana | 3000 | The AgentLoom dashboard, with Prometheus and Jaeger as data sources. |
| Grafana image renderer | 8081 | Renders panels as images. |

Grafana logs in with `admin` / `admin`, and lets anonymous visitors edit: keep the stack on your machine. [Observability](observability.md) covers what the dashboard shows.

## Kubernetes with Kustomize

`deploy/k8s/base` holds the manifests, and three overlays adjust them. Each overlay creates, in the namespace `agentloom`:

- the namespace, labelled for the `restricted` Pod Security profile;
- the ServiceAccount `agentloom`, without an API token;
- the ConfigMap `agentloom-workflows`, whose `workflow.yaml` is the workflow to run;
- the Secret `agentloom-provider-keys`, empty;
- the Job `agentloom-workflow`, which runs at once, and the CronJob `agentloom-scheduled`, which runs every 6 hours (UTC);
- a NetworkPolicy, except in `dev`, and a ResourceQuota.

The Job and the CronJob run `agentloom run /workflows/workflow.yaml` with the Secret's keys as environment variables, a read-only root filesystem and a writable `/tmp`. A run may take 600 seconds.

| Overlay | Image tag | Memory (request / limit) | CPU request | NetworkPolicy | `backoffLimit` |
|---------|-----------|--------------------------|-------------|---------------|----------------|
| `dev` | `latest` | 64Mi / 128Mi | 50m | None | 2 |
| `staging` | `main-abc1234`, a placeholder for a CI tag | 128Mi / 256Mi | 100m | The base's | 2 |
| `production` | The current release | 256Mi / 512Mi | 200m | Stricter: no Ollama | 1 |

No overlay sets a CPU limit: the calls wait on the network, and a CPU limit would only throttle them.

**Before deploying,** replace the workflow in the ConfigMap: the placeholder answers from the mock provider, so it only shows that the Job runs. Then create the namespace and the keys, and apply the overlay:

```bash
kubectl apply -f deploy/k8s/base/namespace.yaml
kubectl create secret generic agentloom-provider-keys -n agentloom --from-env-file=provider-keys.env
kubectl apply -k deploy/k8s/overlays/dev
kubectl logs job/agentloom-workflow -n agentloom
```

`provider-keys.env` holds one `NAME=value` line per key, such as `OPENAI_API_KEY`; every line becomes a variable of the run. Created this way, the Secret keeps its keys when the overlay is applied again. Filled after the overlay with `kubectl apply`, it loses them at the next `kubectl apply -k`.

- **Traces and metrics** go to `http://otel-collector:4317`, a name that only resolves to a collector in the `agentloom` namespace, which the manifests don't create. The `production` overlay sends them to `http://otel-collector.observability.svc.cluster.local:4317`, in the namespace its NetworkPolicy allows. For another collector, set `OTEL_EXPORTER_OTLP_ENDPOINT` to its full name. A run whose export fails still succeeds.
- **The run history and checkpoints** need a volume that outlives the pod: see [Keeping runs](#keeping-runs).
- **A Job that is stopped,** deleted or past its `activeDeadlineSeconds`, gets SIGTERM, which AgentLoom takes as Ctrl-C: with `--checkpoint` on a volume, the run saves a `cancelled` checkpoint that `agentloom resume` continues.
- `kubectl delete -k deploy/k8s/overlays/dev` removes everything, the namespace included.

### Keeping runs

A pod's `/tmp` goes with it, so without a volume a run paused at an approval gate can't be resumed. Add one with a patch in your overlay, on a PersistentVolumeClaim of yours:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: agentloom-workflow
spec:
  template:
    spec:
      containers:
        - name: agentloom
          args: ["run", "/workflows/workflow.yaml", "--checkpoint", "--checkpoint-dir", "/data/checkpoints"]
          env:
            - name: AGENTLOOM_RUNS_DIR
              value: /data/runs
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: agentloom-data
```

The chart does the same with `persistence.existingClaim`.

### Example Jobs

`deploy/k8s/examples/` holds Jobs that check one feature each, in the `agentloom` namespace with its ServiceAccount: apply an overlay first, then `kubectl apply -f` the file.

| File | Checks | Image |
|------|--------|-------|
| `approval-gate-job.yaml` | A gate pauses the run, and resuming it with an approval finishes it. | `agentloom:dev` |
| `pause-resume-job.yaml` | A step that asks to pause, and the resume from its checkpoint. | `agentloom:dev` |
| `webhook-approval-job.yaml` | A gate's notification reaches a server in the pod, and the run resumes. | `agentloom:dev` |
| `mock-provider-job.yaml` | Recording two calls and replaying them. | `agentloom:dev` |
| `conversation-job.yaml` | Three steps sharing a conversation, each with its `speaker`. | The published image |
| `array-index-job.yaml` | That `{state.items[0]}` in a prompt renders the item. | The published image |

The `agentloom:dev` Jobs use an image you build and load into the cluster yourself, as they never pull it: `docker build --target dev -t agentloom:dev .`.

## Helm

The chart in `deploy/helm/agentloom` runs a workflow as a Job, or as a CronJob with `schedule.enabled=true`:

```bash
kubectl apply -f deploy/k8s/base/namespace.yaml
kubectl create secret generic provider-keys -n agentloom --from-env-file=provider-keys.env
helm install agentloom deploy/helm/agentloom -n agentloom \
  --set-file workflow.definition=workflows/report.yaml \
  --set provider.existingSecret=<secret-name>
```

`<secret-name>` is the Secret's name, `provider-keys` here. `--set-file` reads the workflow as it is; `--set` would split it at its commas. Helm's `--create-namespace` makes a namespace without the Pod Security labels, which `namespace.yaml` sets.

The chart refuses to render without a workflow, or with both `workflow.definition` and `workflow.existingConfigMap`, and its `values.schema.json` refuses unknown keys and wrong types. A Job can't change once created: apply a change with `helm uninstall` and `helm install`, or with Argo CD's `Replace=true`.

| Value | Default | What it does |
|-------|---------|--------------|
| `image.repository`, `image.tag` | `ghcr.io/cchinchilla-dev/agentloom`, the chart's `appVersion` | The image. |
| `workflow.definition` | | The workflow YAML, put in the ConfigMap `<name>-workflow`. |
| `workflow.existingConfigMap` | | A ConfigMap of yours to mount instead, with a `workflow.yaml` key. |
| `workflow.args` | `[]` | Options added after `run /workflows/workflow.yaml`. |
| `schedule.enabled`, `schedule.cron`, `schedule.timeZone` | `false`, `0 */6 * * *`, `UTC` | A CronJob instead of a Job. `concurrencyPolicy` is `Forbid`. |
| `job.backoffLimit`, `job.ttlSecondsAfterFinished`, `job.activeDeadlineSeconds` | `2`, `3600`, none | Retries of the pod, how long a finished Job stays, the time limit. |
| `provider.existingSecret` | | A Secret whose every key becomes a variable of the run. |
| `provider.openaiApiKey`, `anthropicApiKey`, `googleApiKey` | | Keys for a Secret the chart creates. They end up in the release's stored values: prefer `existingSecret`. |
| `observability.enabled`, `observability.otelEndpoint` | `true`, `http://otel-collector:4317` | Sets `OTEL_EXPORTER_OTLP_ENDPOINT`. Use the collector's full name. |
| `persistence.existingClaim` | | A PersistentVolumeClaim of yours, mounted on `/data`. It keeps the run history and turns checkpoints on, so a run paused in one pod can be resumed from another. |
| `resources` | 128Mi and 100m requested, 256Mi limit | |
| `serviceAccount.create` | `true` | A ServiceAccount without an API token. |
| `namespace.create` | `false` | A namespace with the Pod Security labels. |
| `networkPolicy.enabled`, `networkPolicy.allowOllama` | `true`, `true` | The NetworkPolicy, and egress to port 11434. |
| `resourceQuota.enabled` | `false` | A ResourceQuota. Its 2Gi of `limits.memory` is below Ollama's limit: with `ollama.enabled`, raise `resourceQuota.hard` or lower `ollama.resources`. |
| `ollama.enabled`, `ollama.model` | `false`, `phi4` | An Ollama Deployment and Service in the release. |

The chart names everything after the release, adding `-agentloom` unless the release name contains `agentloom`.

**With `ollama.enabled`,** the release runs Ollama, which pulls the model when its pod starts, onto an `emptyDir`, so again after each restart. The run gets `OLLAMA_BASE_URL` pointing at it and `AGENTLOOM_OLLAMA_FALLBACK=1`, so Ollama also answers models no other provider takes. Ollama may use up to 10Gi of memory; `phi4` takes about 9 GB. The run waits for Ollama to be ready, up to 30 minutes, and fails if it isn't by then.

## Terraform

`deploy/terraform` builds a local environment: a kind cluster with the chart installed and, by default, the observability stack.

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init && terraform apply
```

| Variable | Default | What it does |
|----------|---------|--------------|
| `cluster_name` | `agentloom-dev` | The kind cluster. |
| `agentloom_image_tag` | `latest` | The image the chart runs. |
| `enable_observability` | `true` | Jaeger, the OTel Collector, Prometheus and Grafana, in the namespace `observability`. |
| `enable_ollama` | `true` | Ollama in the release, with `ollama_model` (`phi4`). |
| `provider_api_keys` | `{}` | Keys for a Secret the chart then uses. |
| `workflow_definition` | A one-step workflow on Ollama | The workflow to run. |

Grafana is on `http://localhost:3000`, Prometheus on `http://localhost:9090` and Jaeger on `http://localhost:16686`; `terraform output` prints them. The collector is reachable inside the cluster only. Ollama with `phi4` needs a Docker host with about 12 GB of memory.

## Argo CD

`deploy/argocd/application.yaml` syncs the Helm chart from the repository's `HEAD`, with automated sync, pruning and self-heal. It mounts the ConfigMap `agentloom-workflows` as the workflow and gives the run the keys in the Secret `agentloom-provider-keys`, so create both before the first sync:

```bash
kubectl apply -f deploy/k8s/base/namespace.yaml
kubectl create configmap agentloom-workflows -n agentloom --from-file=workflow.yaml=workflows/report.yaml
kubectl create secret generic agentloom-provider-keys -n agentloom --from-env-file=provider-keys.env
kubectl apply -f deploy/argocd/application.yaml
```

- **`Replace=true`** replaces the Job on each change, as a Job can't be patched, and `ignoreDifferences` keeps its generated selector and labels from showing as out of sync.
- **Failed syncs** are retried 5 times, waiting from 5 seconds up to 3 minutes.
- `managedNamespaceMetadata` labels the namespace for the `restricted` Pod Security profile, even if Argo CD creates it.
- The Application uses the `default` project; a dedicated AppProject suits production better.

## Environment variables

| Variable | What it does |
|----------|--------------|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | Set up each provider, or give their keys to the providers a config file lists. |
| `OLLAMA_BASE_URL` | Ollama's address; `http://localhost:11434` by default. |
| `AGENTLOOM_OLLAMA_FALLBACK` | `1` makes Ollama answer the models no other provider takes. |
| `AGENTLOOM_CONFIG` | The [config file](providers.md#config-file) that sets up the providers. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Where traces and metrics go, over gRPC. Without the `observability` extra, or with `--lite`, nothing is sent. |
| `AGENTLOOM_RUNS_DIR` | Where run history records go; `./agentloom_runs` by default. |
| `AGENTLOOM_REDACT_STATE_KEYS` | State keys to mask in checkpoints and results. See [Redaction](state-and-templates.md#redaction). |
| `AGENTLOOM_LOG_LEVEL`, `AGENTLOOM_LOG_FORMAT` | The level and format of the CLI's logs, on stderr. See [Logs](observability.md#logs). |

[Providers](providers.md#environment-variables) lists every variable AgentLoom reads. In Kubernetes, the provider keys come from a Secret, and the other variables are set in the pod spec.

## Approval callback server

`agentloom callback-server` takes approvals and rejections over HTTP for runs paused at an approval gate; the [CLI reference](cli.md#callback-server) lists its endpoints and options. The manifests don't deploy it. To run it in a cluster:

- It resumes runs in its own process, so it needs the provider keys and the paused runs' checkpoint files, on a volume shared with the Jobs that paused, set with `--checkpoint-dir`.
- The shipped NetworkPolicies deny all ingress to pods labelled `app.kubernetes.io/name: agentloom`: give the server other labels, or a policy of its own.
- A gate notifying a Service in the cluster needs `sandbox.allow_internal_webhook_targets: true`, as cluster addresses are internal, and egress to the Service's port.
- `--token` is a command-line option, so it shows in the pod spec.

A Job running `agentloom resume <run_id> --approve --checkpoint-dir <dir>` resumes a run without a server.

## CI/CD pipeline

| Workflow | Runs on | What it does |
|----------|---------|--------------|
| `ci.yml` | Pushes to `main`, pull requests | Lint, types and the workflow audit; tests on Python 3.11 to 3.14 on Linux, 3.14 on macOS and Windows, and the lowest dependency versions; the docs check; the wheel and sdist; version linearity; the Kubernetes manifests and the Helm chart. |
| `docker.yml` | Pushes and pull requests to `main`, releases, manual | Builds the image and pushes it, except for pull requests; runs pull request and release images once to test them. |
| `auto-tag.yml` | Pushes to `main` | Tags `vX.Y.Z`, from `pyproject.toml`, when the commit is `bump version to X.Y.Z`. |
| `release.yml` | `v*` tags, after Auto Tag, manual | Builds and checks the wheel and sdist, then the GitHub release, the Docker image and the PyPI upload; once PyPI serves the version, rebuilds this site. |
| `e2e-ollama.yml` | `release/**` branches, pull requests labelled `e2e`, manual | Tests against a live Ollama. |
| `pricing-update.yml` | Mondays, manual | Refreshes the price table and opens a pull request. |
| `pr-labeler.yml` | Pull requests | Labels them by the files they change. |
| `labels.yml` | Changes to `.github/labels.json`, manual | Syncs the repository's labels. |
| `stale.yml` | Mondays, manual | Closes `question` issues left unanswered. |

Every action is pinned to a commit, and `make check` and the lint job audit the workflows with [zizmor](https://docs.zizmor.sh). Dependabot opens weekly pull requests for the actions, `uv.lock` and the Dockerfile's images, for releases at least a week old.

## Security

The Kustomize manifests and the chart run AgentLoom under the Kubernetes `restricted` Pod Security profile:

- **Non-root:** UID 1000, `runAsNonRoot`, no privilege escalation, every capability dropped, the `RuntimeDefault` seccomp profile.
- **Read-only root filesystem,** with an `emptyDir` on `/tmp`.
- **No API server access:** the ServiceAccount mounts no token.
- **Egress** to DNS in `kube-system`, HTTPS anywhere, and ports 4317 and 11434 in the cluster; production drops 11434 and allows 4317 only to the `observability` namespace. No ingress. Traffic leaving the cluster may use port 443 only, so an Ollama or an OpenAI-compatible server outside it on another port is blocked.
- **Keys** live in Secrets, never in the image.

The profile is enforced only in a namespace labelled for it: the one `namespace.yaml`, Terraform, the chart's `namespace.create` or the Argo CD Application creates. The example Jobs, except `array-index-job.yaml`, and the chart's Ollama don't use a read-only root filesystem.

## What's next

<!-- cards -->

- **[Observability](observability.md)** — The traces, metrics and dashboard of a deployed run.
- **[Reliability](reliability.md)** — Checkpoints and resuming paused runs.
- **[Providers](providers.md)** — The keys and endpoints a deployment sets.

<!-- /cards -->
