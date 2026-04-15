# -----------------------------------------------------------
# Kind cluster
# -----------------------------------------------------------

resource "kind_cluster" "this" {
  name           = var.cluster_name
  wait_for_ready = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"

      # Port mappings for local access to observability stack
      extra_port_mappings {
        container_port = 30000
        host_port      = 3000
        protocol       = "TCP"
      }
      extra_port_mappings {
        container_port = 30090
        host_port      = 9090
        protocol       = "TCP"
      }
      extra_port_mappings {
        container_port = 30686
        host_port      = 16686
        protocol       = "TCP"
      }
    }
  }
}

# -----------------------------------------------------------
# Providers configured after cluster creation
# -----------------------------------------------------------

provider "kubernetes" {
  host                   = kind_cluster.this.endpoint
  cluster_ca_certificate = kind_cluster.this.cluster_ca_certificate
  client_certificate     = kind_cluster.this.client_certificate
  client_key             = kind_cluster.this.client_key
}

provider "helm" {
  kubernetes {
    host                   = kind_cluster.this.endpoint
    cluster_ca_certificate = kind_cluster.this.cluster_ca_certificate
    client_certificate     = kind_cluster.this.client_certificate
    client_key             = kind_cluster.this.client_key
  }
}

# -----------------------------------------------------------
# Namespace
# -----------------------------------------------------------

resource "kubernetes_namespace" "agentloom" {
  metadata {
    name = "agentloom"
    labels = {
      "app.kubernetes.io/part-of"          = "agentloom"
      "pod-security.kubernetes.io/enforce" = "restricted"
      "pod-security.kubernetes.io/audit"   = "restricted"
      "pod-security.kubernetes.io/warn"    = "restricted"
    }
  }

  depends_on = [kind_cluster.this]
}

# -----------------------------------------------------------
# Provider API keys secret
# -----------------------------------------------------------

resource "kubernetes_secret" "provider_keys" {
  count = length(var.provider_api_keys) > 0 ? 1 : 0

  metadata {
    name      = "agentloom-provider-keys"
    namespace = kubernetes_namespace.agentloom.metadata[0].name
  }

  string_data = var.provider_api_keys
}

# -----------------------------------------------------------
# Observability stack (conditional)
# -----------------------------------------------------------
# When enable_observability = true, deploys:
#   - OpenTelemetry Collector (receives traces + metrics from agentloom)
#   - Jaeger (trace storage and UI)
#   - kube-prometheus-stack (Prometheus + Grafana with dashboards)
#
# When false, the OTel SDK in agentloom degrades gracefully to no-ops.

resource "kubernetes_namespace" "observability" {
  count = var.enable_observability ? 1 : 0

  metadata {
    name = "observability"
    labels = {
      "app.kubernetes.io/part-of"          = "agentloom"
      "pod-security.kubernetes.io/enforce" = "baseline"
      "pod-security.kubernetes.io/audit"   = "baseline"
      "pod-security.kubernetes.io/warn"    = "baseline"
    }
  }

  depends_on = [kind_cluster.this]
}

# --- Jaeger (all-in-one for development) ---

resource "helm_release" "jaeger" {
  count      = var.enable_observability ? 1 : 0
  name       = "jaeger"
  namespace  = kubernetes_namespace.observability[0].metadata[0].name
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger"
  version    = "3.4.1"
  wait       = true
  timeout    = 300

  set {
    name  = "allInOne.enabled"
    value = "true"
  }
  set {
    name  = "provisionDataStore.cassandra"
    value = "false"
  }
  set {
    name  = "storage.type"
    value = "memory"
  }
  set {
    name  = "agent.enabled"
    value = "false"
  }
  set {
    name  = "collector.enabled"
    value = "false"
  }
  set {
    name  = "query.enabled"
    value = "false"
  }
  # Expose Jaeger UI via NodePort for kind port mapping
  set {
    name  = "query.service.type"
    value = "NodePort"
  }
  set {
    name  = "query.service.nodePort"
    value = "30686"
  }
}

# --- OpenTelemetry Collector ---

resource "helm_release" "otel_collector" {
  count      = var.enable_observability ? 1 : 0
  name       = "otel-collector"
  namespace  = kubernetes_namespace.observability[0].metadata[0].name
  repository = "https://open-telemetry.github.io/opentelemetry-helm-charts"
  chart      = "opentelemetry-collector"
  version    = "0.108.0"
  wait       = true
  timeout    = 300

  values = [yamlencode({
    mode  = "deployment"
    image = { repository = "otel/opentelemetry-collector-contrib" }
    ports = {
      otlp      = { enabled = true, containerPort = 4317, servicePort = 4317, protocol = "TCP" }
      otlp-http = { enabled = true, containerPort = 4318, servicePort = 4318, protocol = "TCP" }
      metrics   = { enabled = true, containerPort = 8889, servicePort = 8889, protocol = "TCP" }
    }
    config = {
      receivers = {
        otlp = {
          protocols = {
            grpc = { endpoint = "0.0.0.0:4317" }
            http = { endpoint = "0.0.0.0:4318" }
          }
        }
      }
      processors = {
        batch = { timeout = "5s", send_batch_size = 1024 }
      }
      exporters = {
        "otlp/jaeger" = {
          endpoint = "jaeger-collector.observability.svc.cluster.local:4317"
          tls      = { insecure = true }
        }
        prometheus = {
          endpoint                         = "0.0.0.0:8889"
          metric_expiration                = "30m"
          resource_to_telemetry_conversion = { enabled = true }
        }
      }
      service = {
        pipelines = {
          traces  = { receivers = ["otlp"], processors = ["batch"], exporters = ["otlp/jaeger"] }
          metrics = { receivers = ["otlp"], processors = ["batch"], exporters = ["prometheus"] }
        }
      }
    }
  })]

  depends_on = [helm_release.jaeger]
}

# --- kube-prometheus-stack (Prometheus + Grafana) ---

resource "helm_release" "kube_prometheus" {
  count      = var.enable_observability ? 1 : 0
  name       = "kube-prometheus"
  namespace  = kubernetes_namespace.observability[0].metadata[0].name
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "67.9.0"
  wait       = true
  timeout    = 600

  values = [yamlencode({
    # Disable components not needed for agentloom dev
    kubeStateMetrics      = { enabled = false }
    nodeExporter          = { enabled = false }
    alertmanager          = { enabled = false }
    kubeControllerManager = { enabled = false }
    kubeScheduler         = { enabled = false }
    kubeProxy             = { enabled = false }
    kubeEtcd              = { enabled = false }

    prometheus = {
      prometheusSpec = {
        serviceMonitorSelectorNilUsesHelmValues = false
        # Scrape OTel Collector metrics
        additionalScrapeConfigs = [
          {
            job_name        = "otel-collector"
            scrape_interval = "5s"
            static_configs  = [{ targets = ["otel-collector-opentelemetry-collector.observability.svc.cluster.local:8889"] }]
          }
        ]
      }
      service = {
        type     = "NodePort"
        nodePort = 30090
      }
    }

    grafana = {
      service = {
        type     = "NodePort"
        nodePort = 30000
      }
      adminPassword = "admin"
      additionalDataSources = [
        {
          name   = "Jaeger"
          type   = "jaeger"
          uid    = "jaeger"
          access = "proxy"
          url    = "http://jaeger-query.observability.svc.cluster.local:16686"
        }
      ]
      dashboardProviders = {
        "dashboardproviders.yaml" = {
          apiVersion = 1
          providers = [{
            name            = "agentloom"
            type            = "file"
            disableDeletion = false
            editable        = true
            options         = { path = "/var/lib/grafana/dashboards/agentloom" }
          }]
        }
      }
      dashboardsConfigMaps = {
        agentloom = "agentloom-grafana-dashboard"
      }
    }
  })]

  depends_on = [helm_release.otel_collector]
}

# --- Grafana dashboard ConfigMap ---

resource "kubernetes_config_map" "grafana_dashboard" {
  count = var.enable_observability ? 1 : 0

  metadata {
    name      = "agentloom-grafana-dashboard"
    namespace = kubernetes_namespace.observability[0].metadata[0].name
    labels = {
      grafana_dashboard = "1"
    }
  }

  data = {
    "agentloom.json" = file("${path.module}/../grafana/dashboards/agentloom.json")
  }

  depends_on = [kubernetes_namespace.observability]
}

# -----------------------------------------------------------
# AgentLoom Helm release
# -----------------------------------------------------------

resource "helm_release" "agentloom" {
  name      = "agentloom"
  namespace = kubernetes_namespace.agentloom.metadata[0].name
  chart     = "${path.module}/../helm/agentloom"
  wait      = true
  timeout   = 300

  set {
    name  = "image.tag"
    value = var.agentloom_image_tag
  }

  set {
    name  = "observability.enabled"
    value = tostring(var.enable_observability)
  }

  set {
    name  = "observability.otelEndpoint"
    value = var.enable_observability ? "http://otel-collector-opentelemetry-collector.observability.svc.cluster.local:4317" : "http://localhost:4317"
  }

  set {
    name  = "ollama.enabled"
    value = tostring(var.enable_ollama)
  }

  set {
    name  = "ollama.model"
    value = var.ollama_model
  }

  set {
    name  = "workflow.definition"
    value = var.workflow_definition
  }

  dynamic "set" {
    for_each = length(var.provider_api_keys) > 0 ? [1] : []
    content {
      name  = "provider.existingSecret"
      value = "agentloom-provider-keys"
    }
  }

  depends_on = [
    kubernetes_namespace.agentloom,
    kubernetes_secret.provider_keys,
    helm_release.otel_collector,
  ]
}
