output "cluster_name" {
  description = "Name of the kind cluster"
  value       = kind_cluster.this.name
}

output "kubeconfig" {
  description = "Path to the generated kubeconfig"
  value       = kind_cluster.this.kubeconfig_path
  sensitive   = true
}

output "grafana_url" {
  description = "Grafana URL (available when enable_observability = true)"
  value       = var.enable_observability ? "http://localhost:3000 (admin/admin)" : null
}

output "prometheus_url" {
  description = "Prometheus URL (available when enable_observability = true)"
  value       = var.enable_observability ? "http://localhost:9090" : null
}

output "jaeger_url" {
  description = "Jaeger URL (available when enable_observability = true)"
  value       = var.enable_observability ? "http://localhost:16686" : null
}
