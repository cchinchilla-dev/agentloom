variable "cluster_name" {
  description = "Name of the kind cluster"
  type        = string
  default     = "agentloom-dev"

  validation {
    condition     = length(var.cluster_name) > 0 && can(regex("^[a-z0-9][a-z0-9-]*$", var.cluster_name))
    error_message = "cluster_name must be non-empty and contain only lowercase alphanumeric characters and hyphens."
  }
}

variable "agentloom_image_tag" {
  description = "Docker image tag for agentloom"
  type        = string
  default     = "latest"

  validation {
    condition     = length(var.agentloom_image_tag) > 0
    error_message = "agentloom_image_tag must not be empty."
  }
}

variable "enable_observability" {
  description = "Deploy the full observability stack (OTel Collector, Prometheus, Grafana, Jaeger) and configure agentloom pods to send telemetry. Set to false for a lightweight setup without metrics/traces."
  type        = bool
  default     = true
}

variable "provider_api_keys" {
  description = "Map of provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)"
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "workflow_definition" {
  description = "Inline workflow YAML definition"
  type        = string
  default     = <<-EOT
    name: default
    version: "1.0"
    config:
      provider: ollama
      model: phi4
    state:
      question: "What is Python in one sentence?"
    steps:
      - id: hello
        type: llm_call
        prompt: "Answer concisely: {state.question}"
        output: answer
  EOT

  validation {
    condition     = length(trimspace(var.workflow_definition)) > 0
    error_message = "workflow_definition must not be empty."
  }
}
