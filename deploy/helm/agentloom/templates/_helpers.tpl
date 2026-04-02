{{/*
Expand the name of the chart.
*/}}
{{- define "agentloom.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "agentloom.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "agentloom.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "agentloom.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "agentloom.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentloom.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "agentloom.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "agentloom.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image reference
*/}}
{{- define "agentloom.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end }}

{{/*
Shared pod spec for Job and CronJob templates.
Avoids duplication and ensures both stay in sync.
*/}}
{{- define "agentloom.podSpec" -}}
restartPolicy: {{ .Values.job.restartPolicy }}
serviceAccountName: {{ include "agentloom.serviceAccountName" . }}
automountServiceAccountToken: {{ .Values.serviceAccount.automountServiceAccountToken | default false }}
{{- with .Values.securityContext }}
securityContext:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
containers:
  - name: agentloom
    image: {{ include "agentloom.image" . }}
    imagePullPolicy: {{ .Values.image.pullPolicy }}
    {{- with .Values.containerSecurityContext }}
    securityContext:
      {{- toYaml . | nindent 6 }}
    {{- end }}
    args:
      - run
      - /workflows/workflow.yaml
      {{- range .Values.workflow.args }}
      - {{ . | quote }}
      {{- end }}
    {{- if or .Values.provider.existingSecret .Values.provider.openaiApiKey .Values.provider.anthropicApiKey .Values.provider.googleApiKey }}
    envFrom:
      - secretRef:
          name: {{ .Values.provider.existingSecret | default (printf "%s-provider-keys" (include "agentloom.fullname" .)) }}
    {{- end }}
    {{- if or .Values.observability.enabled .Values.ollama.enabled }}
    env:
      {{- if .Values.observability.enabled }}
      - name: OTEL_EXPORTER_OTLP_ENDPOINT
        value: {{ .Values.observability.otelEndpoint | quote }}
      {{- end }}
      {{- if .Values.ollama.enabled }}
      - name: OLLAMA_BASE_URL
        value: "http://{{ include "agentloom.fullname" . }}-ollama.{{ .Release.Namespace }}.svc.cluster.local:11434"
      {{- end }}
    {{- end }}
    resources:
      {{- toYaml .Values.resources | nindent 6 }}
    volumeMounts:
      - name: workflows
        mountPath: /workflows
        readOnly: true
      - name: tmp
        mountPath: /tmp
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
volumes:
  - name: workflows
    configMap:
      name: {{ .Values.workflow.existingConfigMap | default (printf "%s-workflow" (include "agentloom.fullname" .)) }}
  - name: tmp
    emptyDir: {}
{{- end }}
