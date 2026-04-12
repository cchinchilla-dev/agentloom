# Providers

AgentLoom ships with four providers. The gateway routes requests based on model name and falls back automatically when a provider is unavailable.

## Capability matrix

| Capability | OpenAI | Anthropic | Google | Ollama |
|---|---|---|---|---|
| Models | `gpt-*`, `o3*`, `o4*` | `claude*` | `gemini*` | Any local model |
| Streaming | SSE | SSE | SSE | NDJSON |
| Image input | :material-check: | :material-check: | :material-check: | :material-check: |
| PDF input | :material-close: | :material-check: | :material-check: | :material-close: |
| Audio input | :material-check: | :material-close: | :material-check: | :material-close: |
| Cost tracking | :material-check: | :material-check: | :material-check: | Free (local) |

## Configuration

Switch provider in any workflow:

```yaml
config:
  provider: google
  model: gemini-2.5-flash
```

Or override at runtime via CLI:

```bash
agentloom run workflow.yaml --provider anthropic --model claude-sonnet-4-20250514
```

## Environment variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google |
| `OLLAMA_BASE_URL` | Ollama (default: `http://localhost:11434`) |

## Circuit breaker

The gateway wraps each provider with a circuit breaker:

| State | Behavior | Transition |
|-------|----------|------------|
| **Closed** | Requests pass through normally | :material-arrow-right: Open after 5 consecutive failures |
| **Open** | Requests rejected immediately, fallback provider used | :material-arrow-right: Half-open after 60s |
| **Half-open** | One test request allowed | :material-arrow-right: Closed on success, Open on failure |

## Rate limiter

Dual token-bucket rate limiting per provider:

- **Requests per minute** — default 60 RPM
- **Tokens per minute** — default 100,000 TPM

```python
gateway.register(
    provider,
    max_rpm=120,          # requests/minute
    max_tpm=200_000,      # tokens/minute
)
```

## Fallback chain

Providers are tried in priority order. Register multiple providers for automatic fallback:

```python
gateway.register(openai_provider, priority=0)
gateway.register(anthropic_provider, priority=1, is_fallback=True)
gateway.register(ollama_provider, priority=2, is_fallback=True)
```

If OpenAI fails or its circuit breaker trips, the gateway automatically routes to Anthropic. If Anthropic also fails, it falls back to Ollama.

## Multi-modal attachments

LLM steps support image, PDF, and audio attachments:

```yaml
steps:
  - id: analyze
    type: llm_call
    prompt: "Describe what you see in this image."
    attachments:
      - type: image
        source: "{state.image_url}"
        fetch: local
    output: description
```

| Field | Description |
|-------|-------------|
| `type` | `image`, `pdf`, or `audio` |
| `source` | URL, local file path, or base64 data |
| `media_type` | Optional; inferred from type if omitted |
| `fetch` | `local` (engine downloads) or `provider` (provider fetches URL directly) |

!!! warning "Provider support varies"
    Check the [capability matrix](#capability-matrix) above. Sending a PDF to OpenAI or audio to Anthropic will raise a `ProviderError`.

## Security

### SSRF protection

URL-based attachments (`fetch: local`) are protected against Server-Side Request Forgery. The engine blocks requests to private and reserved IP ranges (RFC 1918, loopback, link-local) before any network call is made.

### Attachment size limit

All attachments are limited to **20 MB** per file. Larger files are rejected before being sent to the provider.
