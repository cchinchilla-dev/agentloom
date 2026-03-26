# ============================================================
# AgentLoom — Multi-stage Dockerfile
# ============================================================
# Stages:
#   builder    — install deps + build wheel
#   dev        — full dev environment (--target dev)
#   production — minimal runtime image (default)
#
# Usage:
#   docker build -t agentloom .
#   docker build --build-arg BUILD_OBSERVABILITY=true -t agentloom:obs .
#   docker build --target dev -t agentloom:dev .
# ============================================================

# --------------- Stage 1: builder ---------------
FROM python:3.12-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

WORKDIR /build

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen = exact lockfile versions)
RUN uv sync --frozen --no-install-project --no-dev

# Copy source and build wheel
COPY src/ src/
COPY README.md ./
RUN uv build --wheel --out-dir /build/dist

# --------------- Stage 2: dev (--target dev) ---------------
FROM builder AS dev

# Install dev dependencies + all extras (observability)
RUN uv sync --frozen --group dev --all-extras

COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/

WORKDIR /build
ENTRYPOINT ["uv", "run"]
CMD ["pytest"]

# --------------- Stage 3: production (default) ---------------
FROM python:3.12-slim AS production

ARG BUILD_OBSERVABILITY=false

# Non-root user
RUN groupadd --gid 1000 agentloom \
    && useradd --uid 1000 --gid agentloom --create-home agentloom

# Install the wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/agentloom-*.whl) \
    && if [ "$BUILD_OBSERVABILITY" = "true" ]; then \
         pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
           "${WHEEL}[observability]"; \
       else \
         pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
           "$WHEEL"; \
       fi \
    && rm -f /tmp/*.whl

# Copy example workflows so validate works out of the box
COPY examples/ /workflows/

WORKDIR /workflows
USER agentloom

# Default OTel endpoint for containerized environments
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

ENTRYPOINT ["agentloom"]
CMD ["--help"]
