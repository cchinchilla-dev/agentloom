# ============================================================
# AgentLoom — Multi-stage Dockerfile
# ============================================================
# Stages:
#   builder         — install deps + build wheel
#   dev             — full dev environment (--target dev)
#   production      — runtime image WITH observability (default)
#   production-lite — runtime image WITHOUT observability (~30 MB smaller)
#
# Usage:
#   docker build -t agentloom .                       # production (observability)
#   docker build --target production-lite -t al:lite .
#   docker build --target dev -t agentloom:dev .
#
# The default image ships the [observability] extra so a container that
# sets OTEL_EXPORTER_OTLP_ENDPOINT exports traces out of the box. Choose
# `--target production-lite` for the smaller image when OTel is unused.
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

# --------------- Stage 3: runtime-base ---------------
# Everything the two runtime images share EXCEPT the dependency install.
# Keeping it in one stage means `production` and `production-lite` differ
# by exactly one line — the pip install — so they cannot drift apart.
FROM python:3.12-slim AS runtime-base

# Non-root user
RUN groupadd --gid 1000 agentloom \
    && useradd --uid 1000 --gid agentloom --create-home agentloom

# Stage the wheel for the install step in the child stages
COPY --from=builder /build/dist/*.whl /tmp/

# Copy example workflows so validate works out of the box. Recordings
# ride alongside so the mock-provider examples (embeddings, conversation,
# structured output) resolve their ``responses_file: recordings/…`` paths
# against the container's WORKDIR without a runtime mount.
COPY examples/ /workflows/
COPY recordings/ /workflows/recordings/

WORKDIR /workflows

# Default OTel endpoint for containerized environments
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
# OLLAMA_BASE_URL intentionally unset — set at runtime or via compose.
# Code defaults to http://localhost:11434 when unset.

ENTRYPOINT ["agentloom"]
CMD ["--help"]

# --------------- Stage 4: production-lite (--target production-lite) ---------------
# Runtime WITHOUT the observability extra — smaller image for deployments
# that do not export OTel traces / Prometheus metrics.
FROM runtime-base AS production-lite

RUN WHEEL=$(ls /tmp/agentloom-*.whl) \
    && pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
       "$WHEEL" \
    && rm -f /tmp/*.whl

USER agentloom

# --------------- Stage 5: production (default) ---------------
# Runtime WITH the observability extra. Last stage in the file, so a
# bare `docker build .` produces this image.
FROM runtime-base AS production

RUN WHEEL=$(ls /tmp/agentloom-*.whl) \
    && pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
       "${WHEEL}[observability]" \
    && rm -f /tmp/*.whl

USER agentloom
