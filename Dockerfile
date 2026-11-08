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
# The default ships the [observability] extra: setting OTEL_EXPORTER_OTLP_ENDPOINT exports traces.
# ============================================================

# --------------- Stage 1: builder ---------------
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

WORKDIR /build

# Lockfile before the source: the dependency layer stays cached across code changes
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY README.md ./
RUN uv build --wheel --out-dir /build/dist

# --------------- Stage 2: dev (--target dev) ---------------
FROM builder AS dev

RUN uv sync --frozen --group dev --all-extras

COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/

WORKDIR /build
ENTRYPOINT ["uv", "run"]
CMD ["pytest"]

# --------------- Stage 3: runtime-base ---------------
# Shared by both runtime images, which differ only in their install step and so cannot drift.
# Each install step removes pip afterwards: nothing runs it at runtime.
FROM python:3.14-slim AS runtime-base

RUN groupadd --gid 1000 agentloom \
    && useradd --uid 1000 --gid agentloom --create-home agentloom

# The child stages install the wheel, then delete it
COPY --from=builder /build/dist/*.whl /tmp/

# Bundled so validate runs out of the box; mock examples resolve recordings/ against WORKDIR
COPY examples/ /workflows/
COPY recordings/ /workflows/recordings/

WORKDIR /workflows

ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
# OLLAMA_BASE_URL stays unset: set it at run time, or the code falls back to localhost:11434

ENTRYPOINT ["agentloom"]
CMD ["--help"]

# --------------- Stage 4: production-lite (--target production-lite) ---------------
FROM runtime-base AS production-lite

RUN WHEEL=$(ls /tmp/agentloom-*.whl) \
    && pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
       "$WHEEL" \
    && rm -f /tmp/*.whl \
    && pip uninstall --yes --root-user-action=ignore --disable-pip-version-check pip

USER agentloom

# --------------- Stage 5: production (default) ---------------
# Last stage in the file, so a bare `docker build .` builds this one.
FROM runtime-base AS production

RUN WHEEL=$(ls /tmp/agentloom-*.whl) \
    && pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
       "${WHEEL}[observability]" \
    && rm -f /tmp/*.whl \
    && pip uninstall --yes --root-user-action=ignore --disable-pip-version-check pip

USER agentloom
