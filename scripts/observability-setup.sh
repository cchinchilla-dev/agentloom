#!/usr/bin/env bash
# Install observability dependencies and start the monitoring stack.
# Works with Docker Desktop and Colima. Auto-installs missing components.
#
# Usage:
#   ./scripts/observability-setup.sh        # start everything
#   ./scripts/observability-setup.sh down    # stop the stack

set -e

DEPLOY_DIR="$(cd "$(dirname "$0")/../deploy" && pwd)"

# Global: filled by _ensure_compose
DC=""

# ---------------------------------------------------------------------------
# Detect or install docker-compose
# ---------------------------------------------------------------------------
_detect_compose() {
    if docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose &> /dev/null 2>&1; then
        echo "docker-compose"
    else
        echo ""
    fi
}

_ensure_compose() {
    DC="$(_detect_compose)"
    if [[ -n "$DC" ]]; then
        return
    fi

    echo "docker-compose not found — installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "ERROR: Homebrew not found. Install docker-compose manually:"
        echo "  brew install docker-compose"
        exit 1
    fi

    brew install docker-compose

    # Link as Docker CLI plugin so 'docker compose' works too
    mkdir -p ~/.docker/cli-plugins
    ln -sfn "$(brew --prefix)/opt/docker-compose/bin/docker-compose" \
        ~/.docker/cli-plugins/docker-compose

    DC="$(_detect_compose)"
    if [[ -z "$DC" ]]; then
        echo "ERROR: docker-compose installed but still not detected."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "down" ]]; then
    _ensure_compose
    echo "Stopping observability stack..."
    $DC -f "$DEPLOY_DIR/docker-compose.yml" down
    echo "Done."
    exit 0
fi

echo "=== AgentLoom Observability Setup ==="
echo ""

# ---------------------------------------------------------------------------
# Check Docker engine (install if missing on macOS)
# ---------------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    echo "Docker not found."
    if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
        echo "Installing Colima + Docker CLI via Homebrew..."
        brew install colima docker
    else
        echo "ERROR: Install Docker manually:"
        echo "  macOS: brew install colima docker"
        echo "  Linux: https://docs.docker.com/engine/install/"
        exit 1
    fi
fi

if ! docker info &> /dev/null 2>&1; then
    if command -v colima &> /dev/null; then
        echo "Docker daemon not running — starting Colima..."
        colima start
    else
        echo "ERROR: Docker daemon is not running."
        echo "  Docker Desktop: open the app"
        echo "  Colima:         colima start"
        exit 1
    fi
fi

echo "Docker: $(docker --version)"

# ---------------------------------------------------------------------------
# Ensure docker-compose (install if missing)
# ---------------------------------------------------------------------------
_ensure_compose
echo "Compose: $DC"
echo ""

# ---------------------------------------------------------------------------
# Install Python observability extras
# ---------------------------------------------------------------------------
echo "Installing observability Python packages..."
if command -v uv &> /dev/null; then
    uv sync --group dev --all-extras
else
    pip install -e ".[observability]"
fi

# ---------------------------------------------------------------------------
# Start the stack
# ---------------------------------------------------------------------------
echo ""
echo "Starting observability stack..."
$DC -f "$DEPLOY_DIR/docker-compose.yml" up -d

# Wait for services
echo ""
echo "Waiting for services..."
for i in {1..15}; do
    if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

echo ""
echo "=== Ready ==="
echo ""
echo "  Grafana:      http://localhost:3000  (admin/admin)"
echo "  Prometheus:    http://localhost:9090"
echo "  Jaeger:        http://localhost:16686"
echo "  OTel gRPC:     localhost:4317"
echo ""
echo "Run a workflow with tracing:"
echo "  uv run agentloom run examples/01_simple_qa.yaml --provider ollama --model phi4"
echo ""
echo "Stop with:"
echo "  ./scripts/observability-setup.sh down"
