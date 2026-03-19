#!/usr/bin/env bash
# Install Ollama and pull recommended models for AgentForge development.
# Works on macOS and Linux.

set -e

echo "=== Ollama Setup for AgentForge ==="

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo "Ollama already installed: $(ollama --version)"
fi

# Start Ollama server if not running
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Pull recommended models
# Small and fast — good for testing and development
MODELS=(
    "qwen3:8b"         # Best small general-purpose model (March 2026)
    "llama3.3:8b"      # Strong all-rounder
    "phi4"             # Best reasoning per GB of RAM
    "deepseek-r1:8b"   # Chain-of-thought reasoning
)

echo ""
echo "Pulling recommended models..."
for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Pulling $model ---"
    ollama pull "$model"
done

echo ""
echo "=== Done ==="
echo ""
echo "Models available:"
ollama list
echo ""
echo "Test with AgentForge:"
echo "  agentforge run examples/01_simple_qa.yaml --provider ollama --model qwen3:8b"
