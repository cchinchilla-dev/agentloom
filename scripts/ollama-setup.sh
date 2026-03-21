#!/usr/bin/env bash
# Install Ollama and pull recommended models for AgentLoom development.
# Works on macOS and Linux.

set -e

echo "=== Ollama Setup for AgentLoom ==="

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
    "phi4"             # Default model for examples — best reasoning per GB
    "llama3.1:8b"      # Strong all-rounder
    "qwen3:8b"         # Great multilingual model
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
echo "Test with AgentLoom:"
echo "  agentloom run examples/01_simple_qa.yaml --provider ollama --model qwen3:8b"
