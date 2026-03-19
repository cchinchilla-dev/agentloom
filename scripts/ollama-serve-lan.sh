#!/usr/bin/env bash
# Start Ollama listening on all interfaces so other machines on the LAN can use it.
# Useful when you want a more powerful machine to serve models for lighter devices.
#
# Usage:
#   ./scripts/ollama-serve-lan.sh
#
# Then on another machine:
#   export OLLAMA_BASE_URL=http://<this-machine-ip>:11434
#   agentforge run workflow.yaml --provider ollama --model qwen3:8b

set -e

LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')

echo "Starting Ollama on all interfaces (0.0.0.0:11434)"
echo "LAN IP: ${LAN_IP:-unknown}"
echo ""
echo "Other machines can connect with:"
echo "  export OLLAMA_BASE_URL=http://${LAN_IP}:11434"
echo ""

OLLAMA_HOST=0.0.0.0:11434 exec ollama serve
