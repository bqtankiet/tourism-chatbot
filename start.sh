#!/bin/bash

set -e

echo "Starting Ollama..."
ollama serve &

# Wait until Ollama ready
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
  sleep 2
done

echo "Ollama is ready!"

# Pull model (chỉ pull nếu chưa có)
if ! ollama list | grep -q "qwen3:0.6b"; then
  echo "Pulling model..."
  ollama pull qwen3:0.6b
else
  echo "Model already exists"
fi

# Start app
echo "Starting chatbot..."
export OLLAMA_API_BASE=http://localhost:11434

exec python serve.py