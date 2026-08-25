#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama service to start up
until curl -s http://127.0.0.1:11434/ > /dev/null; do
    sleep 1
done

# Pull the gemma2:2b model (old, but small and fast)
echo "Pulling gemma2:2b model..."
ollama pull gemma2:2b

# Start the main FastAPI application
echo "Starting FastAPI..."
exec uvicorn cortex:app --host 0.0.0.0 --port 80