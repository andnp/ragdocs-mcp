#!/usr/bin/env bash
# Development server with auto-reload

set -e

# Use development config if it exists
if [ -f "config.dev.toml" ]; then
    export CONFIG_PATH="config.dev.toml"
    echo "Using development config: config.dev.toml"
fi

# Enable debug logging
export RUST_LOG=debug
export PYTHONUNBUFFERED=1

# Start server with uvicorn reload
echo "Starting development server with auto-reload..."
uv run uvicorn src.server:create_app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --factory \
    --log-level info
