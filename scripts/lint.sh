#!/usr/bin/env bash
# Run linters and type checker

set -e

echo "==> Running ruff check..."
uv run ruff check .

echo ""
echo "==> Running ruff format check..."
uv run ruff format --check .

echo ""
echo "==> Running pyrefly..."
uv run pyrefly check

echo ""
echo "All checks passed!"
