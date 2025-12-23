# Multi-stage build for mcp-markdown-ragdocs

# Stage 1: Base image with uv
FROM python:3.13-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Stage 2: Dependencies
FROM base AS dependencies

# Copy dependency files
COPY pyproject.toml /app/

# Install dependencies
RUN uv sync --no-dev --frozen

# Stage 3: Application
FROM base AS application

# Copy dependencies from previous stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy application source
COPY src /app/src
COPY pyproject.toml /app/

# Create directory for index data
RUN mkdir -p /app/.index_data

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command to run server
CMD ["python", "-m", "src.cli", "run", "--host", "0.0.0.0", "--port", "8000"]
