# Integration Guide

This document describes how to integrate mcp-markdown-ragdocs with MCP clients including VS Code, Claude Desktop, and other compatible applications.

## MCP Protocol Overview

The Model Context Protocol (MCP) enables LLMs to access external tools and data sources. This server exposes a `query_documents` tool that allows AI assistants to search and retrieve information from local Markdown documentation.

## Integration with VS Code

### Prerequisites

- VS Code with MCP extension installed
- Python 3.13+
- mcp-markdown-ragdocs installed via `uv`

### Step 1: Start the Server

Start the server in a terminal:

```zsh
cd /path/to/your/documentation
uv run mcp-markdown-ragdocs run
```

The server will index your documentation and start listening on `http://127.0.0.1:8000`.

### Step 2: Configure VS Code MCP Settings

Open VS Code settings (JSON format) and add the MCP server configuration:

**File:** `.vscode/settings.json` or User Settings

```json
{
  "mcp.servers": {
    "markdown-docs": {
      "type": "http",
      "url": "http://127.0.0.1:8000",
      "tools": [
        {
          "name": "query_documents",
          "description": "Search project documentation using hybrid search (semantic, keyword, graph traversal)",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "Natural language query or question about the documentation"
              }
            },
            "required": ["query"]
          }
        }
      ]
    }
  }
}
```

### Step 3: Verify Connection

Use the MCP extension UI or command palette to verify the connection:

1. Open Command Palette (`Cmd/Ctrl+Shift+P`)
2. Run "MCP: List Available Tools"
3. Verify `query_documents` appears in the list

### Step 4: Query from Copilot Chat

In a GitHub Copilot Chat session, reference the tool:

```
@mcp query_documents How do I configure authentication in the API?
```

The assistant will call the `query_documents` tool and incorporate the results into its response.

## Integration with Claude Desktop

### Prerequisites

- Claude Desktop app with MCP support
- Python 3.13+
- mcp-markdown-ragdocs installed

### Step 1: Start the Server

```zsh
uv run mcp-markdown-ragdocs run --host 127.0.0.1 --port 8000
```

### Step 2: Configure Claude Desktop

Edit the Claude Desktop MCP configuration file:

**File:** `~/Library/Application Support/Claude/mcp_config.json` (macOS)
**File:** `%APPDATA%\Claude\mcp_config.json` (Windows)
**File:** `~/.config/Claude/mcp_config.json` (Linux)

```json
{
  "mcpServers": {
    "markdown-ragdocs": {
      "url": "http://127.0.0.1:8000",
      "tools": {
        "query_documents": {
          "description": "Search local Markdown documentation using hybrid search"
        }
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop for the configuration to take effect.

### Step 4: Query from Chat

In a Claude chat, use the tool implicitly:

```
What does the authentication guide say about OAuth tokens?
```

Claude will automatically decide when to use the `query_documents` tool based on context.

## Integration with Other MCP Clients

### Generic HTTP MCP Client

Most MCP clients support HTTP-based tool servers. Configuration pattern:

```json
{
  "server_url": "http://127.0.0.1:8000",
  "tools": [
    {
      "endpoint": "/query_documents",
      "method": "POST",
      "name": "query_documents",
      "parameters": {
        "query": "string"
      }
    }
  ]
}
```

### API Testing with curl

Test the endpoint directly:

```zsh
curl -X POST http://127.0.0.1:8000/query_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I configure the search weights?"}'
```

Response:

```json
{
  "answer": "To configure search weights, edit the [search] section in config.toml..."
}
```

## Configuration for Different Use Cases

### Use Case 1: Project-Local Documentation

Run the server from your project root with project-local configuration:

**File:** `./config.toml`

```toml
[indexing]
documents_path = "./docs"
index_path = ".ragdocs-index/"
```

**MCP Configuration:**

```json
{
  "url": "http://127.0.0.1:8000"
}
```

### Use Case 2: Shared Team Documentation

Run the server as a shared service on a team server:

**File:** `/opt/ragdocs/config.toml`

```toml
[server]
host = "0.0.0.0"
port = 8080

[indexing]
documents_path = "/opt/team-docs"
index_path = "/var/lib/ragdocs-indices"
```

**MCP Configuration (each team member):**

```json
{
  "url": "http://docs-server.local:8080"
}
```

### Use Case 3: Personal Knowledge Base (Obsidian)

Run the server pointing at your Obsidian vault:

**File:** `~/.config/mcp-markdown-ragdocs/config.toml`

```toml
[indexing]
documents_path = "~/Documents/ObsidianVault"
index_path = "~/.cache/ragdocs-obsidian"

[search]
semantic_weight = 1.2
recency_bias = 0.7  # Prioritize recent notes
```

**MCP Configuration:**

```json
{
  "url": "http://127.0.0.1:8000"
}
```

## Server Management

### Running as a Background Service

#### systemd (Linux)

Create a systemd service file:

**File:** `/etc/systemd/system/mcp-ragdocs.service`

```ini
[Unit]
Description=MCP Markdown RAG Documentation Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/opt/ragdocs
ExecStart=/usr/local/bin/uv run mcp-markdown-ragdocs run
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```zsh
sudo systemctl enable mcp-ragdocs
sudo systemctl start mcp-ragdocs
```

#### Docker

Create a Dockerfile:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN uv sync

# Expose port
EXPOSE 8000

# Start server
CMD ["uv", "run", "mcp-markdown-ragdocs", "run", "--host", "0.0.0.0"]
```

Build and run:

```zsh
docker build -t mcp-ragdocs .
docker run -d -p 8000:8000 -v /path/to/docs:/docs \
  -e INDEX_PATH=/app/.index_data \
  mcp-ragdocs
```

### Monitoring Server Status

Check health:

```zsh
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

Check detailed status:

```zsh
curl http://127.0.0.1:8000/status
```

Response:

```json
{
  "server_status": "running",
  "indexing_service": {
    "pending_queue_size": 0,
    "last_sync_time": "2025-12-22T10:30:45.123Z",
    "failed_files": []
  },
  "indices": {
    "document_count": 147,
    "index_version": "1.0.0"
  }
}
```

## Troubleshooting

### Problem: MCP Client Cannot Connect

**Symptoms:**
- Connection refused errors
- Tool not appearing in client

**Solutions:**
1. Verify server is running: `curl http://127.0.0.1:8000/health`
2. Check server logs for startup errors
3. Verify port is not blocked by firewall
4. Ensure host/port match in client configuration

### Problem: Query Returns Empty Results

**Symptoms:**
- `query_documents` returns no answer or "No results found"

**Solutions:**
1. Check document count: `curl http://127.0.0.1:8000/status`
2. Verify documents_path points to correct directory
3. Rebuild index: `uv run mcp-markdown-ragdocs rebuild-index`
4. Check for indexing errors in server logs

### Problem: Server Uses High CPU

**Symptoms:**
- CPU usage spikes during query or indexing
- Slow response times

**Solutions:**
1. Reduce concurrent queries (serial querying)
2. Adjust search weights to disable expensive strategies
3. Check for file watcher loops (rapid file changes)
4. Consider smaller documents_path scope

### Problem: Index Rebuild Loops

**Symptoms:**
- Server rebuilds index on every startup
- Manifest file not persisted

**Solutions:**
1. Verify index_path is writable
2. Check for file permission errors in logs
3. Ensure index_path is not in `.gitignore` or deleted on restart
4. Verify manifest file exists: `ls {index_path}/index.manifest.json`

### Problem: File Changes Not Detected

**Symptoms:**
- Edited documents not reflected in search results
- File watcher not triggering updates

**Solutions:**
1. Check pending queue: `curl http://127.0.0.1:8000/status`
2. Verify watchdog is installed: `uv pip list | grep watchdog`
3. Check for failed files in status endpoint
4. Manually rebuild: `uv run mcp-markdown-ragdocs rebuild-index`

### Problem: Authentication Errors (MCP Client)

**Symptoms:**
- 401 Unauthorized or 403 Forbidden errors

**Note:** This server does not implement authentication. All requests are unauthenticated. If you see authentication errors, they are likely from:
1. MCP client misconfiguration (expecting auth headers)
2. Reverse proxy or firewall requiring authentication
3. Incorrect URL (pointing to different server)

**Solutions:**
1. Verify URL points directly to mcp-ragdocs server
2. Remove any authentication configuration from MCP client
3. Check reverse proxy logs if using one

## Security Considerations

### Network Exposure

By default, the server binds to `127.0.0.1` (localhost only). To expose on a network:

```toml
[server]
host = "0.0.0.0"  # Listen on all interfaces
```

**Warning:** This exposes the server to any client on your network. Consider:
- Running behind a reverse proxy with authentication
- Using firewall rules to restrict access
- Deploying in a trusted network only

### File System Access

The server has read access to all files under `documents_path`. Avoid pointing to:
- System directories (`/`, `/etc`, `/usr`)
- Home directory root (`$HOME` without subdirectory)
- Directories with sensitive files (credentials, keys)

**Recommended:** Scope to specific documentation directories.

### Resource Limits

No built-in resource limits. Consider:
- Using systemd resource limits (CPUQuota, MemoryLimit)
- Docker resource constraints (--memory, --cpus)
- Rate limiting via reverse proxy

## API Reference

### POST /query_documents

Query the documentation index.

**Request:**

```json
{
  "query": "How do I configure the server?"
}
```

**Response:**

```json
{
  "answer": "To configure the server, create a config.toml file..."
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

### GET /status

Detailed server status.

**Response:**

```json
{
  "server_status": "running",
  "indexing_service": {
    "pending_queue_size": 0,
    "last_sync_time": "2025-12-22T10:30:45.123Z",
    "failed_files": []
  },
  "indices": {
    "document_count": 147,
    "index_version": "1.0.0"
  }
}
```
