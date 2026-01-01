# Integration Guide

This document describes how to integrate mcp-markdown-ragdocs with MCP clients including VS Code, Claude Desktop, and other compatible applications.

## Transport Methods

This server supports two transport methods:

1. **Stdio Transport (Recommended)**: Used by VS Code, Claude Desktop, and most MCP clients. Server communicates via stdin/stdout.
2. **HTTP Transport**: REST API for development, testing, or custom integrations.

| Transport | Command | Use Case |
|-----------|---------|----------|
| Stdio | `mcp-markdown-ragdocs mcp` | VS Code, Claude Desktop, MCP client integrations |
| HTTP | `mcp-markdown-ragdocs run` | Development, testing, direct HTTP API access |

## MCP Protocol Overview

The Model Context Protocol (MCP) enables LLMs to access external tools and data sources. This server exposes a `query_documents` tool that allows AI assistants to search and retrieve information from local Markdown documentation.

## Integration with VS Code

### Prerequisites

- VS Code with MCP extension support
- Python 3.13+
- mcp-markdown-ragdocs installed via `uv`

### Configuration

The server uses **stdio transport**, not HTTP. VS Code manages the server lifecycle automatically.

**File:** `.vscode/settings.json` (workspace) or `~/.config/Code/User/mcp.json` (user-global)

```json
{
  "mcpServers": {
    "markdown-docs": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-markdown-ragdocs",
        "run",
        "mcp-markdown-ragdocs",
        "mcp"
      ],
      "type": "stdio"
    }
  }
}
```

**Path Resolution:**

Replace `/absolute/path/to/mcp-markdown-ragdocs` with the actual installation directory:

```zsh
cd /path/to/mcp-markdown-ragdocs
pwd  # Copy this path
```

**For installed package (not recommended for development):**

```json
{
  "mcpServers": {
    "markdown-docs": {
      "command": "mcp-markdown-ragdocs",
      "args": ["mcp"],
      "type": "stdio"
    }
  }
}
```

### Multi-Project Configuration

To specify which project to activate:

```json
{
  "mcpServers": {
    "markdown-docs": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-markdown-ragdocs",
        "run",
        "mcp-markdown-ragdocs",
        "mcp",
        "--project",
        "my-project"
      ],
      "type": "stdio"
    }
  }
}
```

The `--project` flag accepts:
- Project name from global config (e.g., `"my-project"`)
- Absolute path to project root (e.g., `"/home/user/project"`)

### Verification

After configuration:

1. Restart VS Code or reload window
2. Open MCP extension panel
3. Verify `markdown-docs` server appears as connected
4. Check available tools list includes `query_documents`

### Usage in Copilot Chat

Query your documentation from Copilot Chat:

```
@markdown-docs How do I configure authentication?
```

Or let Copilot automatically invoke the tool based on context:

```
I need to implement OAuth2 authentication according to our project docs
```

The tool returns ranked document chunks. Use the file paths and sections to locate and read the full documentation.

## Integration with Claude Desktop

### Prerequisites

- Claude Desktop app with MCP support
- Python 3.13+
- mcp-markdown-ragdocs installed

### Configuration

Claude Desktop uses **stdio transport** for MCP servers.

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
**File:** `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
**File:** `~/.config/Claude/claude_desktop_config.json` (Linux)

```json
{
  "mcpServers": {
    "markdown-docs": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-markdown-ragdocs",
        "run",
        "mcp-markdown-ragdocs",
        "mcp"
      ]
    }
  }
}
```

**With project override:**

```json
{
  "mcpServers": {
    "markdown-docs": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-markdown-ragdocs",
        "run",
        "mcp-markdown-ragdocs",
        "mcp",
        "--project",
        "my-project"
      ]
    }
  }
}
```

### Verification

1. Close and reopen Claude Desktop
2. Start a new conversation
3. Type a documentation-related question
4. Claude will automatically invoke `query_documents` when appropriate

### Usage

Claude will implicitly use the tool based on conversation context:

```
What does the authentication guide say about OAuth tokens?
```

The tool returns ranked document chunks. Review the results to identify relevant sections, then use file reading capabilities to access full context.

## HTTP API Integration (Alternative)

For custom integrations or testing, use the HTTP API.

### Start HTTP Server

```zsh
uv run mcp-markdown-ragdocs run --host 127.0.0.1 --port 8000
```

### Generic HTTP MCP Client

Configuration pattern for HTTP-based MCP clients:

```json
{
  "server_url": "http://127.0.0.1:8000",
  "tools": [
    {
      "endpoint": "/query_documents",
      "method": "POST",
      "name": "query_documents",
      "parameters": {
        "query": "string",
        "top_n": "integer"
      }
    }
  ]
}
```

### Direct API Testing

Test the endpoint with curl:

```zsh
curl -X POST http://127.0.0.1:8000/query_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I configure the search weights?"}'
```

Response:

```json
{
  "results": [
    {
      "chunk_id": "configuration_search_0",
      "content": "To configure search weights, edit the [search] section in config.toml...",
      "file_path": "docs/configuration.md",
      "header_path": ["Configuration Sections", "search"],
      "score": 0.92
    }
  ]
}
```

**Note:** HTTP endpoint uses full JSON format. MCP stdio transport uses compact text format: `[N] file § section (score)\ncontent`.
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

## Response Format Differences

### HTTP vs MCP Stdio

The server uses different response formats for HTTP and MCP stdio transport:

**HTTP JSON (Full Format):**
```json
{
  "results": [
    {
      "chunk_id": "auth_0",
      "content": "Authentication uses JWT tokens...",
      "file_path": "docs/auth.md",
      "header_path": ["Authentication", "Tokens"],
      "score": 0.95
    }
  ]
}
```

**MCP Stdio (Compact Format, 60% less overhead):**
```
[1] docs/auth.md § Authentication > Tokens (0.95)
Authentication uses JWT tokens...
```

### Query-Aware Truncation

Factual queries (configuration, commands, syntax) return truncated content to reduce token usage:

**Detection Patterns:**
- camelCase or snake_case identifiers
- Backticks or code-like terms
- Keywords: "what is", "configure", "syntax", "command"

**Example:**

Query: `getUserById function`

Response:
```
[1] api.md § Functions (0.92)
getUserById(id: string): User | null

Retrieves user by ID. Returns null if not found...
```

Conceptual queries (explanations, guides) return full content:

Query: `why use JWT authentication?`

Response includes complete section content.

### Compression Statistics

Default compression enabled (`min_confidence=0.3`, `max_chunks_per_doc=2`, `dedup_enabled=true`).

Set `show_stats=true` to see filtering results:

```
Compression Stats:
- Original results: 47
- After score filter (≥0.3): 23
- After deduplication: 12
- After document limit (2 per doc): 8
- Clusters merged: 11
```

**Token Savings:** Compression reduces context usage by 40-60% depending on query and corpus.

## Troubleshooting

### Problem: VS Code Cannot Start MCP Server

**Symptoms:**
- Server not appearing in MCP extension
- "Failed to start server" error
- Tool not available

**Solutions:**

1. **Verify uv installation:**
   ```zsh
   which uv
   uv --version
   ```

2. **Check absolute path in config:**
   ```zsh
   cd /path/to/mcp-markdown-ragdocs
   pwd  # Use this exact path in config
   ```

3. **Test server manually:**
   ```zsh
   cd /path/to/mcp-markdown-ragdocs
   uv run mcp-markdown-ragdocs mcp
   ```
   Server should start without errors. Press Ctrl+C to stop.

4. **Check VS Code logs:**
   - Open Output panel
   - Select "MCP" from dropdown
   - Look for startup errors

5. **Verify file permissions:**
   ```zsh
   ls -la /path/to/mcp-markdown-ragdocs
   ```

### Problem: Query Returns Empty Results

**Symptoms:**
- `query_documents` returns no answer or "No results found"

**Solutions:**

1. **Check indices exist:**
   ```zsh
   ls ~/.local/share/mcp-markdown-ragdocs/  # Multi-project mode
   ls .index_data/  # Local mode
   ```

2. **Verify project detection:**
   ```zsh
   cd /your/project
   uv run mcp-markdown-ragdocs check-config
   ```
   Should show detected project and document count.

3. **Rebuild index manually:**
   ```zsh
   cd /your/project
   uv run mcp-markdown-ragdocs rebuild-index
   ```

4. **Check document paths in config:**
   ```toml
   [indexing]
   documents_path = "/correct/path/to/docs"
   ```

### Problem: Wrong Project Detected

**Symptoms:**
- Queries search incorrect project
- Multi-project setup not working

**Solutions:**

1. **Explicitly specify project:**
   ```json
   {
     "args": [
       "mcp",
       "--project",
       "correct-project-name"
     ]
   }
   ```

2. **Verify project configuration:**
   ```zsh
   cat ~/.config/mcp-markdown-ragdocs/config.toml
   ```
   Check `[[projects]]` paths are absolute and correct.

3. **Test detection:**
   ```zsh
   cd /project/directory
   uv run mcp-markdown-ragdocs check-config
   ```

### Problem: Claude Desktop Not Finding Tool

**Symptoms:**
- Claude does not use documentation tool
- No indication tool is available

**Solutions:**

1. **Verify config file location:**
   ```zsh
   # macOS
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

   # Linux
   cat ~/.config/Claude/claude_desktop_config.json
   ```

2. **Check JSON syntax:**
   Use `jq` to validate:
   ```zsh
   cat path/to/config.json | jq
   ```

3. **Restart Claude Desktop completely:**
   - Quit application (not just close window)
   - Reopen

4. **Test server manually:**
   ```zsh
   uv run mcp-markdown-ragdocs mcp
   ```

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
  "query": "How do I configure the server?",
  "top_n": 5
}
```

**Response:**

```json
{
  "results": [
    {
      "chunk_id": "configuration_0",
      "content": "To configure the server, create a config.toml file...",
      "file_path": "docs/configuration.md",
      "header_path": ["Configuration Reference"],
      "score": 0.95
    }
  ]
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
