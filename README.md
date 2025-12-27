# mcp-markdown-ragdocs

A Model Context Protocol server that provides semantic search over local Markdown documentation using hybrid retrieval.

## What it is

This is an MCP server that indexes local Markdown files and exposes a `query_documents` tool for retrieval-augmented generation. The server combines semantic search, keyword matching, and graph traversal to retrieve relevant document chunks.

## Why it exists

Technical documentation, personal notes, and project wikis are typically stored as Markdown files. Searching these collections manually or with grep is inefficient. This server provides a conversational interface to query documentation using natural language while automatically keeping the index synchronized with file changes.

Existing RAG solutions require manual database setup, explicit indexing steps, and ongoing maintenance. This server eliminates that friction with automatic file watching, zero-configuration defaults, and built-in index versioning.

## Features

- Hybrid search combining semantic embeddings (FAISS), keyword search (Whoosh), and graph traversal (NetworkX)
- Cross-encoder re-ranking for improved precision (optional, ~50ms latency)
- Query expansion via concept vocabulary for better recall
- **Multi-project support:** Manage isolated indices for multiple projects on one machine with automatic project detection
- Server-Sent Events (SSE) streaming for real-time response delivery
- CLI query command with rich formatted output
- Automatic file watching with debounced incremental indexing
- Zero-configuration operation with sensible defaults
- Index versioning with automatic rebuild on configuration changes
- Pluggable parser architecture (Markdown with tree-sitter)
- Rich Markdown parsing: frontmatter, wikilinks, tags, transclusions
- Reciprocal Rank Fusion for multi-strategy result merging
- Recency bias for recently modified documents
- Local-first architecture with no external dependencies

## Installation

Requires Python 3.13+.

```zsh
git clone https://github.com/yourusername/mcp-markdown-ragdocs.git
cd mcp-markdown-ragdocs
uv sync
```

## Quick Start

### For VS Code / MCP Clients (Recommended)

Start the stdio-based MCP server for use with VS Code or other MCP clients:

```zsh
uv run mcp-markdown-ragdocs mcp
```

The server will:
1. Scan for `*.md` files in the current directory
2. Build vector, keyword, and graph indices
3. Start file watching for automatic updates
4. Expose query_documents tool via stdio transport

See [MCP Integration](#mcp-integration) below for VS Code configuration.

### For HTTP API / Development

Start the HTTP server on default port 8000:

```zsh
uv run mcp-markdown-ragdocs run
```

The server will:
1. Index documents (same as mcp command)
2. Expose HTTP API at `http://127.0.0.1:8000`
3. Provide REST endpoints for queries

See [API Endpoints](#api-endpoints) below for HTTP usage.

## Basic Usage

### Configuration

Create `.mcp-markdown-ragdocs/config.toml` in your project directory or at `~/.config/mcp-markdown-ragdocs/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "~/Documents/Notes"  # Path to your Markdown files
index_path = ".index_data/"           # Where to store indices

[search]
semantic_weight = 1.0      # Weight for semantic search results
keyword_weight = 1.0       # Weight for keyword search results
recency_bias = 0.5         # Boost for recently modified documents
rrf_k_constant = 60        # Reciprocal Rank Fusion constant
min_confidence = 0.0       # Score threshold (0.0 = disabled)
max_chunks_per_doc = 0     # Per-document limit (0 = disabled)
dedup_enabled = false      # Semantic deduplication
```

The server searches for configuration files in this order:
1. `.mcp-markdown-ragdocs/config.toml` in current directory
2. `.mcp-markdown-ragdocs/config.toml` in parent directories (walks up to root)
3. `~/.config/mcp-markdown-ragdocs/config.toml` (global fallback)

This supports **monorepo workflows** where you can place a shared configuration in the repository root.

If no configuration file exists, the server uses these defaults:
- Documents path: `.` (current directory)
- Server: `127.0.0.1:8000`
- Index storage: `.index_data/`

### CLI Commands

#### Start MCP Server (stdio)

```zsh
uv run mcp-markdown-ragdocs mcp
```

Starts stdio-based MCP server for VS Code and compatible MCP clients. Runs persistently until stopped.

#### Start HTTP Server

```zsh
uv run mcp-markdown-ragdocs run
```

Starts HTTP API server on port 8000 (default). Override with:

```zsh
uv run mcp-markdown-ragdocs run --host 0.0.0.0 --port 8080
```

#### Query Documents (CLI)

Query documents directly from command line:

```zsh
uv run mcp-markdown-ragdocs query "How do I configure authentication?"
```

With options:

```zsh
# JSON output for scripting
uv run mcp-markdown-ragdocs query "authentication" --json

# Limit number of results
uv run mcp-markdown-ragdocs query "authentication" --top-n 3

# Specify project context
uv run mcp-markdown-ragdocs query "authentication" --project my-project
```

#### Configuration Management

Check your configuration:

```zsh
uv run mcp-markdown-ragdocs check-config
```

Force a full index rebuild:

```zsh
uv run mcp-markdown-ragdocs rebuild-index
```

| Command | Purpose | Use When |
|---------|---------|----------|
| `mcp` | Stdio MCP server | Integrating with VS Code or MCP clients |
| `run` | HTTP API server | Development, testing, or HTTP-based integrations |
| `query` | CLI query | Scripting or quick document searches |
| `check-config` | Validate config | Debugging configuration issues |
| `rebuild-index` | Force reindex | Config changes or corrupted indices |

### MCP Integration

#### VS Code Configuration

Configure the MCP server in VS Code user settings or workspace settings.

**File:** `.vscode/settings.json` or `~/.config/Code/User/mcp.json`

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
      ],
      "type": "stdio"
    }
  }
}
```

#### Claude Desktop Configuration

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

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

#### Available Tools

The server exposes one MCP tool:

- `query_documents(query: string, top_n?: int)`: Search indexed documents using hybrid search and return synthesized answer with source documents.

**Parameters:**
- `query` (required): Natural language query or question
- `top_n` (optional): Maximum results to return (1-100, default: 5)

**Example query from MCP client:**

```json
{
  "query": "How do I configure authentication in the API?",
  "top_n": 5
}
```

The server returns a synthesized answer with source document citations.

### API Endpoints

Health check:

```zsh
curl http://127.0.0.1:8000/health
```

Server status (document count, queue size, failed files):

```zsh
curl http://127.0.0.1:8000/status
```

Query endpoint (standard):

```zsh
curl -X POST http://127.0.0.1:8000/query_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication configuration"}'
```

Query endpoint (streaming SSE):

```zsh
curl -X POST http://127.0.0.1:8000/query_documents_stream \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication configuration", "top_n": 3}' \
  -N
```

The streaming endpoint returns Server-Sent Events:

```
event: search_complete
data: {"count": 3}

event: token
data: {"token": "Authentication"}

event: token
data: {"token": " is"}

event: done
data: {"results": [{"content": "...", "file_path": "auth.md", "header_path": ["Configuration"], "score": 1.0}]}
```

Example response (standard endpoint):

```json
{
  "answer": "Authentication is configured via the auth.toml file...",
  "results": [
    {
      "content": "Authentication is configured in the auth section...",
      "file_path": "docs/authentication.md",
      "header_path": ["Configuration", "Authentication"],
      "score": 1.0
    },
    {
      "content": "Security settings include authentication tokens...",
      "file_path": "docs/security.md",
      "header_path": ["Security", "API Keys"],
      "score": 0.85
    }
  ]
}
```

Each result contains:
- `content`: The text from the matching document chunk
- `file_path`: Source file path relative to documents directory
- `header_path`: Document structure showing nested headers (semantic "breadcrumbs")
- `score`: Normalized similarity score [0, 1] where 1.0 is the best match

## Configuration Details

See [docs/configuration.md](docs/configuration.md) for exhaustive configuration reference including all TOML options, defaults, and environment variable support.

## Documentation

- [Architecture](docs/architecture.md) - System design, component overview, data flow
- [Configuration](docs/configuration.md) - Complete configuration reference
- [Hybrid Search](docs/hybrid-search.md) - Search strategies and RRF fusion algorithm
- [Integration](docs/integration.md) - VS Code MCP setup and client integration
- [Development](docs/development.md) - Development setup, testing, contributing

## License

MIT
