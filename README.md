# mcp-markdown-ragdocs

A Model Context Protocol server that provides semantic search over local Markdown documentation using hybrid retrieval.

## What it is

This is an MCP server that indexes local Markdown files and exposes a `query_documents` tool for hybrid semantic search. The server identifies relevant document sections using semantic search, keyword matching, and graph traversal, enabling efficient discovery without loading entire documentation collections into LLM context.

## Why it exists

Technical documentation, personal notes, and project wikis are typically stored as Markdown files. Searching these collections manually or with grep is inefficient. This server provides a conversational interface to query documentation using natural language while automatically keeping the index synchronized with file changes.

Existing RAG solutions require manual database setup, explicit indexing steps, and ongoing maintenance. This server eliminates that friction with automatic file watching, zero-configuration defaults, and built-in index versioning.

## Features

- Hybrid search combining semantic embeddings (FAISS), keyword search (Whoosh), and graph traversal (NetworkX)
- Cross-encoder re-ranking for improved precision (optional, ~50ms latency)
- Query expansion via concept vocabulary for better recall
- **Git history search:** Semantic search over commit history with metadata and delta context
- **Multi-project support:** Manage isolated indices for multiple projects on one machine with automatic project detection
- Server-Sent Events (SSE) streaming for real-time response delivery
- CLI query command with rich formatted output
- Automatic file watching with debounced incremental indexing
- Zero-configuration operation with sensible defaults
- Index versioning with automatic rebuild on configuration changes
- **Pluggable parser architecture:** Markdown and plain text (.txt) support out-of-the-box
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
1. Scan for `*.md` and `*.txt` files in the current directory
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

[parsers]
"**/*.md" = "MarkdownParser"          # Markdown files
"**/*.txt" = "PlainTextParser"        # Plain text files

[search]
semantic_weight = 1.0      # Weight for semantic search results
keyword_weight = 1.0       # Weight for keyword search results
recency_bias = 0.5         # Boost for recently modified documents
rrf_k_constant = 60        # Reciprocal Rank Fusion constant
min_confidence = 0.3       # Score threshold (default: 0.3)
max_chunks_per_doc = 2     # Per-document limit (default: 2)
dedup_enabled = true       # Semantic deduplication (default: true)
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

The server exposes two MCP tools:

**`query_documents`**: Search indexed documents using hybrid search and return ranked document chunks.

**`search_git_history`**: Search git commit history using natural language queries. Returns relevant commits with metadata, message, and diff context.

**Parameters:**
- `query` (required): Natural language query or question
- `top_n` (optional): Maximum results to return (1-100, default: 5)
- `min_score` (optional): Minimum confidence threshold (0.0-1.0, default: 0.3)
- `similarity_threshold` (optional): Semantic deduplication threshold (0.5-1.0, default: 0.85)
- `show_stats` (optional): Show compression statistics (default: false)

**Note:** Compression is enabled by default (`min_score=0.3`, `max_chunks_per_doc=2`, `dedup_enabled=true`) to reduce token overhead by 40-60%. Results use compact format: `[N] file § section (score)\ncontent`

**Usage Pattern:**
1. Call `query_documents` to identify relevant sections
2. Review returned chunks to locate specific files and sections
3. Use file reading tools to access full document context

**Example query from MCP client:**

```json
{
  "query": "How do I configure authentication in the API?",
  "top_n": 5,
  "min_score": 0.3
}
```

The server returns ranked document chunks with file paths, header hierarchies, and relevance scores.

**`search_git_history`**: Search git commit history using natural language queries.

**Parameters:**
- `query` (required): Natural language query describing commits to find
- `top_n` (optional): Maximum commits to return (1-100, default: 5)
- `min_score` (optional): Minimum relevance threshold (0.0-1.0, default: 0.0)
- `file_pattern` (optional): Glob pattern to filter by changed files (e.g., `src/**/*.py`)
- `author` (optional): Filter commits by author name or email
- `after` (optional): Unix timestamp to filter commits after this date
- `before` (optional): Unix timestamp to filter commits before this date

**Note:** Git history search indexes up to 200 lines of diff per commit. Indexing processes 60 commits/sec on average. Search latency averages 5ms for 10k commits.

**Example query:**

```json
{
  "query": "fix authentication bug",
  "top_n": 5,
  "file_pattern": "src/auth/**",
  "after": 1704067200
}
```

The server returns ranked commits with hash, title, author, timestamp, message, files changed, and truncated diff.

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
  "results": [
    {
      "chunk_id": "authentication_0",
      "content": "Authentication is configured in the auth section...",
      "file_path": "docs/authentication.md",
      "header_path": ["Configuration", "Authentication"],
      "score": 1.0
    },
    {
      "chunk_id": "security_2",
      "content": "Security settings include authentication tokens...",
      "file_path": "docs/security.md",
      "header_path": ["Security", "API Keys"],
      "score": 0.85
    }
  ]
}
```

**MCP Stdio Format (Compact):**

For MCP clients (VS Code, Claude Desktop), results use compact format:

```
[1] docs/authentication.md § Configuration > Authentication (1.00)
Authentication is configured in the auth section...

[2] docs/security.md § Security > API Keys (0.85)
Security settings include authentication tokens...
```

Factual queries (e.g., "getUserById function", "configure auth") truncate content to 200 characters:

```
[1] docs/api.md § Functions > getUserById (0.92)
getUserById(id: string): User | null

Retrieves user by ID. Returns null if not found. Example:
  const user = getUserById("123");
  if (user) ...
```
```

Each result contains:
- `chunk_id`: Unique identifier for the document chunk
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
