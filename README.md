# mcp-markdown-ragdocs

A Model Context Protocol server that provides semantic search over local Markdown documentation using hybrid retrieval.

## What it is

This is an MCP server that indexes local Markdown files and exposes a `query_documents` tool for retrieval-augmented generation. The server combines semantic search, keyword matching, and graph traversal to retrieve relevant document chunks.

## Why it exists

Technical documentation, personal notes, and project wikis are typically stored as Markdown files. Searching these collections manually or with grep is inefficient. This server provides a conversational interface to query documentation using natural language while automatically keeping the index synchronized with file changes.

Existing RAG solutions require manual database setup, explicit indexing steps, and ongoing maintenance. This server eliminates that friction with automatic file watching, zero-configuration defaults, and built-in index versioning.

## Features

- Hybrid search combining semantic embeddings (FAISS), keyword search (Whoosh), and graph traversal (NetworkX)
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

Start the server on default port 8000, indexing the current directory:

```zsh
uv run mcp-markdown-ragdocs run
```

The server will:
1. Scan for `*.md` files in the current directory
2. Build vector, keyword, and graph indices
3. Start file watching for automatic updates
4. Expose the MCP API at `http://127.0.0.1:8000`

## Basic Usage

### Configuration

Create `config.toml` in your project directory or at `~/.config/mcp-markdown-ragdocs/config.toml`:

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
```

If no configuration file exists, the server uses these defaults:
- Documents path: `.` (current directory)
- Server: `127.0.0.1:8000`
- Index storage: `.index_data/`

### CLI Commands

Check your configuration:

```zsh
uv run mcp-markdown-ragdocs check-config
```

Force a full index rebuild:

```zsh
uv run mcp-markdown-ragdocs rebuild-index
```

Start the server with custom host/port:

```zsh
uv run mcp-markdown-ragdocs run --host 0.0.0.0 --port 8080
```

### MCP Integration

The server exposes one MCP tool:

- `query_documents(query: string)`: Search the indexed documents and return a synthesized answer.

Example query from an MCP client:

```json
{
  "query": "How do I configure authentication in the API?"
}
```

The server returns a synthesized answer based on the most relevant document chunks retrieved by hybrid search.

### API Endpoints

Health check:

```zsh
curl http://127.0.0.1:8000/health
```

Server status (document count, queue size, failed files):

```zsh
curl http://127.0.0.1:8000/status
```

Query endpoint:

```zsh
curl -X POST http://127.0.0.1:8000/query_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication configuration"}'
```

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
