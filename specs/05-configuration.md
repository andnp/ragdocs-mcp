# 5. Configuration

The server's behavior is controlled by a `config.toml` file. The system is designed with strong defaults, so this file is only needed for customization.

## 5.1. Configuration File Location

The server will look for `config.toml` in the following locations, in order. The first one found will be used.

1.  `.mcp-markdown-ragdocs/config.toml` in current directory
2.  `.mcp-markdown-ragdocs/config.toml` in parent directories (walks up to filesystem root)
3.  `$HOME/.config/mcp-markdown-ragdocs/config.toml` (user-specific global configuration)

This discovery order supports **monorepo workflows** where a single configuration can be placed in a parent directory and shared across multiple projects. The server walks up the directory tree until it finds a `.mcp-markdown-ragdocs/config.toml` file or reaches the filesystem root.

If no configuration file is found, the default values specified below will be used.

## 5.2. Configuration Parameters

### `[[projects]]` (Multi-Project Support)

Define multiple projects with automatic detection and isolated storage (global config only).

```toml
[[projects]]
name = "monorepo"
path = "/home/user/work/monorepo"

[[projects]]
name = "personal-notes"
path = "/home/user/Documents/notes"
```

- `name`: string, required. Alphanumeric, hyphens, underscores only. Used as directory name in data storage.
- `path`: string, required. Absolute path to project root. Server detects project when CWD is under this path.

See [Multi-Project Support Spec](10-multi-project-support.md) for full details.

### `[server]`

Controls the behavior of the MCP server itself.
- `host`: string, default `"127.0.0.1"`
- `port`: integer, default `8000`

### `[indexing]`

Controls document discovery and index management.
- `documents_path`: string, default `"."`
  - The path to the directory containing the documents to be indexed.
  - **Security Note:** For security, avoid pointing this to high-level system directories (e.g., `/`, `/etc`, `$HOME`). It's best to scope it to a specific project or notes folder.
- `index_path`: string, default `".index_data/"`
  - The path where the persistent indices (Vector, Keyword, Graph) will be saved.
  - **Note:** When using multi-project support, this is automatically set to `~/.local/share/mcp-markdown-ragdocs/{project-name}/` unless explicitly overridden in project-local config.
- `recursive`: boolean, default `true`
  - Whether glob patterns should search recursively.

### `[parsers]`
A mapping of file glob patterns to the parser to be used. This allows for extending the server to new file types.
- **Default:** `{"**/*.md": "MarkdownParser", "**/*.markdown": "MarkdownParser"}`

### `[search]`

Controls the behavior of the hybrid search engine.
- `semantic_weight`: float, default `1.0`
- `keyword_weight`: float, default `1.0`
- `recency_bias`: float, default `0.5`
- `rrf_k_constant`: integer, default `60`

### `[llm]`

Controls the LLM used for synthesis. *(Note: For v1, these may not be fully implemented.)*
- `embedding_model`: string, default `"local"`
- `llm_provider`: string, default `null`

## 5.3. Example `config.toml`

```toml
# .mcp-markdown-ragdocs/config.toml
# Example configuration for mcp-markdown-ragdocs

[server]
host = "0.0.0.0"
port = 8080

[indexing]
# Path to my knowledge base
documents_path = "~/Documents/Notes"

# Store all indices in a central location
index_path = "~/.cache/mcp_rag_index"

[parsers]
# Use the markdown parser for all .md and .txt files in the vault.
"**/*.md" = "MarkdownParser"
"**/*.txt" = "MarkdownParser" # Or a future "PlainTextParser"

[search]
# Prefer semantic search results slightly over keyword results
semantic_weight = 1.2
keyword_weight = 1.0

# Apply a moderate bias towards recently edited notes
recency_bias = 0.6
```
