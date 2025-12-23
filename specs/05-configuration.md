# 5. Configuration

The server's behavior is controlled by a `config.toml` file. The system is designed with strong defaults, so this file is only needed for customization.

## 5.1. Configuration File Location

The server will look for `config.toml` in the following locations, in order. The first one found will be used.

1.  `./config.toml` (in the current working directory where the server is launched)
2.  `$HOME/.config/mcp-markdown-ragdocs/config.toml` (user-specific global configuration)

If no configuration file is found, the default values specified below will be used.

## 5.2. Configuration Parameters

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

