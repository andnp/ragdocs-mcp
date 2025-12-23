# Configuration Reference

This document provides an exhaustive reference for all configuration options in mcp-markdown-ragdocs.

## Configuration File Discovery

The server searches for `config.toml` in the following order. The first file found is used:

1. `./config.toml` (current working directory)
2. `$HOME/.config/mcp-markdown-ragdocs/config.toml` (user-global configuration)

If no configuration file is found, all options use their default values.

## Configuration Sections

### [server]

Controls the HTTP server behavior.

#### `host`

- **Type:** string
- **Default:** `"127.0.0.1"`
- **Description:** IP address to bind the server to. Use `"0.0.0.0"` to listen on all interfaces.
- **Example:**
  ```toml
  [server]
  host = "0.0.0.0"  # Listen on all interfaces
  ```

#### `port`

- **Type:** integer
- **Default:** `8000`
- **Description:** TCP port for the HTTP server.
- **Example:**
  ```toml
  [server]
  port = 8080
  ```

### [indexing]

Controls document discovery and index management.

#### `documents_path`

- **Type:** string
- **Default:** `"."`
- **Description:** Path to the directory containing documents to index. Supports tilde expansion (`~`) and relative paths (resolved to absolute).
- **Security Note:** Avoid pointing to high-level system directories (e.g., `/`, `/etc`, `$HOME`). Scope to specific project or notes folders.
- **Example:**
  ```toml
  [indexing]
  documents_path = "~/Documents/ProjectDocs"
  ```

#### `index_path`

- **Type:** string
- **Default:** `".index_data/"`
- **Description:** Directory where persistent indices are stored. Supports tilde expansion and relative paths.
- **Storage Structure:**
  ```
  {index_path}/
  ├── index.manifest.json
  ├── vector/
  ├── keyword/
  └── graph/
  ```
- **Example:**
  ```toml
  [indexing]
  index_path = "~/.cache/mcp-ragdocs-indices"
  ```

#### `recursive`

- **Type:** boolean
- **Default:** `true`
- **Description:** Whether to search subdirectories recursively when discovering documents.
- **Example:**
  ```toml
  [indexing]
  recursive = false  # Only index top-level directory
  ```

### [parsers]

Maps file glob patterns to parser class names. Enables extending the server to new file types.

- **Type:** dict[string, string]
- **Default:**
  ```toml
  [parsers]
  "**/*.md" = "MarkdownParser"
  "**/*.markdown" = "MarkdownParser"
  ```
- **Description:** Keys are glob patterns matched against file paths. Values are parser class names registered in `src/parsers/`.
- **Behavior:** First matching pattern wins (pattern order matters).
- **Example:**
  ```toml
  [parsers]
  "**/*.md" = "MarkdownParser"
  "**/*.txt" = "PlainTextParser"  # Future extension
  "docs/api/*.md" = "APIDocParser"  # Specific parser for API docs
  ```

### [search]

Controls hybrid search behavior and result fusion.

#### `semantic_weight`

- **Type:** float
- **Default:** `1.0`
- **Description:** Weight multiplier for semantic (vector) search results in RRF fusion. Higher values increase influence of semantic similarity.
- **Range:** 0.0 to infinity (typical: 0.5 to 2.0)
- **Example:**
  ```toml
  [search]
  semantic_weight = 1.5  # Prefer semantic matches
  ```

#### `keyword_weight`

- **Type:** float
- **Default:** `1.0`
- **Description:** Weight multiplier for keyword (BM25) search results in RRF fusion. Higher values increase influence of exact term matches.
- **Range:** 0.0 to infinity (typical: 0.5 to 2.0)
- **Example:**
  ```toml
  [search]
  keyword_weight = 0.8  # De-emphasize keyword matches
  ```

#### `recency_bias`

- **Type:** float
- **Default:** `0.5`
- **Description:** Multiplier for recency boost. Controls how strongly recently modified documents are prioritized.
- **Range:** 0.0 (no recency bias) to 1.0 (maximum recency bias)
- **Recency Tiers (applied during fusion):**
  - Last 7 days: `1.0 + (recency_bias * 0.2)`
  - Last 30 days: `1.0 + (recency_bias * 0.1)`
  - Over 30 days: `1.0`
- **Example:**
  ```toml
  [search]
  recency_bias = 0.0  # Disable recency boosting
  ```

#### `rrf_k_constant`

- **Type:** integer
- **Default:** `60`
- **Description:** Constant `k` in Reciprocal Rank Fusion formula: `score = 1 / (k + rank)`. Higher values dampen the effect of top-ranked results.
- **Range:** 1 to infinity (typical: 20 to 100)
- **Effect:** Lower values make top results more dominant. Higher values distribute scores more evenly.
- **Example:**
  ```toml
  [search]
  rrf_k_constant = 30  # Increase influence of top-ranked results
  ```

### [llm]

Controls embedding model and LLM provider configuration.

#### `embedding_model`

- **Type:** string
- **Default:** `"local"`
- **Description:** Identifier for the embedding model. Currently only `"local"` is supported, which uses HuggingFace `BAAI/bge-small-en-v1.5` (384 dimensions).
- **Note:** Changing this value triggers a full index rebuild on next startup.
- **Example:**
  ```toml
  [llm]
  embedding_model = "local"
  ```

#### `llm_provider`

- **Type:** string or null
- **Default:** `null`
- **Description:** LLM provider for answer synthesis. Not currently implemented. Reserved for future use.
- **Example:**
  ```toml
  [llm]
  llm_provider = null
  ```

## Complete Example Configuration

```toml
# mcp-markdown-ragdocs configuration

[server]
host = "127.0.0.1"
port = 8000

[indexing]
# Path to documentation directory
documents_path = "~/Projects/my-project/docs"

# Store indices in project-local directory
index_path = ".ragdocs-index/"

# Search subdirectories recursively
recursive = true

[parsers]
# Use MarkdownParser for all .md and .markdown files
"**/*.md" = "MarkdownParser"
"**/*.markdown" = "MarkdownParser"

[search]
# Balance semantic and keyword search equally
semantic_weight = 1.0
keyword_weight = 1.0

# Moderate recency bias (prioritize documents modified in last 30 days)
recency_bias = 0.6

# Standard RRF constant
rrf_k_constant = 60

[llm]
# Use local embedding model
embedding_model = "local"

# No external LLM provider configured
llm_provider = null
```

## Environment Variables

No environment variables are currently supported. All configuration is file-based.

## Configuration Scenarios

### Scenario 1: Personal Notes (Obsidian Vault)

```toml
[indexing]
documents_path = "~/Documents/ObsidianVault"
index_path = "~/.cache/ragdocs-obsidian"

[search]
semantic_weight = 1.2  # Prefer conceptual connections
keyword_weight = 0.8
recency_bias = 0.7     # Prioritize recent notes
```

### Scenario 2: Project Documentation

```toml
[indexing]
documents_path = "~/Projects/myapp/docs"
index_path = ".ragdocs-index/"

[search]
semantic_weight = 1.0
keyword_weight = 1.2  # Prefer exact API/function names
recency_bias = 0.3    # Documentation changes less frequently
```

### Scenario 3: Research Papers

```toml
[indexing]
documents_path = "~/Research/Papers"
index_path = "~/.cache/ragdocs-research"
recursive = true

[search]
semantic_weight = 1.5  # Emphasize semantic similarity
keyword_weight = 0.5
recency_bias = 0.0     # Publication date more relevant than file mtime
```

### Scenario 4: Multi-Team Documentation Server

```toml
[server]
host = "0.0.0.0"       # Listen on all interfaces
port = 8080

[indexing]
documents_path = "/opt/team-docs"
index_path = "/var/lib/ragdocs-indices"

[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
```

## Index Rebuild Triggers

A full index rebuild is automatically triggered on server startup if:

1. **No manifest exists**: First run or corrupted index
2. **spec_version changed**: Index format upgrade
3. **embedding_model changed**: Different embedding dimensions or model
4. **parsers changed**: Different parser configuration

To force a manual rebuild at any time:

```zsh
uv run mcp-markdown-ragdocs rebuild-index
```

## Configuration Validation

Check your configuration file for errors:

```zsh
uv run mcp-markdown-ragdocs check-config
```

Output shows resolved paths and all configuration values.

## CLI Override Options

The `run` command accepts command-line overrides for server settings:

```zsh
uv run mcp-markdown-ragdocs run --host 0.0.0.0 --port 8080
```

These override any values in the configuration file for that server instance only.
