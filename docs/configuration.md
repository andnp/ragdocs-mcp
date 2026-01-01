# Configuration Reference

This document provides an exhaustive reference for all configuration options and CLI commands in mcp-markdown-ragdocs.

## CLI Commands

### mcp

Start stdio-based MCP server for VS Code and compatible MCP clients.

```zsh
uv run mcp-markdown-ragdocs mcp [OPTIONS]
```

**Options:**
- `--project TEXT`: Override project detection by specifying project name or absolute path

**Usage:**

Starts a persistent MCP server using stdio transport. VS Code or Claude Desktop manages the server lifecycle. The server remains running until terminated by the client or user.

**Examples:**

```zsh
# Start with automatic project detection
uv run mcp-markdown-ragdocs mcp

# Start with specific project
uv run mcp-markdown-ragdocs mcp --project monorepo

# Start with project path
uv run mcp-markdown-ragdocs mcp --project /home/user/myproject
```

**When to Use:**
- Integrating with VS Code MCP extension
- Integrating with Claude Desktop
- Any MCP client supporting stdio transport

**Behavior:**
- Indexes documents on startup (or loads existing index)
- Starts file watcher for automatic updates
- Exposes `query_documents` tool via stdio
- Runs persistently until stopped

### run

Start HTTP API server for development and testing.

```zsh
uv run mcp-markdown-ragdocs run [OPTIONS]
```

**Options:**
- `--host TEXT`: IP address to bind (default: 127.0.0.1)
- `--port INTEGER`: TCP port to bind (default: 8000)
- `--project TEXT`: Override project detection

**Usage:**

Starts an HTTP server exposing REST API endpoints. Suitable for development, testing, or custom HTTP-based integrations.

**Examples:**

```zsh
# Start on default host/port
uv run mcp-markdown-ragdocs run

# Listen on all interfaces
uv run mcp-markdown-ragdocs run --host 0.0.0.0

# Custom port
uv run mcp-markdown-ragdocs run --port 8080

# With project override
uv run mcp-markdown-ragdocs run --project my-docs
```

**When to Use:**
- Development and testing
- Direct HTTP API access
- Custom integrations not using MCP
- Debugging query behavior

**Behavior:**
- Same indexing and file watching as `mcp` command
- Exposes HTTP endpoints: `/health`, `/status`, `/query_documents`
- Runs persistently until stopped (Ctrl+C)

### query

Query documents directly from command line.

```zsh
uv run mcp-markdown-ragdocs query QUERY_TEXT [OPTIONS]
```

**Arguments:**
- `QUERY_TEXT`: Natural language query or question (required)

**Options:**
- `--json`: Output results as JSON instead of formatted text
- `--top-n INTEGER`: Maximum number of results (default: 5, max: 100)
- `--project TEXT`: Override project detection

**Usage:**

Executes a one-time query against the indexed documents and outputs results to stdout. Suitable for scripting, testing, or quick searches.

**Examples:**

```zsh
# Basic query with formatted output
uv run mcp-markdown-ragdocs query "How do I configure authentication?"

# JSON output for scripting
uv run mcp-markdown-ragdocs query "authentication" --json

# Limit results
uv run mcp-markdown-ragdocs query "deployment" --top-n 3

# Query specific project
uv run mcp-markdown-ragdocs query "API reference" --project monorepo
```

**Output Formats:**

**Formatted (default):**

```
Query: How do I configure authentication?

Found 3 results:

╭─ #1 Score: 0.8542 ──────────────────────────────╮
│ Document: authentication                         │
│ Section: Configuration > OAuth                   │
│ File: docs/auth.md                              │
│                                                  │
│ To configure authentication, set the auth        │
│ section in config.toml...                       │
╰──────────────────────────────────────────────────╯
```

**JSON:**

```json
{
  "query": "How do I configure authentication?",
  "results": [
    {
      "doc_id": "authentication",
      "content": "To configure authentication...",
      "file_path": "docs/auth.md",
      "header_path": "Configuration > OAuth",
      "score": 0.8542
    }
  ]
}
```

**When to Use:**
- Quick documentation searches from terminal
- Shell scripts processing query results
- Testing search behavior
- Verifying indexed content

**Behavior:**
- Loads existing index (does not rebuild)
- Fails if no index exists (run `rebuild-index` first)
- Suppresses logging for clean output
- Exits after displaying results

### check-config

Validate configuration and display resolved settings.

```zsh
uv run mcp-markdown-ragdocs check-config [OPTIONS]
```

**Options:**
- `--project TEXT`: Override project detection

**Usage:**

Loads configuration, validates all settings, and displays effective values including project detection results.

**Examples:**

```zsh
# Check configuration
uv run mcp-markdown-ragdocs check-config

# Check with project override
uv run mcp-markdown-ragdocs check-config --project monorepo
```

**Output:**

```
╭─ Configuration ─────────────────────────────────╮
│ Setting              │ Value                    │
├──────────────────────┼──────────────────────────┤
│ Server Host          │ 127.0.0.1               │
│ Server Port          │ 8000                     │
│ Documents Path       │ /home/user/docs         │
│ Index Path           │ .index_data/            │
│ Recursive            │ True                     │
│                      │                          │
│ Registered Projects  │ 3 project(s)            │
│   • monorepo         │ /home/user/work/mono    │
│   • notes            │ /home/user/notes        │
│                      │                          │
│ Active Project       │ ✅ monorepo             │
│                      │                          │
│ Semantic Weight      │ 1.0                      │
│ Keyword Weight       │ 1.0                      │
│ Recency Bias         │ 0.5                      │
╰──────────────────────────────────────────────────╯

✅ Configuration is valid

📊 Index exists at: /home/user/.local/share/mcp-markdown-ragdocs/monorepo
```

**When to Use:**
- Debugging configuration issues
- Verifying project detection
- Checking resolved paths before indexing
- Confirming multi-project setup

**Behavior:**
- Loads and validates configuration
- Does not modify any files
- Does not build or load indices
- Exits after displaying information

### rebuild-index

Force a full index rebuild.

```zsh
uv run mcp-markdown-ragdocs rebuild-index [OPTIONS]
```

**Options:**
- `--project TEXT`: Override project detection

**Usage:**

Forces a complete reindex of all documents. Deletes existing indices and rebuilds from scratch.

**Examples:**

```zsh
# Rebuild index for current directory/project
uv run mcp-markdown-ragdocs rebuild-index

# Rebuild specific project
uv run mcp-markdown-ragdocs rebuild-index --project monorepo
```

**Output:**

```
Indexing documents... ████████████████████ 147/147 00:00:45

✅ Successfully rebuilt index: 147 documents indexed
```

**When to Use:**
- Configuration changes (embedding model, parsers, chunking)
- Corrupted or missing index
- After bulk document changes
- Testing indexing behavior

**Behavior:**
- Discovers all matching files based on `include`/`exclude` patterns
- Displays progress bar during indexing
- Persists new index and manifest
- Overwrites existing index files

## Configuration File Discovery

The server searches for `config.toml` in the following order. The first file found is used:

1. `.mcp-markdown-ragdocs/config.toml` in current directory
2. `.mcp-markdown-ragdocs/config.toml` in parent directories (walks up to filesystem root)
3. `$HOME/.config/mcp-markdown-ragdocs/config.toml` (user-global configuration)

This discovery order supports **monorepo workflows** where a single configuration can be placed in a parent directory and shared across multiple projects. The server walks up the directory tree until it finds a `.mcp-markdown-ragdocs/config.toml` file or reaches the filesystem root.

If no configuration file is found, all options use their default values.

## Multi-Project Support

The global configuration file (`~/.config/mcp-markdown-ragdocs/config.toml`) supports registering multiple projects. The server automatically detects which project you're working in based on your current directory and uses isolated indices for each project.

See [Multi-Project Setup Guide](guides/multi-project-setup.md) for complete details.

### [[projects]]

Define multiple projects with automatic detection and isolated storage.

```toml
[[projects]]
name = "monorepo"
path = "/home/user/work/monorepo"

[[projects]]
name = "personal-notes"
path = "/home/user/Documents/notes"
```

#### `name`

- **Type:** string
- **Required:** yes
- **Constraints:** Alphanumeric, hyphens, and underscores only
- **Description:** Unique identifier for the project. Used as directory name in data storage.

#### `path`

- **Type:** string
- **Required:** yes
- **Constraints:** Must be absolute path
- **Description:** Project root directory. The server detects the project when your current directory is under this path.

**Project Detection:**
- If CWD matches or is a subdirectory of a registered project path, that project is active
- For nested projects, the deepest match wins
- Indices stored in `~/.local/share/mcp-markdown-ragdocs/{project-name}/`

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

#### `include`

- **Type:** list[string]
- **Default:** `["**/*"]`
- **Description:** Glob patterns for files to include in indexing. Only files matching at least one include pattern will be indexed. Uses glob syntax similar to `.gitignore`.
- **Example:**
  ```toml
  [indexing]
  include = ["**/*.md", "**/*.txt"]  # Only markdown and text files
  ```

#### `exclude`

- **Type:** list[string]
- **Default:** `["**/.venv/**", "**/venv/**", "**/build/**", "**/dist/**", "**/.git/**", "**/node_modules/**", "**/__pycache__/**", "**/.pytest_cache/**"]`
- **Description:** Glob patterns for files/directories to exclude from indexing. Exclude patterns take precedence over include patterns.
- **Common patterns:**
  - Virtual environments: `**/.venv/**`, `**/venv/**`
  - Build artifacts: `**/build/**`, `**/dist/**`, `**/target/**`
  - Version control: `**/.git/**`, `**/.svn/**`
  - Dependencies: `**/node_modules/**`, `**/vendor/**`
  - Python cache: `**/__pycache__/**`, `**/.pytest_cache/**`
- **Example:**
  ```toml
  [indexing]
  exclude = ["**/drafts/**", "**/archive/**", "**/templates/**"]
  ```

#### `exclude_hidden_dirs`

- **Type:** boolean
- **Default:** `true`
- **Description:** Automatically exclude files in hidden directories (directories starting with `.`). When enabled, any file path containing a directory component that starts with a dot will be excluded, regardless of include/exclude patterns. This is useful for avoiding indexing of directories like `.stversions`, `.cache`, `.config`, etc.
- **Behavior:** Hidden directory exclusion is checked before include/exclude pattern matching. Set to `false` to disable this feature and rely only on explicit exclude patterns.
- **Example:**
  ```toml
  [indexing]
  exclude_hidden_dirs = false  # Disable automatic hidden directory exclusion
  ```
- **Note:** This setting works independently of the exclude patterns. Even if `.git` is in your exclude list, setting `exclude_hidden_dirs = true` will also exclude `.stversions`, `.cache`, and any other hidden directories without needing to list them explicitly.

#### `reconciliation_interval_seconds`

- **Type:** integer
- **Default:** `3600` (1 hour)
- **Description:** Interval in seconds between automatic reconciliation checks. Reconciliation compares the filesystem with indexed files and automatically removes stale entries (deleted files or newly excluded files) and indexes new files. Set to `0` to disable periodic reconciliation.
- **Behavior:**
  - Reconciliation always runs once on server startup
  - If enabled, runs periodically in the background at the specified interval
  - Catches edge cases like files deleted while server was offline, or config changes
- **Range:** 0 (disabled) to any positive integer (recommended: 1800-7200 seconds)
- **Example:**
  ```toml
  [indexing]
  reconciliation_interval_seconds = 1800  # Every 30 minutes
  # reconciliation_interval_seconds = 0  # Disable periodic reconciliation
  ```
- **Performance Impact:** Reconciliation is lightweight (just filesystem scan + comparison), typically adds <1s overhead per run.

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

### [chunking]

Controls document chunking strategy for vector indexing.

#### `strategy`

- **Type:** string
- **Default:** `"header_based"`
- **Description:** Chunking strategy to use. Currently only `"header_based"` is supported, which splits documents at Markdown headers.
- **Note:** Changing this value triggers a full index rebuild on next startup.
- **Example:**
  ```toml
  [chunking]
  strategy = "header_based"
  ```

#### `min_chunk_chars`

- **Type:** integer
- **Default:** `200`
- **Description:** Minimum chunk size in characters. Chunks smaller than this will be merged with adjacent chunks.
- **Range:** 50 to 10000 (typical: 100 to 500)
- **Effect:** Smaller values create more granular chunks; larger values create broader context chunks.
- **Example:**
  ```toml
  [chunking]
  min_chunk_chars = 200
  ```

#### `max_chunk_chars`

- **Type:** integer
- **Default:** `2000`
- **Description:** Maximum chunk size in characters. Chunks larger than this will be split at sentence boundaries.
- **Range:** 500 to 20000 (typical: 1000 to 4000)
- **Effect:** Smaller values create more focused chunks; larger values preserve more context.
- **Example:**
  ```toml
  [chunking]
  max_chunk_chars = 1500  # Smaller chunks for focused retrieval
  ```

#### `overlap_chars`

- **Type:** integer
- **Default:** `100`
- **Description:** Number of overlapping characters between adjacent chunks. Preserves context across chunk boundaries.
- **Range:** 0 to 500 (typical: 50 to 200)
- **Effect:** Larger overlap increases context preservation but storage overhead.
- **Example:**
  ```toml
  [chunking]
  overlap_chars = 100
  ```

#### `include_parent_headers`

- **Type:** boolean
- **Default:** `true`
- **Description:** Whether to include parent section headers in chunk metadata. Enables semantic "breadcrumbs" in search results.
- **Example:**
  ```toml
  [chunking]
  include_parent_headers = true
  ```

#### `parent_retrieval_enabled`

- **Type:** boolean
- **Default:** `false`
- **Description:** Enable two-level chunking with parent document retrieval. When enabled, documents are chunked at two levels: larger parent sections (return unit) and smaller child chunks (retrieval unit). Search matches child chunks but returns parent sections for better context.
- **Effect:** Improves retrieval precision while providing sufficient context for LLM consumption.
- **Note:** Requires index rebuild when changing this setting.
- **Example:**
  ```toml
  [chunking]
  parent_retrieval_enabled = true
  ```

#### `parent_chunk_min_chars`

- **Type:** integer
- **Default:** `1500`
- **Description:** Minimum size in characters for parent chunks. Child chunks are grouped into parent sections until this minimum is reached.
- **Range:** 500 to 10000 (typical: 1000 to 2000)
- **Requires:** `parent_retrieval_enabled = true`
- **Example:**
  ```toml
  [chunking]
  parent_retrieval_enabled = true
  parent_chunk_min_chars = 1500
  ```

#### `parent_chunk_max_chars`

- **Type:** integer
- **Default:** `2000`
- **Description:** Maximum size in characters for parent chunks. When accumulated child content exceeds this, a new parent section begins.
- **Range:** 1000 to 20000 (typical: 1500 to 3000)
- **Requires:** `parent_retrieval_enabled = true`
- **Example:**
  ```toml
  [chunking]
  parent_retrieval_enabled = true
  parent_chunk_max_chars = 2000
  ```

**Chunking Trade-offs:**

| Setting | Small Values | Large Values |
|---------|-------------|--------------|
| `min_chunk_chars` | More granular results, may lose context | Better context, fewer results |
| `max_chunk_chars` | Focused results, less context | Broader context, may dilute relevance |
| `overlap_chars` | Less storage, less context | Better boundary matching, more storage |

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
- **Default:** `0.8`
- **Description:** Weight multiplier for keyword (BM25) search results in RRF fusion. Higher values increase influence of exact term matches.
- **Range:** 0.0 to infinity (typical: 0.5 to 2.0)
- **Example:**
  ```toml
  [search]
  keyword_weight = 1.0  # Increase keyword match weight
  ```

#### `recency_bias`

- **Type:** float
- **Default:** `0.5`
- **Description:** Configuration option for recency boost. **Note:** The current implementation uses fixed tier multipliers (1.2x for 7 days, 1.1x for 30 days) regardless of this setting. Reserved for future dynamic tier calculation.
- **Range:** 0.0 (no recency bias) to 1.0 (maximum recency bias)
- **Recency Tiers (applied during fusion):**
  - Last 7 days: 1.2x
  - Last 30 days: 1.1x
  - Over 30 days: 1.0x
- **Example:**
  ```toml
  [search]
  recency_bias = 0.0  # (Currently unused, tiers are fixed)
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

#### `min_confidence`

- **Type:** float
- **Default:** `0.0`
- **Description:** Minimum normalized score threshold for results. Results below this threshold are filtered out. Set to `0.0` to disable filtering (backward compatible default).
- **Range:** 0.0 to 1.0 (recommended: 0.3 for filtering low-relevance results)
- **Effect:** Higher values return fewer but more relevant results. When no results meet the threshold, an empty list is returned.
- **Example:**
  ```toml
  [search]
  min_confidence = 0.3  # Filter results below 30% confidence
  ```

#### `max_chunks_per_doc`

- **Type:** integer
- **Default:** `2`
- **Description:** Maximum number of chunks from a single document in results. Prevents result lists dominated by one document with many matching sections. Set to `0` to disable.
- **Range:** 0 (disabled) to any positive integer (recommended: 2-3)
- **Effect:** Lower values increase result diversity across documents.
- **Example:**
  ```toml
  [search]
  max_chunks_per_doc = 3  # Max 3 chunks per document
  ```

#### `dedup_enabled`

- **Type:** boolean
- **Default:** `false`
- **Description:** Enable semantic deduplication of results. When enabled, chunks with high cosine similarity are clustered, and only one representative per cluster is returned. Reduces redundancy in results.
- **Example:**
  ```toml
  [search]
  dedup_enabled = true
  ```

#### `dedup_similarity_threshold`

- **Type:** float
- **Default:** `0.80`
- **Description:** Cosine similarity threshold for clustering chunks during deduplication. Chunks with similarity above this threshold are considered duplicates.
- **Range:** 0.0 to 1.0 (recommended: 0.80 to 0.90)
- **Effect:** Lower values cluster more aggressively (fewer results). Higher values preserve more distinct chunks.
- **Requires:** `dedup_enabled = true`
- **Example:**
  ```toml
  [search]
  dedup_enabled = true
  dedup_similarity_threshold = 0.85  # Stricter deduplication
  ```

#### `rerank_enabled`

- **Type:** boolean
- **Default:** `false`
- **Description:** Enable cross-encoder re-ranking. When enabled, a cross-encoder model re-scores the top candidates after fusion and filtering, computing query-document relevance jointly for higher precision.
- **Performance:** Adds ~50ms latency for 10 candidates on CPU.
- **Example:**
  ```toml
  [search]
  rerank_enabled = true
  ```

#### `rerank_model`

- **Type:** string
- **Default:** `"cross-encoder/ms-marco-MiniLM-L-6-v2"`
- **Description:** HuggingFace model identifier for the cross-encoder. The model is downloaded on first use and cached locally.
- **Requires:** `rerank_enabled = true`
- **Options:**
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB, ~50ms/10 docs, recommended)
  - `cross-encoder/ms-marco-TinyBERT-L-2-v2` (17MB, ~30ms/10 docs, faster)
  - `BAAI/bge-reranker-base` (110MB, ~150ms/10 docs, higher quality)
- **Example:**
  ```toml
  [search]
  rerank_enabled = true
  rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
  ```

#### `rerank_top_n`

- **Type:** integer
- **Default:** `10`
- **Description:** Maximum number of candidates to pass to the cross-encoder for re-ranking. The re-ranker processes this many top results from the fusion pipeline.
- **Range:** 1 to 100 (recommended: 5 to 20)
- **Effect:** Higher values may improve recall at the cost of latency (~5ms per additional candidate).
- **Requires:** `rerank_enabled = true`
- **Example:**
  ```toml
  [search]
  rerank_enabled = true
  rerank_top_n = 10
  ```

#### `adaptive_weights_enabled`

- **Type:** boolean
- **Default:** `false`
- **Description:** Enable automatic query type classification and adaptive weight adjustment. When enabled, the system detects query intent (factual, navigational, exploratory) and adjusts search strategy weights accordingly.
- **Query Types:**
  - **Factual**: Queries with code identifiers, versions, quoted phrases → keyword weight × 1.5
  - **Navigational**: Queries mentioning sections, guides, documentation → graph weight × 1.5
  - **Exploratory**: Questions (what, how, why) → semantic weight × 1.3
- **Example:**
  ```toml
  [search]
  adaptive_weights_enabled = true
  ```

#### `code_search_enabled`

- **Type:** boolean
- **Default:** `false`
- **Description:** Enable specialized code block search index. When enabled, code blocks extracted from Markdown are indexed separately with code-aware tokenization that handles camelCase, snake_case, and programming identifiers.
- **Effect:** Improves retrieval of code snippets, function names, and technical identifiers.
- **Example:**
  ```toml
  [search]
  code_search_enabled = true
  ```

#### `code_search_weight`

- **Type:** float
- **Default:** `1.0`
- **Description:** Weight multiplier for code search results in RRF fusion.
- **Range:** 0.0 to infinity (typical: 0.5 to 2.0)
- **Requires:** `code_search_enabled = true`
- **Example:**
  ```toml
  [search]
  code_search_enabled = true
  code_search_weight = 1.2
  ```

#### `mmr_enabled`

- **Type:** boolean
- **Default:** `false`
- **Description:** Enable Maximal Marginal Relevance (MMR) for result selection. MMR balances relevance with diversity by penalizing results similar to already-selected items. When enabled, replaces per-document limiting as the diversity mechanism.
- **Example:**
  ```toml
  [search]
  mmr_enabled = true
  ```

#### `mmr_lambda`

- **Type:** float
- **Default:** `0.7`
- **Description:** Lambda parameter for MMR selection. Controls the trade-off between relevance (1.0) and diversity (0.0).
- **Range:** 0.0 to 1.0
  - `1.0`: Pure relevance ranking (no diversity)
  - `0.7`: Balanced (default, recommended)
  - `0.5`: Equal weight to relevance and diversity
  - `0.3`: Diversity-focused
- **Requires:** `mmr_enabled = true`
- **Example:**
  ```toml
  [search]
  mmr_enabled = true
  mmr_lambda = 0.7
  ```

#### `ngram_dedup_enabled`

- **Type:** boolean
- **Default:** `true`
- **Description:** Enable n-gram overlap deduplication as a fast pre-filter before semantic deduplication. Uses character trigrams and Jaccard similarity to detect near-duplicate content.
- **Effect:** Removes obvious duplicates cheaply, reducing candidates for expensive embedding-based dedup.
- **Example:**
  ```toml
  [search]
  ngram_dedup_enabled = true
  ```

#### `ngram_dedup_threshold`

- **Type:** float
- **Default:** `0.7`
- **Description:** Jaccard similarity threshold for n-gram deduplication. Chunks with n-gram similarity above this threshold are considered duplicates.
- **Range:** 0.0 to 1.0 (recommended: 0.6 to 0.8)
- **Effect:** Lower values cluster more aggressively (fewer results). Higher values preserve more distinct chunks.
- **Requires:** `ngram_dedup_enabled = true`
- **Example:**
  ```toml
  [search]
  ngram_dedup_enabled = true
  ngram_dedup_threshold = 0.7
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

## Complete Example Configuration

```toml
# .mcp-markdown-ragdocs/config.toml

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

# Optional: Filter files by pattern
include = ["**/*.md", "**/*.txt"]
exclude = ["**/drafts/**", "**/archive/**"]

[parsers]
# Use MarkdownParser for all .md and .markdown files
"**/*.md" = "MarkdownParser"
"**/*.markdown" = "MarkdownParser"

[chunking]
# Chunking strategy for vector indexing
strategy = "header_based"
min_chunk_chars = 200
max_chunk_chars = 1500
overlap_chars = 100
include_parent_headers = true

# Parent document retrieval (two-level chunking)
parent_retrieval_enabled = false
parent_chunk_min_chars = 1500
parent_chunk_max_chars = 2000

[search]
# Balance semantic and keyword search equally
semantic_weight = 1.0
keyword_weight = 1.0

# Moderate recency bias (prioritize documents modified in last 30 days)
recency_bias = 0.6

# Standard RRF constant
rrf_k_constant = 60

# Result filtering
min_confidence = 0.3           # Filter results below 30% confidence
max_chunks_per_doc = 2         # Limit chunks per document for diversity
dedup_enabled = true           # Enable semantic deduplication
dedup_similarity_threshold = 0.85  # Similarity threshold for clustering

# N-gram deduplication (fast pre-filter)
ngram_dedup_enabled = true
ngram_dedup_threshold = 0.7

# MMR diversity selection (alternative to max_chunks_per_doc)
mmr_enabled = false
mmr_lambda = 0.7

# Query type classification
adaptive_weights_enabled = false

# Code search
code_search_enabled = false
code_search_weight = 1.0

# Re-ranking (adds ~50ms latency)
rerank_enabled = true
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_top_n = 10

[llm]
# Use local embedding model
embedding_model = "local"
```

## Environment Variables

No environment variables are currently supported. All configuration is file-based.

## Configuration Scenarios

### Scenario 1: Personal Notes (Obsidian Vault)

```toml
# .mcp-markdown-ragdocs/config.toml

[indexing]
documents_path = "~/Documents/ObsidianVault"
index_path = "~/.cache/ragdocs-obsidian"

[search]
semantic_weight = 1.2  # Prefer conceptual connections
keyword_weight = 0.8
recency_bias = 0.7     # Prioritize recent notes
```

### Scenario 2: Project Documentation
# .mcp-markdown-ragdocs/config.toml


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
# .mcp-markdown-ragdocs/config.toml


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
# .mcp-markdown-ragdocs/config.toml


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

### Scenario 5: Monorepo Workflow

For monorepo setups, place `.mcp-markdown-ragdocs/config.toml` in the repository root. All projects within the monorepo will inherit this configuration:

```
monorepo/
├── .mcp-markdown-ragdocs/
│   └── config.toml          # Shared configuration
├── project-a/
│   └── docs/
├── project-b/
│   └── docs/
└── shared-docs/
```

When running the server from `monorepo/project-a/`, it will discover and use `monorepo/.mcp-markdown-ragdocs/config.toml`.

```toml
# .mcp-markdown-ragdocs/config.toml (in monorepo root)

[indexing]
# Index all docs in the monorepo
documents_path = "."
index_path = ".index_data/"
recursive = true

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
