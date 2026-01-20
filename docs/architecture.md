# Architecture

This document describes the system architecture of mcp-markdown-ragdocs, including component responsibilities, data flow, and key design decisions.

## High-Level Architecture

The system consists of three primary subsystems with two transport modes:

1. **Indexing Service**: Monitors file changes and updates three distinct indices
2. **Query Orchestrator**: Executes parallel searches and fuses results
3. **Server Layer**: Exposes interfaces via stdio (MCP) or HTTP (REST API)
4. **Memory Management** (optional): Parallel "Memory Lane" with separate indices for AI memory persistence

**Transport Modes:**

- **Stdio Transport (`mcp` command)**: Used by VS Code, Claude Desktop, and MCP clients. Server communicates via stdin/stdout using MCP protocol.
- **HTTP Transport (`run` command)**: REST API for development, testing, and custom integrations.

Both transport modes use the same indexing and query orchestration subsystems.

```
┌──────────────────────────────────────────────────────────────┐
│                    Server Layer                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ MCP Server (stdio)   │  │ HTTP Server (FastAPI)        │ │
│  │ src/mcp_server.py    │  │ src/server.py                │ │
│  │                      │  │                              │ │
│  │ query_documents tool │  │ /health  /status  /query     │ │
│  └──────────┬───────────┘  └──────────┬───────────────────┘ │
└─────────────┼──────────────────────────┼──────────────────────┘
              │                          │
              │         ┌────────────────┴──────────────┐
              │         │                                │
              └─────────▼────────────────▼───────────────┘
        ┌───────────────────────────┐         ┌─────────────────────────┐
        │  Indexing Service         │         │  Query Orchestrator     │
        │                           │         │                         │
        │  ┌──────────────────┐    │         │  ┌──────────────────┐  │
        │  │ FileWatcher      │    │         │  │ Semantic Search  │  │
        │  │ (Watchdog)       │    │         │  │ (VectorIndex)    │  │
        │  └────────┬─────────┘    │         │  └──────────────────┘  │
        │           ▼               │         │  ┌──────────────────┐  │
        │  ┌──────────────────┐    │         │  │ Keyword Search   │  │
        │  │ Parser Dispatcher│    │         │  │ (KeywordIndex)   │  │
        │  └────────┬─────────┘    │         │  └──────────────────┘  │
        │           ▼               │         │  ┌──────────────────┐  │
        │  ┌──────────────────┐    │         │  │ Graph Traversal  │  │
        │  │ MarkdownParser   │    │         │  │ (GraphStore)     │  │
        │  │ (Tree-sitter)    │    │         │  └──────────────────┘  │
        │  └────────┬─────────┘    │         │           │            │
        │           ▼               │         │           ▼            │
        │  ┌──────────────────┐    │         │  ┌──────────────────┐  │
        │  │ IndexManager     │◀───┼─────────┼─▶│ RRF Fusion       │  │
        │  └────────┬─────────┘    │         │  │ + Synthesis      │  │
        │           ▼               │         │  └──────────────────┘  │
        └───────────┼───────────────┘         └─────────────────────────┘
                    ▼
        ┌───────────────────────────┐
        │  Storage Layer            │
        │  ┌──────────────────────┐ │
        │  │ VectorIndex (FAISS)  │ │
        │  └──────────────────────┘ │
        │  ┌──────────────────────┐ │
        │  │ KeywordIndex(Whoosh) │ │
        │  └──────────────────────┘ │
        │  ┌──────────────────────┐ │        │  │ CodeIndex (Whoosh)   │ │
        │  └──────────────────────┘ │
        │  ┌──────────────────────┐ │        │  │ GraphStore(NetworkX) │ │
        │  └──────────────────────┘ │
        │  ┌──────────────────────┐ │
        │  │ Manifest (JSON)      │ │
        │  └──────────────────────┘ │
        └───────────────────────────┘
```

## Component Overview

### Server Layer

#### MCP Server (src/mcp_server.py)

**Transport:** Stdio (stdin/stdout)

**Responsibilities:**
- Implement MCP protocol for tool invocation
- Manage server lifecycle (startup/shutdown)
- Coordinate with indexing service and orchestrator
- Expose `query_documents` tool

**Lifecycle:**
1. Load configuration and detect project
2. Initialize indices (vector, keyword, graph)
3. Check manifest for version changes
4. Index all documents if rebuild needed, otherwise load existing indices
5. Start file watcher
6. Enter stdio communication loop
7. On shutdown: stop watcher, persist indices

**Tool Definitions:**

`query_documents`:
```python
{
  "name": "query_documents",
  "description": "Search local Markdown documentation using hybrid search and return ranked document chunks",
  "inputSchema": {
    "query": "string (required)",
    "top_n": "integer (optional, default: 5, max: 100)"
  }
}
```

`query_documents_compressed`:
```python
{
  "name": "query_documents_compressed",
  "description": "Search with context compression. Filters low-relevance results and removes semantic duplicates.",
  "inputSchema": {
    "query": "string (required)",
    "top_n": "integer (optional, default: 5, max: 100)",
    "min_score": "number (optional, default: 0.3, range: 0.0-1.0)",
    "similarity_threshold": "number (optional, default: 0.85, range: 0.5-1.0)"
  }
}
```

`search_git_history`:
```python
{
  "name": "search_git_history",
  "description": "Search git commit history using natural language queries",
  "inputSchema": {
    "query": "string (required)",
    "top_n": "integer (optional, default: 5, max: 100)",
    "min_score": "number (optional, default: 0.0, range: 0.0-1.0)",
    "file_pattern": "string (optional)",
    "author": "string (optional)",
    "after": "integer (optional, Unix timestamp)",
    "before": "integer (optional, Unix timestamp)"
  }
}
```

#### HTTP Server (src/server.py)

**Transport:** HTTP REST API

**Responsibilities:**
- FastAPI application lifecycle management
- HTTP endpoint definitions
- Dependency injection for indices and orchestrator
- Same indexing logic as MCP server

**Endpoints:**
- `POST /query_documents`: Main query interface
- `POST /query_documents_stream`: Streaming SSE variant
- `GET /health`: Health check (returns `{"status": "ok"}`)
- `GET /status`: Operational status (document count, queue size, failed files)

**Lifecycle:** Same as MCP server, but uses HTTP transport instead of stdio.

### Indexing Service

#### FileWatcher (src/indexing/watcher.py)

**Technology:** Python watchdog library

**Responsibilities:**
- Monitor configured documents_path for file system events
- Debounce rapid file changes (500ms timeout)
- Queue events for batch processing
- Maintain list of failed files for status reporting

**Event Flow:**
1. Watchdog detects file creation/modification/deletion
2. Event placed on `queue.Queue` (thread-safe)
3. Background async task processes queue with debouncing
4. Unique file paths batched and passed to IndexManager
5. Failed operations logged and tracked

**Debouncing Algorithm:**

Data structures:
- Observer (watchdog) runs in separate thread
- Events passed via thread-safe `queue.Queue[tuple[EventType, str]]`
- Event aggregation: `dict[file_path, event_type]` where **last event wins**

Processing loop:
```python
async def _process_events():
    events = {}  # file_path -> event_type

    while running:
        try:
            # Block for 0.5s waiting for event
            event_type, file_path = queue.get(timeout=0.5)
            events[file_path] = event_type  # Last event wins
        except queue.Empty:
            # Queue empty for 0.5s: process accumulated events
            if events:
                await _batch_process(events)
                events = {}
```

Key characteristics:
- Timeout is on `queue.get()`, not cooldown after last event
- Multiple events for same file: **last one wins** (not all processed)
- Batch triggers when queue empty for 0.5s
- Reduces rapid successive operations (save, save, save) to single reindex

#### IndexManager (src/indexing/manager.py)

**Responsibilities:**
- Coordinate updates across all three indices
- Dispatch parsing to appropriate parser based on file extension
- Persist and load all indices atomically
- Track document count across indices

**Methods:**
- `index_document(file_path)`: Parse file and update all indices
- `remove_document(doc_id)`: Remove document from keyword and graph indices
- `persist()`: Save all indices to disk
- `load()`: Load all indices from disk
- `get_document_count()`: Query keyword index for total document count

**Design Decisions:**
- Error in one index does not prevent others from updating
- Errors logged but not raised to allow batch processing to continue
- Document ID derived from file stem (e.g., `notes/api.md` → `api`)

#### Parser System

**ParserRegistry (src/parsers/dispatcher.py):**

Maps glob patterns to parser classes:

```python
{
  "**/*.md": "MarkdownParser",
  "**/*.markdown": "MarkdownParser"
}
```

**MarkdownParser (src/parsers/markdown.py):**

**Technology:** tree-sitter with tree-sitter-markdown grammar

**Extraction Logic:**
1. **YAML Frontmatter**: Parse frontmatter block for metadata (tags, aliases, custom fields)
2. **Content**: Extract body text excluding code blocks
3. **Wikilinks**: Extract `[[Note]]` and `[[Note|Display]]` patterns (excludes transclusions)
4. **Transclusions**: Extract `![[Note]]` patterns into separate metadata field
5. **Tags**: Extract `#tag` and `#hyphenated-tag` patterns from content
6. **Merge Tags**: Combine frontmatter tags with inline tags (deduplicated, sorted)

**Output:** `Document` object with:
- `content`: Plain text content (code blocks excluded)
- `metadata`: Dict with frontmatter fields
- `links`: List of wikilink targets
- `tags`: Sorted list of unique tags
- `doc_id`: File stem
- `modified_time`: File mtime

#### Manifest System (src/indexing/manifest.py)

**Purpose:** Detect configuration changes that require index rebuild, track indexed files for reconciliation

**IndexManifest Structure:**
```python
@dataclass
class IndexManifest:
    spec_version: str        # Index format version (e.g., "1.0.0")
    embedding_model: str     # Embedding model identifier
    parsers: dict[str, str]  # Glob pattern → parser class mapping
    chunking_config: dict[str, Any]  # Chunking parameters
    indexed_files: dict[str, str] | None = None  # doc_id → relative_file_path
```

**indexed_files Field:**
- Type: `dict[doc_id, relative_file_path]` (optional for backward compatibility)
- Purpose: Track which files are currently indexed
- Enables reconciliation feature to detect:
  - Deleted files (doc_id in manifest but file missing from filesystem)
  - Newly excluded files (file matches new exclude pattern)
- Populated during indexing, persisted with manifest
- Relative paths stored to support project relocation

**Rebuild Logic:**

Rebuild triggered if:
- No saved manifest exists (first run or corrupted index)
- `spec_version` changed (index format incompatible)
- `embedding_model` changed (embeddings incompatible)
- `parsers` configuration changed (parsing logic differs)
- `chunking_config` changed (chunk boundaries differ)

**Reconciliation Logic:**

Reconciliation triggered if:
- Server startup and `reconciliation_interval_seconds > 0`
- Periodic background task (every `reconciliation_interval_seconds`)

Reconciliation compares `indexed_files` against current filesystem:
1. Discover all files matching include/exclude patterns
2. Identify stale entries (doc_id in manifest but file missing/excluded)
3. Remove stale documents from indices
4. Identify new files (file exists but doc_id not in manifest)
5. Index new documents

**Example Manifest:**
```json
{
  "spec_version": "1.0.0",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "parsers": {
    "**/*.md": "MarkdownParser"
  },
  "chunking_config": {
    "strategy": "header_based",
    "min_chunk_chars": 200,
    "max_chunk_chars": 2000,
    "overlap_chars": 50
  },
  "indexed_files": {
    "api-reference": "docs/api-reference.md",
    "configuration": "docs/configuration.md",
    "getting-started": "README.md"
  }
}
```

**Rebuild Logic (continued):**

Rebuild triggered if:
- `spec_version` changed
- `embedding_model` changed
- `parsers` configuration changed
- `chunking_config` changed

**Storage:** `{index_path}/index.manifest.json`

### Index Reconciliation

**Purpose:** Maintain index consistency with filesystem state by detecting deleted files and newly excluded files.

**Configuration:**
```toml
[indexing]
reconciliation_interval_seconds = 3600  # 1 hour, 0 to disable
```

**Reconciliation Algorithm:**
```python
def reconcile_index(
    manifest: IndexManifest,
    documents_path: str,
    include: list[str],
    exclude: list[str]
) -> tuple[list[str], list[str]]:
    """Compare manifest against filesystem, return stale and new file lists.

    Returns: (stale_doc_ids, new_file_paths)
    """
    # Step 1: Discover all files matching current include/exclude patterns
    filesystem_files = discover_files(documents_path, include, exclude)
    filesystem_doc_ids = {compute_doc_id(f) for f in filesystem_files}

    # Step 2: Compare against manifest.indexed_files
    indexed_doc_ids = set(manifest.indexed_files.keys()) if manifest.indexed_files else set()

    # Step 3: Identify stale entries (indexed but no longer present/included)
    stale = indexed_doc_ids - filesystem_doc_ids

    # Step 4: Identify new files (present but not indexed)
    new = filesystem_doc_ids - indexed_doc_ids
    new_file_paths = [
        f for f in filesystem_files
        if compute_doc_id(f) in new
    ]

    return list(stale), new_file_paths
```

**Reconciliation Execution:**
```python
async def run_reconciliation(index_manager: IndexManager):
    stale_doc_ids, new_file_paths = reconcile_index(
        manifest=index_manager.manifest,
        documents_path=config.indexing.documents_path,
        include=config.indexing.include,
        exclude=config.indexing.exclude
    )

    # Remove stale documents
    for doc_id in stale_doc_ids:
        logger.info(f"Reconciliation: Removing deleted document {doc_id}")
        index_manager.remove_document(doc_id)

    # Index new files
    for file_path in new_file_paths:
        logger.info(f"Reconciliation: Indexing new file {file_path}")
        index_manager.index_document(file_path)

    # Persist updated index and manifest
    if stale_doc_ids or new_file_paths:
        index_manager.persist()
```

**Trigger Conditions:**
1. **Server Startup:** Runs once after index loaded if `reconciliation_interval_seconds > 0`
2. **Periodic Background Task:** Runs every `reconciliation_interval_seconds` in async loop
3. **Manual Trigger:** `IndexManager.reconcile()` method (future CLI command)

**Use Cases:**
- **Deleted Files:** User deletes `notes/old-draft.md`, reconciliation removes from index
- **Config Changes:** User adds `**/archive/**` to exclude patterns, reconciliation removes archived docs
- **External Modifications:** File added by git pull or external script, reconciliation indexes it
- **Index Corruption Recovery:** Manifest out of sync with index, reconciliation corrects discrepancies

**Behavior:**
- Disabled by default (`reconciliation_interval_seconds = 0`)
- Recommended: 1 hour (3600) for active projects, 24 hours (86400) for stable documentation
- Logs each removal/addition at INFO level for auditability
- Persists index and manifest after changes

**Manifest Updates:**
- `indexed_files` updated after each reconciliation cycle
- Relative paths enable project relocation without reconciliation triggering

### Storage Layer

#### VectorIndex (src/indices/vector.py)

**Technology:** LlamaIndex with FAISS and HuggingFace embeddings

**Configuration:**
- Embedding model: BAAI/bge-small-en-v1.5 (384 dimensions)
- Chunking: LlamaIndex MarkdownNodeParser (512 chars, 50 overlap)
- Storage: FAISS IndexFlatL2 with doc_id to node_ids mapping

**Operations:**
- `add(doc_id, content)`: Chunk content, embed, add to FAISS index
- `remove(doc_id)`: Remove doc_id mapping (FAISS vectors remain)
- `search(query, top_k)`: Embed query, cosine similarity search
- `persist(path)`: Save FAISS index and mappings to disk
- `load(path)`: Load existing index from disk

**Known Limitation:** FAISS does not support efficient vector deletion. The `remove()` method only removes the mapping, not the actual vectors.

#### KeywordIndex (src/indices/keyword.py)

**Technology:** Whoosh with BM25F scoring

**Schema:**
- `ID`: Stored, unique document identifier
- `TEXT`: Document content and aliases (searchable)
- `KEYWORD`: Tags (searchable)

**Operations:**
- `add(doc_id, content, metadata)`: Add document to inverted index
- `remove(doc_id)`: Delete document by ID
- `search(query, top_k)`: BM25 search across TEXT and KEYWORD fields
- `persist(path)`: Copy in-memory index to persistent directory
- `load(path)`: Load existing index from disk

**Parser Configuration:**
- MultifieldParser searches across content, aliases, and tags
- StandardAnalyzer tokenizes and normalizes terms

**Tokenization Limitation:** StandardAnalyzer strips punctuation. Terms like "C++" normalize to "c". Custom analyzer required for preserving special characters.

#### CodeIndex (src/indices/code.py)

**Technology:** Whoosh with custom code-aware analyzer

**Purpose:** Index code blocks extracted from Markdown with programming-aware tokenization.

**Schema:**
- `id`: Unique code block identifier
- `doc_id`: Parent document ID
- `chunk_id`: Associated chunk ID
- `content`: Code block text (searchable with code analyzer)
- `language`: Programming language (if specified in fenced block)

**Custom Analyzer:**
- **RegexTokenizer**: Extracts alphanumeric identifiers and numbers
- **CamelCaseSplitter**: Splits `getUserById` → `["getUserById", "get", "User", "By", "Id"]`
- **SnakeCaseSplitter**: Splits `parse_json` → `["parse_json", "parse", "json"]`

**Operations:**
- `add_code_block(code_block)`: Add code block to index
- `remove_by_doc_id(doc_id)`: Remove all code blocks for a document
- `search(query, top_k)`: BM25 search with code-aware tokenization
- `persist(path)`: Save index to disk
- `load(path)`: Load existing index from disk

#### GraphStore (src/indices/graph.py)

**Technology:** NetworkX directed graph

**Node Attributes:**
- `title`: Document title (from file name or frontmatter)
- `tags`: List of tags
- `aliases`: List of aliases

**Edge Types:**
- `link`: Standard wikilink (`[[Target]]`)
- `transclusion`: Embedded note (`![[Target]]`)

**Operations:**
- `add_document(doc_id, metadata, links)`: Add node and outgoing edges
- `remove_document(doc_id)`: Remove node and all connected edges
- `get_neighbors(doc_id, depth=1)`: Find connected documents
- `persist(path)`: Serialize graph to JSON
- `load(path)`: Deserialize graph from JSON

**Design:** Uses NetworkX's built-in node_link_data format for JSON serialization.

#### CommitIndex (src/git/commit_indexer.py)

**Technology:** SQLite with embedding storage

**Purpose:** Index git commit history for semantic search over commit metadata and diffs.

**Schema:**
- `hash` (TEXT, PRIMARY KEY): Full commit SHA
- `timestamp` (INTEGER): Unix seconds
- `author` (TEXT): Author name and email
- `committer` (TEXT): Committer name and email
- `title` (TEXT): First line of commit message
- `message` (TEXT): Full commit message body
- `files_changed` (TEXT): JSON array of file paths
- `delta_truncated` (TEXT): First 200 lines of diff output
- `embedding` (BLOB): 384-dim float32 embedding
- `indexed_at` (INTEGER): Unix timestamp when indexed
- `repo_path` (TEXT): Absolute path to .git directory

**Commit Document Format:**

Commits are formatted as searchable text before embedding:

```
{title}

{message}

Author: {author}
Committer: {committer}

Files changed:
{file_1}
{file_2}
...

{delta_truncated}
```

**Delta Truncation:**

Diffs are truncated to first 200 lines with indicator if truncated:

```diff
diff --git a/src/auth.py b/src/auth.py
index abc123..def456 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,5 +10,8 @@
 def authenticate(token):
+    if not validate_token(token):
+        raise AuthError("Invalid token")
     ...

... (450 lines omitted)
```

**Repository Discovery:**

Recursively searches for `.git` directories starting from `documents_path`, respecting exclusion patterns from `IndexingConfig.exclude`:

1. Use `os.walk()` with in-place directory filtering
2. Apply glob pattern matching via `Path.match()`
3. Skip hidden directories if `exclude_hidden_dirs=True`
4. Stop descent when `.git` found (no nested repo indexing)

**Embedding Model:**

Shares embedding model with VectorIndex (BAAI/bge-small-en-v1.5, 384 dimensions). Single model instance reduces memory overhead.

**Search Algorithm:**

1. Generate query embedding using shared model
2. Load all embeddings from SQLite (BLOB → numpy array)
3. Compute cosine similarity (numpy vectorized)
4. Sort by similarity descending
5. Apply optional filters (file_pattern, author, timestamp range)
6. Return top-N commits

**Incremental Updates:**

On startup and when GitWatcher detects changes:

1. Query last indexed timestamp for repository (normalized path)
2. Execute `git log --all --after={last_indexed_timestamp}` for new commits
3. Skip indexing if zero new commits found
4. Parse commit metadata and delta for new commits
5. Generate embedding and store in SQLite
6. Update `indexed_at` timestamp

**Path Normalization:**

Repository paths normalized before storage/retrieval to ensure consistent state tracking:
- Absolute path resolution
- `.git` suffix stripped
- Trailing slashes removed
- Example: `/repo/.git/` → `/repo`

This prevents timestamp lookup failures that would cause full reindexing on every startup.

**State Persistence:**

Last indexed timestamp stored per repository in `git_commits` table. The `repo_path` field uses normalized paths for consistent querying across restarts.

**Operations:**
- `add_commit(hash, metadata, delta, document, repo_path)`: Parse, embed, store commit
- `get_last_indexed_timestamp(repo_path)`: Retrieve last index time (with path normalization)
- `clear()`: Remove all commits (used by rebuild-index command)
- `query_by_embedding(query_embedding, top_k, filters)`: Semantic search
- `get_last_indexed_timestamp(repo_path)`: Track incremental indexing
- `persist()`: SQLite auto-persists on commit
- `close()`: Close database connection

**Storage Location:** `{index_path}/commits.db`

**Performance:**
- Indexing: 60 commits/sec (includes git operations, parsing, embedding)
- Query: 5ms average for 10k commits (cosine similarity in-memory)
- Storage: ~2KB per commit (metadata + embedding)

### Memory Management Subsystem

#### Dual-Lane Architecture

The Memory Management System implements a "Dual-Lane" pattern: a parallel corpus with its own indices that mirrors the main document pipeline.

| Component | Main Corpus | Memory Corpus |
|:--|:--|:--|
| **Source** | `docs/**/*.md` | `.memories/` or `~/.local/share/.../memories/` |
| **IndexStorage** | `indices/` | `memories/indices/` |
| **Orchestrator** | `SearchOrchestrator` | `MemorySearchOrchestrator` |
| **Graph** | Document nodes | Memory nodes + Ghost nodes |

#### Ghost Nodes and Typed Edges

To support cross-corpus linking ("What memories reference document X?"):

1. **Memory Graph**: Contains nodes for memory files
2. **Ghost Nodes**: When a memory contains `[[src/server.py]]`, a node `ghost:src/server.py` is created in the Memory Graph
3. **Typed Edges**: Edges carry `type` (e.g., `mentions`, `refactors`, `plans`) derived from context
4. **Anchor Context**: Edges store ~100 characters surrounding the link for relevance scoring

**Graph Structure:**

```
memory:project-notes  ──[mentions]──▶  ghost:src/auth.py
         │
         └──[plans]──▶  ghost:docs/roadmap.md
```

**Query:** `search_linked_memories(query, target_document)` performs graph traversal from `ghost:{target}`, filtering edges by context relevance.

#### MemoryIndexManager (src/memory/manager.py)

**Responsibilities:**
- Parse memory files (YAML frontmatter + Markdown body)
- Extract wikilinks and create ghost nodes
- Chunk memories using HeaderChunker
- Coordinate updates across vector, keyword, and graph indices
- Track memory metadata (type, status, tags, created_at)

**Memory Document Format:**

```yaml
---
type: "plan"       # plan | journal | fact | observation | reflection
status: "active"   # active | archived
tags: ["refactor", "auth"]
created_at: "2025-01-10T10:00:00Z"
---

Memory content with [[wikilinks]] to documents.
```

#### MemorySearchOrchestrator (src/memory/search.py)

**Responsibilities:**
- Execute parallel vector + keyword search on memory corpus
- Apply memory-specific recency boost (configurable days/factor)
- Filter results by tags and type
- Perform linked memory search via ghost node traversal

**Recency Boost Algorithm:**

```python
if (now - created_at).days <= recency_boost_days:
    score *= recency_boost_factor  # Default: 1.2x within 7 days
```

#### Storage Layout

```
{memory_path}/
├── *.md                    # Memory files
├── .trash/                 # Soft-deleted memories
└── indices/
    ├── vector/
    ├── keyword/
    └── graph/
        └── graph.json      # Includes ghost nodes
```

**Storage Strategy:**
- `"project"`: `{project_root}/.memories/`
- `"user"`: `~/.local/share/mcp-markdown-ragdocs/memories/`

### Git History Module

#### GitWatcher (src/git/watcher.py)

**Purpose:** Monitor `.git` directories for commit operations and trigger incremental indexing.

**Watch Targets:**
- `.git/HEAD`: Detects branch switches and commits
- `.git/refs/`: Detects new branches, tags, and ref updates

**Cooldown:** 5 seconds (longer than document watcher due to slower git operations)

**Event Flow:**
1. Watchdog detects file modification in `.git/HEAD` or `.git/refs/`
2. Event queued with associated `.git` directory path
3. After 5s of inactivity, batch process accumulated events
4. For each repository, query new commits and index incrementally

**Debouncing Rationale:**

Git operations often modify multiple refs in rapid succession (e.g., `git pull` updates remote refs and local branch). Cooldown prevents redundant indexing of same commits.

### Query Orchestrator (src/search/orchestrator.py)

**Responsibilities:**
- Execute parallel searches across all indices
- Expand queries via concept vocabulary
- Apply 1-hop graph neighbor boosting
- Fuse results using Reciprocal Rank Fusion
- Apply recency bias
- Filter by confidence threshold and per-document limits
- Deduplicate semantically similar chunks
- Re-rank results using cross-encoder model (optional)
- Return ranked document chunks

#### Search Strategies

**1. Query Expansion (VectorIndex):**
- Builds concept vocabulary from indexed chunks during `persist()`
- Extracts unique terms, embeds each using the same model
- On query, finds top-3 nearest terms via cosine similarity
- Appends expansion terms to query for improved recall
- Vocabulary persisted as `concept_vocabulary.json`

**2. Semantic Search (VectorIndex):**
- Expands query using concept vocabulary
- Embeds query using same model as documents
- Cosine similarity search in FAISS index
- Returns chunk IDs ranked by similarity

**3. Keyword Search (KeywordIndex):**
- BM25F scoring with field boosts:
  - title (3.0), headers (2.5), keywords (2.5)
  - description (2.0), tags (2.0), aliases (1.5)
- Returns chunk IDs ranked by term relevance

**4. Graph Traversal (GraphStore):**
- 1-hop neighbor boosting: for each candidate document from semantic/keyword search, retrieve all directly linked documents
- Neighbor documents added to result pool with reduced score (0.5x multiplier)
- Surfaces structurally related content

**5. Recency Bias:**
- Tier-based score multiplier:
  - Last 7 days: 1.2x
  - Last 30 days: 1.1x
  - Over 30 days: 1.0x
- Applied during fusion stage

**6. Cross-Encoder Re-Ranking (optional):**
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` by default
- Re-scores top candidates after fusion pipeline
- Computes query-document relevance jointly for higher precision
- Model loaded lazily on first rerank call
- Adds ~50ms latency for 10 candidates on CPU

### Search Infrastructure (Spec 17)

Advanced search features implemented in [src/search/](../src/search/).

#### Edge Types

Graph edges carry semantic relationship types inferred from document context:

| Edge Type | Trigger | Description |
|-----------|---------|-------------|
| `links_to` | Default | Standard wikilink `[[Target]]` |
| `implements` | Link under "Implementation" header | Code references |
| `tests` | Link in test files | Test coverage links |
| `related` | Link under "See Also" header | Topical relationship |

Edge types are stored as attributes on NetworkX graph edges and used during traversal scoring.

#### Community Detection

Documents are clustered into communities using the Louvain algorithm (via NetworkX):

1. **Detection:** Runs during `GraphStore.persist()` on undirected graph conversion
2. **Storage:** `community_id` stored per document in `communities.json`
3. **Boosting:** During search, results sharing a community with top-ranked documents receive a score multiplier (default 1.1×)

```python
# Community boost applied in orchestrator
boosts = graph.boost_by_community(doc_ids, seed_doc_ids, boost_factor=1.1)
```

**Configuration:**
- `community_detection_enabled`: Toggle community detection (default: true)
- `community_boost_factor`: Score multiplier (default: 1.1)

**Note:** The spec originally proposed Leiden algorithm. Implementation uses Louvain (available in NetworkX without additional dependencies). Both produce comparable community structures.

#### Score-Aware Fusion

Dynamic weight adjustment based on per-query score variance:

1. **Variance Calculation:** Compute variance of scores from each strategy
2. **Weight Adjustment:** Low variance indicates "muddy" matches; reduce that strategy's contribution
3. **Normalization:** Weights renormalized to maintain original sum

```python
# src/search/variance.py
if vector_variance < variance_threshold:
    vector_factor = max(min_weight_factor, vector_variance / variance_threshold)
```

**Configuration:**
- `dynamic_weights_enabled`: Toggle dynamic weights (default: true)
- `variance_threshold`: Threshold below which weights are reduced (default: 0.1)

#### HyDE (Hypothetical Document Embeddings)

Hypothesis-driven search for vague queries:

1. **Input:** User provides hypothesis describing expected documentation content
2. **Embedding:** Hypothesis text is embedded directly (no query expansion)
3. **Search:** Vector similarity search using hypothesis embedding
4. **Result:** Returns documents matching the hypothetical description

**Tool:** `search_with_hypothesis(hypothesis: str, top_n: int)`

**Use Case:** For queries like "How do I add a tool?", the AI generates a hypothesis: *"To add a tool, modify src/mcp_server.py and register in list_tools method..."* This finds relevant documentation even when the original query lacks specific terms.

**Configuration:**
- `hyde_enabled`: Toggle HyDE search (default: true)

#### Reciprocal Rank Fusion (RRF)

**Algorithm:**

For each document appearing in any ranked list:

```python
rrf_score = sum(1 / (k + rank) for rank in positions)
```

Where:
- `k` is a constant (default 60)
- `rank` is the document's position in each list (1-indexed)
- `positions` are all ranks where document appears across lists

**Example:**

Document appears at rank 3 in semantic search and rank 5 in keyword search:

```
rrf_score = 1/(60+3) + 1/(60+5) = 0.0159 + 0.0154 = 0.0313
```

**Weighting:**

Strategy weights multiply the RRF contribution:

```python
weighted_rrf_score = semantic_weight * (1/(k+rank_semantic))
                   + keyword_weight * (1/(k+rank_keyword))
```

**Final Ranking:**
1. Compute RRF scores for all documents
2. Apply recency bias multiplier
3. Sort by final score descending
4. Return top-k document IDs

#### Reciprocal Rank Fusion (RRF)

**Algorithm:**

### Indexing Flow

```
File Change Event
      ↓
FileWatcher detects change
      ↓
Event placed on queue.Queue
      ↓
Debouncing (500ms cooldown)
      ↓
Batch unique file paths
      ↓
For each file:
  ↓
  ParserRegistry.get_parser(file_path)
  ↓
  MarkdownParser.parse(file_path)
  ↓
  Document(content, metadata, links, tags)
  ↓
  IndexManager.index_document()
  ├─→ VectorIndex.add()
  ├─→ KeywordIndex.add()
  └─→ GraphStore.add_document()
      ↓
IndexManager.persist()
  ├─→ VectorIndex.persist()
  ├─→ KeywordIndex.persist()
  ├─→ GraphStore.persist()
  └─→ save_manifest()
```

### Query Flow

```
POST /query_documents
      ↓
QueryOrchestrator.query(query, top_k)
      ↓
Query Type Classification (if adaptive_weights_enabled)
      ↓
Query Expansion (concept vocabulary)
      ↓
Parallel execution:
  ├─→ VectorIndex.search(expanded_query, top_k)
  ├─→ KeywordIndex.search(query, top_k)
  └─→ CodeIndex.search(query, top_k) (if code_search_enabled)
      ↓
GraphStore.get_neighbors(candidate_docs, depth=1)
      ↓
Combine all ranked lists
      ↓
RRFFusion.fuse(ranked_lists, weights, k_constant)
      ↓
RecencyBias.apply(fused_results)
      ↓
Normalize scores [0.0, 1.0]
      ↓
Filter by min_confidence threshold
      ↓
Content hash deduplication (exact text match)
      ↓
N-gram deduplication (if ngram_dedup_enabled)
      ↓
Semantic deduplication (if dedup_enabled)
      ↓
MMR selection (if mmr_enabled) OR Per-doc limit
      ↓
Cross-encoder re-rank (if rerank_enabled)
      ↓
Parent expansion (if parent_retrieval_enabled)
      ↓
Top-n results
      ↓
Return list[ChunkResult]
```

## Key Design Decisions

### Pluggable Parsers

**Rationale:** Enable future support for non-Markdown formats (RST, AsciiDoc, plain text) without modifying core indexing logic.

**Implementation:** Registry pattern maps glob patterns to parser classes. Each parser implements `DocumentParser` protocol.

### Three-Index Hybrid Search

**Rationale:**
- Semantic search alone misses exact term matches (function names, error codes)
- Keyword search alone misses conceptually related content
- Graph traversal surfaces structurally related content that might not match query terms

**Trade-off:** Increased storage and indexing time for improved retrieval quality.

### In-Memory Graphs

**Rationale:** NetworkX provides fast graph operations (neighbor lookups, traversal) without external database dependency.

**Trade-off:** Graph must fit in memory. Acceptable for typical documentation collections (thousands of documents). Persistence to JSON enables reload on startup.

### Debounced File Watching

**Rationale:** Text editors trigger multiple file system events per save operation. Debouncing prevents redundant indexing.

**Implementation:** 500ms cooldown after last event. Events batched and deduplicated before processing.

### Manifest-Based Versioning

**Rationale:** Index format changes (embedding model upgrades, parser updates) require full rebuild to maintain consistency.

**Implementation:** Store index spec in manifest. Compare on startup. Trigger rebuild on mismatch.

### Local-First Architecture

**Rationale:** Eliminate external database dependencies. Simplify deployment and reduce operational complexity.

**Trade-offs:**
- Storage: All indices stored on local disk
- Scalability: Limited by local disk space and memory
- Concurrency: Single-server deployment (no distributed queries)

**Acceptable for:** Personal knowledge bases, project documentation, development environments.

### Self-Healing Index Infrastructure

**Rationale:** Local file-based indices are susceptible to corruption from crashes, disk errors, or incomplete writes. Rather than failing permanently, the system should detect corruption and recover automatically.

**Implementation:** Each index type implements a `_reinitialize_after_corruption()` method that recreates the index in a clean state. Corruption is detected during normal operations (load, search, remove) and triggers automatic recovery.

**Recovery Strategies by Index Type:**

| Index | Storage Format | Corruption Detection | Recovery Strategy |
|-------|---------------|---------------------|-------------------|
| **VectorIndex** | JSON mappings + FAISS binary | `json.JSONDecodeError` on load | Log warning, rebuild mapping from index or start empty |
| **KeywordIndex** | Whoosh segment files | `FileNotFoundError`, `OSError` on search/remove | Reinitialize to fresh in-memory index |
| **GraphStore** | JSON (`graph.json`, `communities.json`) | `json.JSONDecodeError`, `TypeError`, `KeyError` on load | Log warning, reinitialize empty graph |
| **CodeIndex** | Whoosh segment files | `FileNotFoundError`, `OSError` on search/remove | Reinitialize to fresh in-memory index |
| **CommitIndexer** | SQLite database | `DatabaseError`, malformed header detection | Delete DB file + WAL/SHM, recreate schema |

**Behavior Characteristics:**
- **Non-blocking:** Corruption in one index does not prevent other indices from functioning
- **Graceful degradation:** Search returns empty results rather than raising exceptions
- **Automatic reindexing:** Reconciliation will repopulate indices on next cycle
- **Logged recovery:** All corruption events logged at WARNING level with `exc_info=True`

**Example Recovery Flow (KeywordIndex):**

```
search("query") called
      ↓
self._index.searcher() raises OSError (segment file missing)
      ↓
Catch exception, log warning with context
      ↓
_reinitialize_after_corruption()
  ├─→ Clean up temp directory
  ├─→ Reset internal state
  └─→ Create fresh in-memory index
      ↓
Return empty results (graceful degradation)
      ↓
Next indexing operation repopulates index
```

**Design Trade-offs:**
- **Data loss on corruption:** Corrupt data is discarded rather than repaired. Acceptable because source documents remain intact and reconciliation will rebuild indices.
- **Silent recovery:** Users may not notice corruption occurred. Mitigated by logging and status endpoint reporting.
- **Performance impact:** Rebuilding indices after corruption adds latency. Acceptable for rare corruption events.

## Index Structure

### Storage Layout

```
{index_path}/
├── index.manifest.json
├── commits.db
├── vector/
│   ├── docstore.json
│   ├── index_store.json
│   ├── faiss_index.bin
│   ├── doc_id_mapping.json
│   ├── chunk_id_mapping.json
│   ├── concept_vocabulary.json
│   └── term_counts.json
├── keyword/
│   ├── MAIN_*.toc
│   ├── MAIN_*.seg
│   └── _MAIN_*.pos
├── code/
│   ├── MAIN_*.toc
│   ├── MAIN_*.seg
│   └── _MAIN_*.pos
└── graph/
    └── graph.json
```

### Manifest Format

```json
{
  "spec_version": "1.0.0",
  "embedding_model": "local",
  "parsers": {
    "**/*.md": "MarkdownParser",
    "**/*.markdown": "MarkdownParser"
  }
}
```

### Graph Format

```json
{
  "directed": true,
  "multigraph": false,
  "graph": {},
  "nodes": [
    {
      "id": "authentication",
      "title": "Authentication Guide",
      "tags": ["security", "api"],
      "aliases": ["auth", "credentials"]
    }
  ],
  "links": [
    {
      "source": "getting-started",
      "target": "authentication",
      "type": "link"
    }
  ]
}
```

## Performance Characteristics

### Indexing Performance

- Single document: ~200-500ms (includes embedding generation)
- Batch of 100 documents: ~30-60 seconds
- Bottleneck: Embedding model inference

### Query Performance

- Semantic search: ~50-100ms
- Keyword search: ~10-20ms
- Graph traversal: ~5-10ms
- Total query latency: ~100-150ms (excluding synthesis)

### Storage Requirements

- Vector index: ~1.5KB per document chunk
- Keyword index: ~500 bytes per document
- Graph: ~200 bytes per node + ~100 bytes per edge
- Example: 1000 documents ≈ 2-3MB total storage
