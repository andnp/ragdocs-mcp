# Architecture

This document describes the system architecture of mcp-markdown-ragdocs, including component responsibilities, data flow, and key design decisions.

## High-Level Architecture

The system consists of three primary subsystems with two transport modes:

1. **Indexing Service**: Monitors file changes and updates three distinct indices
2. **Query Orchestrator**: Executes parallel searches and fuses results
3. **Server Layer**: Exposes interfaces via stdio (MCP) or HTTP (REST API)

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

## Index Structure

### Storage Layout

```
{index_path}/
├── index.manifest.json
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
