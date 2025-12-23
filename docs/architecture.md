# Architecture

This document describes the system architecture of mcp-markdown-ragdocs, including component responsibilities, data flow, and key design decisions.

## High-Level Architecture

The system consists of three primary subsystems:

1. **Indexing Service**: Monitors file changes and updates three distinct indices
2. **Query Orchestrator**: Executes parallel searches and fuses results
3. **MCP Server**: Exposes the FastAPI HTTP interface

```
┌──────────────────────────────────────────────────────────────┐
│                      MCP Server (FastAPI)                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ /health    │  │ /status      │  │ /query_documents     │ │
│  └────────────┘  └──────────────┘  └──────────┬───────────┘ │
└──────────────────────────────────────────────┼──────────────┘
                                                │
                        ┌───────────────────────┴──────────────┐
                        ▼                                       ▼
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
        │  ┌──────────────────────┐ │
        │  │ GraphStore(NetworkX) │ │
        │  └──────────────────────┘ │
        │  ┌──────────────────────┐ │
        │  │ Manifest (JSON)      │ │
        │  └──────────────────────┘ │
        └───────────────────────────┘
```

## Component Overview

### Server (src/server.py)

**Responsibilities:**
- FastAPI application lifecycle management
- HTTP endpoint definitions
- Dependency injection for indices and orchestrator

**Endpoints:**
- `POST /query_documents`: Main query interface
- `GET /health`: Health check (returns `{"status": "ok"}`)
- `GET /status`: Operational status (document count, queue size, failed files)

**Lifecycle (lifespan context manager):**
1. Load configuration from TOML file
2. Initialize three indices (vector, keyword, graph)
3. Check index manifest for version changes
4. Index all documents if rebuild needed, otherwise load existing indices
5. Start file watcher
6. Yield control to server
7. On shutdown: stop watcher, persist indices

### Indexing Service

#### FileWatcher (src/indexing/watcher.py)

**Technology:** Python watchdog library

**Responsibilities:**
- Monitor configured documents_path for file system events
- Debounce rapid file changes (500ms cooldown)
- Queue events for batch processing
- Maintain list of failed files for status reporting

**Event Flow:**
1. Watchdog detects file creation/modification/deletion
2. Event placed on `queue.Queue` (thread-safe)
3. Background thread processes queue with debouncing
4. Unique file paths batched and passed to IndexManager
5. Failed operations logged and tracked

**Debouncing Strategy:**
- 500ms cooldown after last event before processing batch
- Multiple events for same file consolidated into single operation
- Thread-safe queue enables async coordination

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

**Purpose:** Detect configuration changes that require index rebuild

**IndexManifest Structure:**
```python
spec_version: str        # Index format version (e.g., "1.0.0")
embedding_model: str     # Embedding model identifier
parsers: dict[str, str]  # Glob pattern → parser class mapping
```

**Rebuild Logic:**

Rebuild triggered if:
- No saved manifest exists (first run or corrupted index)
- `spec_version` changed
- `embedding_model` changed
- `parsers` configuration changed

**Storage:** `{index_path}/index.manifest.json`

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
- Apply 1-hop graph neighbor boosting
- Fuse results using Reciprocal Rank Fusion
- Apply recency bias
- Synthesize answer from top-ranked chunks

#### Search Strategies

**1. Semantic Search (VectorIndex):**
- Embeds query using same model as documents
- Cosine similarity search in FAISS index
- Returns document IDs ranked by similarity

**2. Keyword Search (KeywordIndex):**
- BM25F scoring across content, aliases, tags
- Returns document IDs ranked by term relevance

**3. Graph Traversal (GraphStore):**
- 1-hop neighbor boosting: for each candidate document from semantic/keyword search, retrieve all directly linked documents
- Neighbor documents added to result pool with reduced score (0.5x multiplier)
- Surfaces structurally related content

**4. Recency Bias:**
- Tier-based score multiplier:
  - Last 7 days: 1.2x
  - Last 30 days: 1.1x
  - Over 30 days: 1.0x
- Applied during fusion stage

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

#### Synthesis

**Current Implementation:** Mock LLM that echoes query (for testing)

**Production Pattern:**
1. Retrieve top-k document IDs from fusion
2. Load full content for each document
3. Pass query + context to LLM
4. Return synthesized answer

## Data Flow

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
Parallel execution:
  ├─→ VectorIndex.search(query, top_k)
  ├─→ KeywordIndex.search(query, top_k)
  └─→ (Results from above)
      ↓
GraphStore.get_neighbors(candidate_docs, depth=1)
      ↓
Combine all ranked lists
      ↓
RecencyBias.apply(ranked_lists)
      ↓
RRFFusion.fuse(ranked_lists, weights, k_constant)
      ↓
Top-k document IDs
      ↓
QueryOrchestrator.synthesize_answer(query, doc_ids)
      ↓
Return answer string
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
│   └── doc_id_mapping.json
├── keyword/
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
