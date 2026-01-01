# 8. Document Chunking

## 8.1. Overview

The document chunking system splits markdown files into locally relevant regions based on header structure. Instead of retrieving and synthesizing from entire documents, the system returns focused chunks that match the query, reducing context window usage and improving answer precision.

**Rationale:** Full-document retrieval often includes irrelevant sections. Header-based chunking respects markdown structure while maintaining semantic coherence within each chunk.

## 8.2. Architecture

### 8.2.1. Chunking Pipeline

Documents flow through the chunking pipeline during indexing:

1. Parser extracts full document content and metadata
2. Chunker splits document based on headers and size constraints
3. Each chunk stored in vector and keyword indices
4. Graph index retains document-level relationships

### 8.2.2. Component Integration

**IndexManager** coordinates chunking:
- Instantiates chunker from factory based on config
- Applies chunking to each parsed document
- Distributes chunks to vector and keyword indices
- Preserves document-level metadata for graph

**VectorIndex** stores chunk embeddings:
- Maintains `chunk_id → node_id` and `doc_id → [node_ids]` mappings
- Returns chunk-level results with metadata
- Persists chunk mappings alongside FAISS index

**KeywordIndex** indexes chunk text:
- Schema includes both `chunk_id` (primary) and `doc_id` (reference)
- BM25F scoring operates on chunk content
- Returns chunk-level results

**QueryOrchestrator** synthesizes from chunks:
- Receives chunk IDs from hybrid search
- Retrieves chunk content (not full documents)
- Passes focused chunks to LLM synthesizer

## 8.3. Chunking Algorithm

### 8.3.1. Header-Based Strategy

The `HeaderBasedChunker` uses tree-sitter AST parsing to split documents:

**Process:**
1. Extract header hierarchy from markdown AST (h1-h6)
2. Create initial chunks at header boundaries
3. Merge chunks below `min_chunk_chars`
4. Split chunks above `max_chunk_chars` at paragraph boundaries
5. Apply `overlap_chars` between adjacent chunks

**Header Hierarchy Preservation:**
- Each chunk includes its full header path ("Architecture > Components")
- Header paths stored in chunk metadata for context
- Parent headers included in chunk content when configured

**Fallback Behavior:**
- Documents without headers: split at paragraph boundaries
- Single-paragraph documents: stored as single chunk

### 8.3.2. Size Constraints

**Parameters:**
- `min_chunk_chars`: 200 (prevents context-poor fragments)
- `max_chunk_chars`: 2000 (maintains focus, fits token limits)
- `overlap_chars`: 50 (preserves context between sibling chunks only)

**Merge Logic:**
- Adjacent chunks below minimum combined until threshold met
- Preserves header hierarchy in merged chunks

**Split Logic:**
- Large sections split at paragraph boundaries (double newline)
- Maintains readability by avoiding mid-paragraph splits
- Each sub-chunk tagged with `_sub_N` suffix (e.g., `doc1_chunk_0_sub_0`)

**Overlap Logic:**
- Applied **only** between sibling sub-chunks (force-split from same parent)
- NOT applied between header-based chunks (semantically distinct sections)
- Last `overlap_chars` characters from chunk N prepended to chunk N+1

**Sibling Detection Algorithm:**
```python
def _apply_overlap(chunks, overlap_chars):
    if overlap_chars <= 0:
        return chunks

    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
            continue

        prev_chunk = chunks[i - 1]

        # Check if current and previous are siblings (same parent, both sub-chunks)
        current_is_subchunk = "_sub_" in chunk.chunk_id
        prev_is_subchunk = "_sub_" in prev_chunk.chunk_id

        if current_is_subchunk and prev_is_subchunk:
            # Extract parent IDs by removing "_sub_N" suffix
            current_parent = chunk.chunk_id.rsplit("_sub_", 1)[0]
            prev_parent = prev_chunk.chunk_id.rsplit("_sub_", 1)[0]

            if current_parent == prev_parent:
                # Siblings: prepend overlap from previous chunk
                overlap_text = prev_chunk.content[-overlap_chars:]
                chunk.content = overlap_text + "\n\n" + chunk.content

        result.append(chunk)

    return result
```

**Rationale:**
- Header-based chunks represent distinct semantic sections (different topics)
- Overlap between header sections would pollute semantic boundaries
- Force-split siblings from same section need context restoration
- Sibling relationship identified via `_sub_` suffix in chunk_id

## 8.4. Data Model

### 8.4.1. Chunk Structure

```python
@dataclass
class Chunk:
    chunk_id: str          # {doc_id}_chunk_{index}
    doc_id: str            # Parent document ID
    content: str           # Chunk text (headers + content)
    metadata: dict         # Inherited from document + chunk-specific
    chunk_index: int       # Position in parent document (0-based)
    header_path: str       # "Section > Subsection > Detail"
    start_pos: int         # Character offset in original document
    end_pos: int           # Character offset in original document
    file_path: str         # Absolute path to source file
    modified_time: datetime # Document modification timestamp
```

### 8.4.2. Metadata Inheritance

Each chunk inherits document metadata:
- `tags`: Markdown tags from document
- `links`: Wikilinks within chunk content
- `aliases`: Document aliases from frontmatter
- Custom frontmatter fields preserved

Chunk-specific metadata added:
- `chunk_id`: Unique identifier
- `doc_id`: Parent document reference
- `chunk_index`: Sequential position
- `header_path`: Header hierarchy

## 8.5. Index Integration

### 8.5.1. Vector Index

**Storage:**
- Each chunk embedded separately via `HuggingFaceEmbedding`
- FAISS stores chunk vectors with metadata
- Mappings: `chunk_id → node_id`, `doc_id → [chunk_node_ids]`

**Search:**
- Query embedded and searched against chunk vectors
- Returns list of dicts: `{chunk_id, doc_id, score, content, metadata}`
- Top-k chunks selected by cosine similarity

**Persistence:**
- FAISS index saved to disk
- Chunk mappings serialized alongside index
- Reload reconstructs mappings from stored data

### 8.5.2. Keyword Index

**Schema:**
```python
Schema(
    id=ID(stored=True, unique=True),      # chunk_id
    doc_id=ID(stored=True),                # parent doc_id
    content=TEXT(stored=False),            # chunk text
    tags=KEYWORD(stored=False, commas=True)
)
```

**Indexing:**
- Whoosh inverted index built from chunk content
- BM25F scoring across content and tags fields
- Thread-safe with Lock for concurrent access

**Search:**
- MultifieldParser searches content, aliases, tags
- Returns list: `{chunk_id, doc_id, score}`

### 8.5.3. Graph Index

**Document-Level Relationships:**
- Graph retains document nodes (not chunk nodes)
- Edges represent wikilinks between documents
- 1-hop neighbor lookup augments chunk-based search
- Graph traversal adds related document chunks to result pool

## 8.6. Configuration

### 8.6.1. ChunkingConfig Parameters

```toml
[chunking]
strategy = "header_based"     # "header_based" or "none"
min_chunk_chars = 200         # Merge threshold
max_chunk_chars = 1500        # Split threshold
overlap_chars = 50            # Context overlap
include_parent_headers = true # Include full header path
```

**Strategy Options:**
- `"header_based"`: Tree-sitter AST header extraction
- `"none"`: Disable chunking (full documents)

**Tuning Guidelines:**
- Increase `min_chunk_chars` for more context per chunk
- Decrease `max_chunk_chars` for more focused retrieval
- Increase `overlap_chars` for better inter-chunk continuity

### 8.6.2. Config Integration

`ChunkingConfig` integrated into main config:
```python
@dataclass
class Config:
    server: ServerConfig
    indexing: IndexingConfig
    parsers: dict[str, str]
    search: SearchConfig
    llm: LLMConfig
    chunking: ChunkingConfig  # Chunking parameters
```

Loaded from TOML via `load_config()`:
- Defaults applied if `[chunking]` section omitted
- Strong defaults enable zero-config operation

## 8.7. Query Flow

### 8.7.1. Search Phase

**Hybrid Search with Chunks:**
1. Semantic search returns top-k chunks by embedding similarity
2. Keyword search returns top-k chunks by BM25F score
3. Graph traversal adds chunks from linked documents
4. Recency bias applied to chunk scores based on document modification time
5. RRF fusion merges results into ranked chunk list

### 8.7.2. Synthesis Phase

**Chunk-Based Answer Generation:**
1. `QueryOrchestrator.query()` returns fused chunk IDs
2. `synthesize_answer()` retrieves chunk content (not full documents)
3. Chunks passed to LLM as `TextNode` objects with metadata
4. Synthesizer generates answer from focused chunk content

**Advantages:**
- Reduced context window usage (chunks vs full docs)
- Higher precision (relevant sections only)
- Faster synthesis (less LLM input)

## 8.8. Manifest Versioning

### 8.8.1. Rebuild Triggers

The `IndexManifest` tracks chunking configuration:
```python
@dataclass
class IndexManifest:
    spec_version: str
    embedding_model: str
    parsers: dict[str, str]
    chunking_config: dict[str, Any]  # Chunking parameters
```

**Rebuild Conditions:**
- `chunking_config` changed (strategy, min/max/overlap values)
- Embedding model changed
- Parser configuration changed

**Rebuild Process:**
- On startup, compare saved manifest to current config
- If mismatch detected, clear indices and re-index all documents
- New manifest saved with current config

### 8.8.2. Backward Compatibility

**Migration from Pre-Chunking Indices:**
- Old manifests lack `chunking_config` field
- Missing field triggers rebuild with current chunking config
- No manual migration required

**Disabling Chunking:**
- Set `strategy = "none"` in config
- Triggers rebuild, indices revert to full documents

## 8.9. Testing

### 8.9.1. Unit Tests

**Header Extraction:**
- Nested header hierarchy (h1 > h2 > h3)
- Malformed headers and edge cases
- Documents without headers

**Size Constraints:**
- Merging chunks below `min_chunk_chars`
- Splitting chunks above `max_chunk_chars`
- Overlap application between chunks

**Metadata:**
- Tag and link inheritance from documents
- Header path construction
- Chunk ID generation

### 8.9.2. Integration Tests

**End-to-End Pipeline:**
- Parse → Chunk → Index → Search → Synthesize
- Chunk-based search vs full-document search
- RRF fusion with chunk results

**Index Persistence:**
- Save and load vector index with chunk mappings
- Whoosh schema compatibility
- Manifest versioning triggers rebuild

### 8.9.3. Performance Tests

**Benchmarks:**
- Chunking overhead on indexing speed (<30% increase)
- Query latency with chunk retrieval (<200ms)
- Index size comparison (chunks vs full documents)

**Results (173/174 tests passing):**
- 0 ruff errors
- 0 pyright errors
- 99.4% test pass rate

## 8.10. Implementation Notes

**Files Created:**
- `src/chunking/__init__.py` - Package exports
- `src/chunking/base.py` - `ChunkingStrategy` protocol
- `src/chunking/header_chunker.py` - Header-based implementation
- `src/chunking/factory.py` - Chunker factory function

**Files Modified:**
- `src/models.py` - Added `Chunk` dataclass
- `src/config.py` - Added `ChunkingConfig`
- `src/indexing/manager.py` - Integrated chunker into pipeline
- `src/indexing/manifest.py` - Added `chunking_config` field
- `src/indices/vector.py` - Added `add_chunk()`, chunk mappings
- `src/indices/keyword.py` - Updated schema, added `add_chunk()`
- `src/search/orchestrator.py` - Chunk-based synthesis

**Design Decisions:**
- Chunks stored in indices (not graph) for performance
- Tree-sitter AST ensures accurate header extraction
- Automatic rebuild on config change via manifest
- Document-level graph preserves structural relationships

**Detailed Implementation Reference:**
See [CHUNKING_IMPLEMENTATION_PLAN.md](../CHUNKING_IMPLEMENTATION_PLAN.md) for original implementation roadmap and technical details.
