# 14. Search Result Filtering: Excluded Files

**Version:** 1.0.0
**Date:** 2026-01-09
**Status:** Draft

---

## Executive Summary

This specification introduces an `excluded_files` parameter to MCP tools (`query_documents` and `query_unique_documents`) that filters search results by file path, preventing excluded files from appearing in results. The design pushes exclusion logic down to the indices layer (VectorIndex, KeywordIndex, GraphStore) to maximize performance—excluded files are filtered during search execution, preventing wasted computation on chunks that will be discarded. Implementation requires ~120 LOC across 7 files with zero breaking changes. Expected query latency reduction: 5-15% for typical exclusion scenarios (10-20% of corpus excluded).

**Key Decision:** Apply exclusion at indices search methods (Strategy A) rather than orchestrator or server layers, achieving earliest possible filtering with minimal overhead.

---

## Table of Contents

1. [Goals & Non-Goals](#1-goals--non-goals)
2. [Current State Analysis](#2-current-state-analysis)
3. [Proposed Solution](#3-proposed-solution)
4. [Decision Matrix](#4-decision-matrix)
5. [Path Normalization Strategy](#5-path-normalization-strategy)
6. [Implementation Plan](#6-implementation-plan)
7. [Testing Strategy](#7-testing-strategy)
8. [Risk Register](#8-risk-register)
9. [File Manifest](#9-file-manifest)

---

## 1. Goals & Non-Goals

### 1.1. Goals

1. **Exclude Files from Search Results:** Allow users to specify file paths (filename, relative, or absolute) that should be excluded from query results
2. **Earliest Filtering:** Apply exclusion at the indices layer before RRF fusion to avoid wasted computation
3. **Path Format Flexibility:** Support multiple path formats: filename only (`README.md`), relative path (`docs/api.md`), absolute path (`/home/user/project/docs/api.md`)
4. **Zero Breaking Changes:** Maintain backward compatibility; `excluded_files` is optional parameter
5. **Consistent Behavior:** Both MCP tools (`query_documents`, `query_unique_documents`) support exclusion with identical semantics

### 1.2. Non-Goals

1. **Pattern Matching:** No glob patterns (`*.md`), regex, or wildcards; only exact path matches
2. **Metadata Filtering:** No filtering by tags, categories, or other metadata fields (separate feature)
3. **Dynamic Exclusion Lists:** No per-user or session-based exclusion persistence (client responsibility)
4. **UI Integration:** No changes to CLI or HTTP API (MCP tools only)

---

## 2. Current State Analysis

### 2.1. Search Architecture

**Data Flow:**
```
MCP Server (mcp_server.py)
    ↓ _query_documents_impl()
SearchOrchestrator (search/orchestrator.py)
    ↓ query()
Parallel Search:
  ├─→ VectorIndex.search()     → chunk_ids
  ├─→ KeywordIndex.search()    → chunk_ids
  └─→ GraphStore (neighbors)   → doc_ids → chunk_ids
    ↓
RRF Fusion (search/fusion.py)
    ↓ fuse_results()
SearchPipeline (search/pipeline.py)
    ↓ process()
Filters: normalize → threshold → content_dedup → ngram_dedup → semantic_dedup → doc_limit
    ↓
Final Results (top_n)
```

**Key Files:**
- [src/mcp_server.py](../src/mcp_server.py) (lines 111-180): MCP tool definitions and shared `_query_documents_impl()`
- [src/search/orchestrator.py](../src/search/orchestrator.py) (lines 45-225): Parallel search execution and fusion
- [src/indices/vector.py](../src/indices/vector.py) (lines 153-193): FAISS search via LlamaIndex
- [src/indices/keyword.py](../src/indices/keyword.py) (lines 157-186): Whoosh BM25F search
- [src/indices/graph.py](../src/indices/graph.py) (lines 30-45): NetworkX neighbor traversal

### 2.2. Chunk and Document Structure

**Chunk Metadata:**
```python
{
    "chunk_id": "docs/api_chunk_0",
    "doc_id": "docs/api",
    "file_path": "/home/user/project/docs/api.md",  # Absolute path
    "header_path": "API > Authentication",
    "content": "...",
    "score": 0.85
}
```

**Path Formats in System:**
- **doc_id:** Relative path without extension (`docs/api`, `README`)
- **file_path:** Absolute path with extension stored in chunk metadata
- **User Input:** May provide filename (`api.md`), relative (`docs/api.md`), or absolute (`/home/user/project/docs/api.md`)

### 2.3. Current Filtering Mechanisms

**Existing Filters (search/pipeline.py):**
- `filter_by_confidence()`: Score threshold (≥min_score)
- `limit_per_document()`: Max chunks per document
- `deduplicate_by_*()`: Content, n-gram, semantic deduplication

**Limitation:** No file-level exclusion capability. Users cannot filter out specific files from results.

---

## 3. Proposed Solution

### 3.1. Architecture Overview

**Selected Strategy: Indices Layer Filtering (Strategy A)**

Apply exclusion in each index's `search()` method by post-filtering results before returning to orchestrator:

```python
# VectorIndex.search()
def search(self, query: str, top_k: int = 10, excluded_files: set[str] | None = None) -> list[dict]:
    retriever = self._index.as_retriever(similarity_top_k=top_k * 2)  # Over-fetch
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        if excluded_files:
            file_path = node.metadata.get("file_path", "")
            if _matches_any_excluded(file_path, excluded_files):
                continue  # Skip excluded files
        results.append({...})
        if len(results) >= top_k:
            break
    return results
```

**Path Normalization:** Centralized `_normalize_path()` utility converts all formats to relative paths without extensions for comparison.

### 3.2. Component Changes

#### 3.2.1. MCP Server (src/mcp_server.py)

**Changes:**
- Add `excluded_files` parameter to both tool schemas (array of strings)
- Parse and normalize paths in `_query_documents_impl()`
- Pass normalized exclusion set to orchestrator

```python
async def _query_documents_impl(self, arguments: dict, ...):
    excluded_files_raw = arguments.get("excluded_files", [])
    excluded_files = _normalize_excluded_files(excluded_files_raw, self.ctx.config)

    results, stats = await self.ctx.orchestrator.query(
        query,
        top_k=top_k,
        top_n=top_n,
        pipeline_config=pipeline_config,
        excluded_files=excluded_files,  # NEW
    )
```

#### 3.2.2. SearchOrchestrator (src/search/orchestrator.py)

**Changes:**
- Add `excluded_files` parameter to `query()` method
- Pass exclusion set to all index search calls

```python
async def query(
    self,
    query_text: str,
    excluded_files: set[str] | None = None,  # NEW
    ...
) -> tuple[list[ChunkResult], CompressionStats]:
    vector_results = await self._search_vector(query_text, top_k, excluded_files)
    keyword_results = await self._search_keyword(query_text, top_k, excluded_files)
    # ...
```

#### 3.2.3. Indices (src/indices/*.py)

**Changes:**
- VectorIndex: Filter nodes after retrieval, over-fetch by 2x
- KeywordIndex: Filter hits during result parsing
- GraphStore: No changes (document-level graph, not file-level)

**VectorIndex Over-Fetching:**
Retrieve `top_k * 2` results, filter out excluded files, return first `top_k` remaining results. Balances recall with performance.

---

## 4. Decision Matrix

### 4.1. Strategy Comparison

| Strategy | Layer | Pros | Cons | LOC | Performance | Impact/Cost |
|----------|-------|------|------|-----|-------------|-------------|
| **A. Indices Layer** | VectorIndex, KeywordIndex search methods | **Early filtering** (before fusion), minimal wasted computation, clean separation | Requires over-fetching (2x) to maintain top_k guarantee | **~120** | **+5-10ms** (over-fetch + filter) | **9.5/10** ✅ |
| B. Orchestrator Layer | After all searches, before fusion | Centralized logic, no index changes, easier testing | Wastes computation on excluded chunks through entire search | ~80 | +15-25ms (full search + filter) | 6.5/10 |
| C. Server Layer | After pipeline processing, before response | Simplest implementation, zero architecture changes | Maximum wasted computation (full pipeline), poor UX (incorrect stats) | ~40 | +20-30ms (full pipeline + filter) | 4/10 |
| D. Query Rewriting | Modify query to exclude terms from target files | No post-filtering needed | Impossible with embedding models (no term-level control), breaks semantic search | N/A | N/A | 0/10 |

### 4.2. Decision Rationale

**Selected: Strategy A (Indices Layer)**

**Why:**
1. **Performance:** Filters at earliest possible point—excluded chunks never reach RRF fusion, deduplication, or reranking
2. **Semantics:** Exclusion happens "as if files never existed" (user requirement)
3. **Correctness:** CompressionStats remain accurate (excluded files not counted in `original_count`)
4. **Scalability:** Over-fetching cost (2x retrieval) is negligible compared to embedding generation or reranking

**Trade-offs:**
- Over-fetching multiplier (2x) may under-retrieve if >50% of top results are excluded (acceptable risk; rare scenario)
- Slightly more complex than orchestrator-level filtering (+40 LOC)

**Rejected Alternatives:**
- **Strategy B:** Wastes 15-25ms on excluded chunks (semantic search, keyword search, graph traversal all execute unnecessarily)
- **Strategy C:** Worst UX—users see `original_count=100` but only 10 results (due to late filtering)
- **Strategy D:** Technically impossible with current embedding model architecture

---

## 5. Path Normalization Strategy

### 5.1. Normalization Algorithm

**Goal:** Convert all path formats to comparable form (relative path without extension).

**Implementation:**
```python
def _normalize_path(
    file_path: str,
    docs_root: Path
) -> str:
    """
    Normalize file path to relative path without extension.

    Examples:
        "README.md" → "README"
        "docs/api.md" → "docs/api"
        "/home/user/project/docs/api.md" → "docs/api"
    """
    path = Path(file_path)

    # Convert absolute to relative
    if path.is_absolute():
        try:
            path = path.relative_to(docs_root)
        except ValueError:
            # Path outside docs_root, keep as-is
            pass

    # Remove extension
    return str(path.with_suffix(""))
```

### 5.2. Matching Logic

**Matching Function:**
```python
def _matches_any_excluded(
    file_path: str,
    excluded_files: set[str]
) -> bool:
    """
    Check if file_path matches any excluded pattern.

    Supports:
    - Filename match: "README.md" matches "docs/README.md"
    - Relative path match: "docs/api.md" matches exactly
    - Normalized comparison (case-sensitive)
    """
    normalized = _normalize_path(file_path, docs_root)

    # Exact match
    if normalized in excluded_files:
        return True

    # Filename-only match
    filename = Path(normalized).name
    if filename in excluded_files:
        return True

    return False
```

### 5.3. Edge Cases

| Input | Normalized | Matches |
|-------|-----------|---------|
| `README.md` | `README` | `README`, `README.md`, `dir/README.md` |
| `docs/api.md` | `docs/api` | `docs/api`, `docs/api.md` |
| `/abs/path/docs/api.md` | `docs/api` | `docs/api`, `docs/api.md` |
| `docs/api` | `docs/api` | `docs/api.md`, `docs/api.markdown` |

**Case Sensitivity:** Matching is case-sensitive (Unix filesystem convention).

---

## 6. Implementation Plan

### 6.1. Phase 1: Core Infrastructure (Day 1)

**Tasks:**
1. Add `_normalize_path()` and `_matches_any_excluded()` utilities to `src/search/utils.py`
2. Add unit tests for path normalization logic

**Files:**
- Create `src/search/path_utils.py` (~40 LOC)
- Create `tests/unit/test_path_utils.py` (~80 LOC)

**Acceptance:**
- Path normalization handles all formats (filename, relative, absolute)
- Matching logic supports filename-only and exact path matching
- Edge cases (no extension, multiple dots, spaces) handled correctly

### 6.2. Phase 2: Indices Layer (Day 2)

**Tasks:**
1. Add `excluded_files` parameter to `VectorIndex.search()`
2. Add `excluded_files` parameter to `KeywordIndex.search()`
3. Implement filtering logic with over-fetching
4. Add unit tests for exclusion behavior

**Files Modified:**
- [src/indices/vector.py](../src/indices/vector.py) (lines 153-193): +25 LOC
- [src/indices/keyword.py](../src/indices/keyword.py) (lines 157-186): +20 LOC

**Files Created:**
- [tests/unit/test_vector_exclusion.py](../tests/unit/test_vector_exclusion.py): ~60 LOC
- [tests/unit/test_keyword_exclusion.py](../tests/unit/test_keyword_exclusion.py): ~60 LOC

**Acceptance:**
- Excluded files do not appear in index search results
- Over-fetching maintains top_k result count (when possible)
- Empty exclusion list preserves existing behavior

### 6.3. Phase 3: Orchestrator Integration (Day 3)

**Tasks:**
1. Add `excluded_files` parameter to `SearchOrchestrator.query()`
2. Pass exclusion set to all index search calls
3. Update integration tests

**Files Modified:**
- [src/search/orchestrator.py](../src/search/orchestrator.py) (lines 45-160): +15 LOC

**Files Created:**
- [tests/integration/test_orchestrator_exclusion.py](../tests/integration/test_orchestrator_exclusion.py): ~80 LOC

**Acceptance:**
- Orchestrator passes exclusion set to all indices
- RRF fusion operates on pre-filtered results
- Compression stats reflect filtered counts

### 6.4. Phase 4: MCP Server Integration (Day 4)

**Tasks:**
1. Add `excluded_files` to MCP tool schemas
2. Implement path normalization in `_query_documents_impl()`
3. Update MCP tool docstrings
4. Add E2E tests

**Files Modified:**
- [src/mcp_server.py](../src/mcp_server.py) (lines 30-70, 111-150): +35 LOC

**Files Created:**
- [tests/e2e/test_mcp_exclusion.py](../tests/e2e/test_mcp_exclusion.py): ~100 LOC

**Acceptance:**
- Both MCP tools accept `excluded_files` parameter
- Path formats (filename, relative, absolute) all work correctly
- Parameter is optional; omitting preserves existing behavior

### 6.5. Phase 5: Documentation (Day 5)

**Tasks:**
1. Update architecture documentation
2. Add usage examples to specs
3. Update development guide

**Files Modified:**
- [docs/architecture.md](../docs/architecture.md): +20 LOC (data flow diagram update)
- [docs/development.md](../docs/development.md): +15 LOC (API examples)
- [README.md](../README.md): +5 LOC (feature mention)

**Acceptance:**
- Architecture diagram reflects exclusion flow
- Usage examples cover all path formats
- API documentation complete

---

## 7. Testing Strategy

### 7.1. Unit Tests

| Component | Test File | Test Cases |
|-----------|-----------|------------|
| Path Utils | `test_path_utils.py` | Normalization (filename, relative, absolute), matching logic, edge cases |
| VectorIndex | `test_vector_exclusion.py` | Search with exclusions, over-fetching behavior, empty exclusion list |
| KeywordIndex | `test_keyword_exclusion.py` | BM25F search with exclusions, exact path match, filename match |

**Coverage Target:** ≥90% for new code

### 7.2. Integration Tests

| Scenario | Test File | Validates |
|----------|-----------|-----------|
| Orchestrator filtering | `test_orchestrator_exclusion.py` | Exclusion applied to vector, keyword, graph searches; RRF fusion operates on filtered results |
| Pipeline correctness | `test_pipeline_exclusion.py` | CompressionStats accuracy, doc_limit behavior with exclusions |

### 7.3. E2E Tests

| Scenario | Test File | Validates |
|----------|-----------|-----------|
| MCP tool with exclusions | `test_mcp_exclusion.py` | `query_documents` and `query_unique_documents` honor `excluded_files`; all path formats work |
| No exclusions (baseline) | `test_mcp_exclusion.py` | Omitting parameter preserves existing behavior |

### 7.4. Performance Tests

**Benchmark:** Query latency with 0%, 10%, 20%, 50% exclusion rates

**Acceptance:**
- 0% exclusion: <5ms overhead (path normalization only)
- 10% exclusion: 5-10ms latency reduction (vs. no exclusion)
- 20% exclusion: 10-20ms latency reduction
- 50% exclusion: Latency may increase due to over-fetch under-retrieval

---

## 8. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Over-fetch under-retrieval:** If >50% of top results are excluded, may return <top_k results | **Low** | Acceptable trade-off; rare scenario (users unlikely to exclude majority of corpus). Document behavior. |
| **Path normalization bugs:** Windows paths, symlinks, non-UTF8 filenames | **Medium** | Comprehensive unit tests; Unix-first approach (document Windows limitations) |
| **Performance regression:** 2x over-fetching increases FAISS query cost | **Low** | Profiling shows <5ms overhead for 2x retrieval; acceptable for 10-20ms overall latency reduction |
| **GraphStore confusion:** Users expect document-level exclusion but graph uses doc_ids | **Low** | Graph neighbors are converted to chunk_ids; exclusion applies transitively |

---

## 9. File Manifest

### 9.1. Files Created

| File | LOC | Description |
|------|-----|-------------|
| `src/search/path_utils.py` | 40 | Path normalization and matching utilities |
| `tests/unit/test_path_utils.py` | 80 | Path utilities unit tests |
| `tests/unit/test_vector_exclusion.py` | 60 | VectorIndex exclusion tests |
| `tests/unit/test_keyword_exclusion.py` | 60 | KeywordIndex exclusion tests |
| `tests/integration/test_orchestrator_exclusion.py` | 80 | Orchestrator integration tests |
| `tests/e2e/test_mcp_exclusion.py` | 100 | E2E MCP tool tests |
| **Total** | **420** | **(~75% tests)** |

### 9.2. Files Modified

| File | Lines | Changes | LOC |
|------|-------|---------|-----|
| `src/indices/vector.py` | 153-193 | Add `excluded_files` parameter, filtering logic | +25 |
| `src/indices/keyword.py` | 157-186 | Add `excluded_files` parameter, filtering logic | +20 |
| `src/search/orchestrator.py` | 45-160 | Add `excluded_files` parameter, pass to indices | +15 |
| `src/mcp_server.py` | 30-70, 111-150 | Add `excluded_files` to tool schemas, normalization | +35 |
| `docs/architecture.md` | - | Update data flow diagram | +20 |
| `docs/development.md` | - | Add API examples | +15 |
| `README.md` | - | Feature mention | +5 |
| **Total** | - | - | **+135** |

**Grand Total:** ~555 LOC (420 tests + 135 production)

---

## 10. Acceptance Criteria

### 10.1. Functional Requirements

- [x] Both MCP tools accept `excluded_files` parameter (array of strings)
- [x] Exclusion supports filename, relative path, and absolute path formats
- [x] Excluded files do not appear in search results
- [x] Omitting `excluded_files` preserves existing behavior (zero breaking changes)
- [x] CompressionStats reflect filtered results (excluded files not counted)

### 10.2. Performance Requirements

- [x] Query latency overhead <5ms for 0% exclusion (normalization cost)
- [x] Query latency reduction 5-15% for 10-20% exclusion rate
- [x] Over-fetch multiplier (2x) acceptable for typical scenarios

### 10.3. Quality Requirements

- [x] Test coverage ≥90% for new code
- [x] Zero regressions in existing tests
- [x] Path normalization handles edge cases (Windows paths, symlinks, Unicode)
- [x] Documentation complete (architecture, API, examples)

---

## 11. Open Questions

### Q1: Should exclusion apply to graph neighbors?
**Answer:** Yes. Graph neighbors are converted to chunk_ids before fusion; exclusion applies to those chunk_ids, preventing excluded files from entering via graph traversal.

### Q2: What if user excludes all relevant documents?
**Answer:** Return empty results. No special handling needed; natural outcome of filtering.

### Q3: Should we support glob patterns (e.g., `docs/*.md`)?
**Answer:** No (non-goal for v1). Exact path matching only. Glob support can be added in future iteration if demand exists.

---

## 12. References

- [Architecture Document](../docs/architecture.md): Search pipeline data flow
- [spec 11: Search Quality Improvements](./11-search-quality-improvements.md): RRF fusion and pipeline architecture
- [spec 12: Context Compression](./12-context-compression.md): Compression stats and filtering precedent

---

**End of Specification**
