# 19. Self-Healing Index Infrastructure

**Version:** 1.0.0
**Date:** 2026-01-20
**Status:** Implemented

---

## Executive Summary

**Purpose:** Document the self-healing architecture that enables automatic recovery from index corruption across all storage backends (FAISS, Whoosh, NetworkX, SQLite).

**Scope:** All index types implement corruption detection during normal operations (load, search, remove) and automatic recovery via reinitialization. This applies to VectorIndex, KeywordIndex, GraphStore, CodeIndex, and CommitIndexer.

**Decision:** Detect corruption at operation boundaries and reinitialize to a clean state rather than attempting repair. Source documents remain intact, so reconciliation rebuilds indices automatically.

---

## 1. Goals & Non-Goals

### Goals

1. **Graceful Degradation:** Return empty results rather than raising exceptions on corruption
2. **Automatic Recovery:** Reinitialize indices without manual intervention
3. **Non-Blocking:** Corruption in one index does not prevent other indices from functioning
4. **Observable:** Log all corruption events with context for debugging
5. **Zero Data Loss (Documents):** Original source files remain untouched; only derived indices are affected

### Non-Goals

1. **Data Repair:** No attempt to fix corrupted data structures (too complex, error-prone)
2. **Transactional Guarantees:** No ACID semantics for index persistence
3. **Hot Standby:** No redundant index copies for instant failover
4. **Corruption Prevention:** Focus is recovery, not prevention (separate concern)

---

## 2. Current State Analysis

### 2.1. Corruption Sources

| Source | Description | Affected Indices |
|--------|-------------|-----------------|
| **Incomplete writes** | Crash during persist, power loss | All |
| **Disk errors** | Bad sectors, filesystem corruption | All |
| **Version mismatch** | Schema changes between releases | Whoosh, SQLite |
| **External modification** | User/tool editing index files | All |
| **Resource exhaustion** | Out of disk space during write | All |

### 2.2. Index Storage Formats

| Index | Format | Corruption Indicators |
|-------|--------|----------------------|
| VectorIndex | JSON (mappings) + FAISS binary | `json.JSONDecodeError` |
| KeywordIndex | Whoosh segments (`.seg`, `.toc`, `.pos`) | `FileNotFoundError`, `OSError` |
| GraphStore | JSON (`graph.json`, `communities.json`) | `json.JSONDecodeError`, `TypeError` |
| CodeIndex | Whoosh segments | `FileNotFoundError`, `OSError` |
| CommitIndexer | SQLite (`commits.db`) | `DatabaseError`, "malformed" header |

---

## 3. Proposed Solution

### 3.1. Detection Strategy

Corruption is detected at **operation boundaries** where indices interact with persistent storage:

```
┌─────────────────────────────────────────────────────────────┐
│                     Operation Flow                          │
├─────────────────────────────────────────────────────────────┤
│  load()  ─┬─▶ Read file from disk                          │
│           └─▶ Parse/deserialize         ◀── DETECT HERE    │
│                                                             │
│  search() ─┬─▶ Open index reader                           │
│            └─▶ Execute query            ◀── DETECT HERE    │
│                                                             │
│  remove() ─┬─▶ Open index writer                           │
│            └─▶ Delete entry             ◀── DETECT HERE    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2. Recovery Pattern

All indices implement a `_reinitialize_after_corruption()` method:

```python
def _reinitialize_after_corruption(self) -> None:
    """Reset index to clean state after corruption detection."""
    # 1. Log warning with context
    logger.warning(f"Index corruption detected. Reinitializing.")

    # 2. Clean up resources (close connections, delete temp files)
    self._cleanup_resources()

    # 3. Reset internal state
    self._reset_state()

    # 4. Initialize fresh index
    self._initialize_index()
```

### 3.3. Implementation Per Index Type

#### VectorIndex (FAISS/LlamaIndex)

**Corruption Detection:** `json.JSONDecodeError` when loading mapping files

**Recovery:** Log warning, continue with empty mappings. Index remains usable but loses doc_id ↔ node_id associations until rebuilt.

```python
# src/indices/vector.py
mapping_file = path / "doc_id_mapping.json"
if mapping_file.exists():
    try:
        with open(mapping_file, "r") as f:
            self._doc_id_to_node_ids = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to load doc_id mapping (corrupted JSON): {e}")
        logger.info("Rebuilding doc_id mapping from index")
        self._doc_id_to_node_ids = {}
```

**Trade-off:** Partial recovery (index works but mappings lost) vs. full reinit (simpler but loses all vectors).

#### KeywordIndex (Whoosh)

**Corruption Detection:** `FileNotFoundError`, `OSError` when opening searcher or writer

**Recovery:** Reinitialize to fresh in-memory index, discarding corrupted segments.

```python
# src/indices/keyword.py
def search(self, query: str, top_k: int = 10) -> list[dict]:
    try:
        searcher = self._index.searcher(weighting=BM25F())
    except (FileNotFoundError, OSError) as e:
        logger.warning(
            f"Keyword index corruption detected during search: {e}. "
            "Reinitializing index and returning empty results."
        )
        self._reinitialize_after_corruption()
        return []
```

**Trade-off:** All indexed documents lost until reconciliation rebuilds them.

#### GraphStore (NetworkX)

**Corruption Detection:** `json.JSONDecodeError`, `TypeError`, `KeyError`, `AttributeError` during JSON parsing

**Recovery:** Reinitialize empty directed graph. Separate handling for `graph.json` (critical) vs `communities.json` (optional).

```python
# src/indices/graph.py
try:
    with open(graph_file, "r") as f:
        graph_data = json.load(f)
    self._graph = nx.node_link_graph(graph_data, directed=True)
except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
    logger.warning(
        f"Graph index corruption detected (malformed graph.json): {e}. "
        "Reinitializing graph.",
        exc_info=True,
    )
    self._reinitialize_after_corruption()
```

**Trade-off:** Community detection results lost; must re-run during next persist.

#### CodeIndex (Whoosh)

**Corruption Detection:** Same as KeywordIndex (Whoosh segment corruption)

**Recovery:** Same pattern as KeywordIndex—reinitialize fresh index.

```python
# src/indices/code.py
def _reinitialize_after_corruption(self) -> None:
    if self._index_path and self._index_path in _temp_dirs:
        shutil.rmtree(self._index_path, ignore_errors=True)
        _temp_dirs.discard(self._index_path)
    self._index = None
    self._index_path = None
    self._initialize_index()
```

#### CommitIndexer (SQLite)

**Corruption Detection:** SQLite error messages matching corruption patterns

**Recovery:** Delete corrupted database file (+ WAL/SHM), recreate schema

```python
# src/git/commit_indexer.py
SQLITE_CORRUPTION_PATTERNS = (
    "database disk image is malformed",
    "disk I/O error",
    "unable to open database file",
    "database is locked",
    "file is not a database",
)

def _reinitialize_after_corruption(self) -> None:
    logger.warning(
        f"SQLite database corruption detected at {self._db_path}. "
        "Recreating database (commits will be re-indexed)."
    )
    # Close connection
    if self._conn is not None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None

    # Delete corrupted files
    if self._db_path.exists():
        self._db_path.unlink()
    for suffix in [".db-wal", ".db-shm"]:
        wal_path = self._db_path.with_suffix(suffix)
        if wal_path.exists():
            wal_path.unlink()

    # Recreate schema
    self._ensure_schema()
```

**Trade-off:** All commit history lost until git watcher triggers re-indexing.

---

## 4. Decision Matrix

| Option | Complexity | Data Preservation | Recovery Speed | Risk |
|--------|:----------:|:-----------------:|:--------------:|:----:|
| **Reinitialize on corruption (chosen)** | Low | None (derived data) | Fast | Low |
| Repair corrupted structures | High | Partial | Slow | High |
| Redundant index copies | Medium | Full | Fast | Medium |
| External backup/restore | Medium | Full | Slow | Low |

**Decision:** Reinitialize on corruption. Source documents are the authoritative data; indices are derived and can be rebuilt. Repair is too complex and error-prone. Redundancy adds storage overhead.

---

## 5. Testing Strategy

### Unit Tests

| Test File | Test Cases |
|-----------|------------|
| `tests/unit/test_vector_corruption.py` | `test_load_with_corrupted_*`, `test_persist_recovers_from_partial_write` |
| `tests/unit/test_keyword_index.py` | `test_search_handles_corrupted_segment`, `test_remove_handles_corrupted_segment`, `test_recovery_allows_reindexing` |
| `tests/unit/test_graph_corruption.py` | `test_load_with_corrupted_graph_json`, `test_load_with_corrupted_communities_json`, `test_recovery_allows_new_nodes` |
| `tests/unit/test_code_index_corruption.py` | `test_search_handles_corrupted_segment`, `test_remove_handles_corrupted_segment`, `test_recovery_allows_reindexing` |
| `tests/unit/test_commit_indexer.py` | `test_corrupted_database_triggers_recovery`, `test_recovery_allows_reindexing`, `test_query_on_corrupted_db_returns_empty` |

### Integration Tests

| Test File | Test Cases |
|-----------|------------|
| `tests/integration/test_index_manager_resilience.py` | `test_remove_document_continues_on_partial_failure`, `test_startup_reconciliation_survives_corrupted_keyword_index` |

### Test Pattern

Tests follow a consistent structure:

1. Create index with valid data
2. Persist to disk
3. Corrupt the persisted files (delete segments, write garbage)
4. Trigger operation that detects corruption
5. Verify graceful degradation (empty results, no exception)
6. Verify subsequent operations succeed (recovery complete)

---

## 6. Observability

### Logging

All corruption events logged at WARNING level with `exc_info=True`:

```
WARNING - Keyword index corruption detected during search: [Errno 2] No such file or directory: '.../MAIN_abc123.seg'. Reinitializing index and returning empty results.
```

### Status Endpoint

The `/status` endpoint reports index health:

```json
{
  "status": "ok",
  "document_count": 42,
  "failed_files": [],
  "indices": {
    "vector": {"loaded": true, "doc_count": 42},
    "keyword": {"loaded": true, "doc_count": 42},
    "graph": {"loaded": true, "node_count": 42}
  }
}
```

After corruption recovery, counts may temporarily drop to 0 until reconciliation rebuilds the index.

---

## 7. Implementation Files

| File | Role |
|------|------|
| [src/indices/vector.py](../../src/indices/vector.py) | VectorIndex corruption handling in `load()` |
| [src/indices/keyword.py](../../src/indices/keyword.py) | KeywordIndex corruption handling in `search()`, `remove()` |
| [src/indices/graph.py](../../src/indices/graph.py) | GraphStore corruption handling in `load()` |
| [src/indices/code.py](../../src/indices/code.py) | CodeIndex corruption handling in `search()`, `remove()` |
| [src/git/commit_indexer.py](../../src/git/commit_indexer.py) | CommitIndexer corruption handling with SQLite pattern matching |

---

## 8. Architecture Decision Records

### ADR-1: Reinitialize vs. Repair

**Status:** Accepted

**Context:** When index corruption is detected, the system must decide whether to attempt repair or reinitialize from scratch.

**Decision:** Reinitialize to a clean state, discarding corrupted data.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Reinitialize (selected)** | Simple, reliable, fast recovery | All derived data lost |
| Attempt repair | Preserves some data | Complex, error-prone, may propagate corruption |
| Fail permanently | No risk of bad data | Requires manual intervention |

**Rationale:** Indices are derived from source documents. Discarding corrupted indices and rebuilding via reconciliation is simpler and more reliable than repair. The minor cost (temporary empty results) is acceptable for a local-first system.

### ADR-2: Detection at Operation Boundaries

**Status:** Accepted

**Context:** Corruption could be detected proactively (health checks) or reactively (when operations fail).

**Decision:** Detect corruption reactively at operation boundaries (load, search, remove).

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Reactive detection (selected)** | Simple, no overhead, covers real failures | May not catch silent corruption |
| Proactive health checks | Catches issues before queries | Adds complexity, performance overhead |
| Checksums/integrity verification | Cryptographic guarantees | Significant implementation effort |

**Rationale:** Reactive detection catches all corruption that affects operations. Proactive checks add complexity for marginal benefit in a local-first system where the user is the only operator.

### ADR-3: Graceful Degradation vs. Fast Failure

**Status:** Accepted

**Context:** When corruption is detected during search, should the system return empty results or raise an exception?

**Decision:** Return empty results (graceful degradation).

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Empty results (selected)** | Non-blocking, search continues | User may not notice issue |
| Raise exception | Explicit failure, user informed | Blocks workflow, requires handling |
| Return partial results | Best effort | Complex, may return inconsistent data |

**Rationale:** For an AI assistant use case, empty results are preferable to exceptions. The assistant can fall back to other strategies or inform the user. Corruption is logged for observability.

---

## 9. References

1. **Implementation Files:**
   - [src/indices/vector.py](../../src/indices/vector.py) - VectorIndex
   - [src/indices/keyword.py](../../src/indices/keyword.py) - KeywordIndex
   - [src/indices/graph.py](../../src/indices/graph.py) - GraphStore
   - [src/indices/code.py](../../src/indices/code.py) - CodeIndex
   - [src/git/commit_indexer.py](../../src/git/commit_indexer.py) - CommitIndexer

2. **Test Files:**
   - [tests/unit/test_vector_corruption.py](../../tests/unit/test_vector_corruption.py)
   - [tests/unit/test_keyword_index.py](../../tests/unit/test_keyword_index.py)
   - [tests/unit/test_graph_corruption.py](../../tests/unit/test_graph_corruption.py)
   - [tests/unit/test_code_index_corruption.py](../../tests/unit/test_code_index_corruption.py)
   - [tests/unit/test_commit_indexer.py](../../tests/unit/test_commit_indexer.py)
   - [tests/integration/test_index_manager_resilience.py](../../tests/integration/test_index_manager_resilience.py)

3. **Architecture:**
   - [docs/architecture.md](../architecture.md) - Self-Healing Index Infrastructure section
