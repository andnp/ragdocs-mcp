# ADR-022: Memory System Independence from Document Snapshots

## Status
Accepted

## Context

A bug was discovered where memory search returned 0 results in multiprocess mode despite memory files existing on disk. Investigation revealed:

1. **Document indices** follow a snapshot-based architecture:
   - Worker process indexes documents → persists to `snapshot_base/vN/`
   - Main process loads from snapshots via `IndexSyncReceiver`

2. **Memory indices** follow a direct-persistence architecture:
   - Main process manages memories → persists to `memory_path/indices/`
   - No snapshot synchronization needed (single owner)

3. **The bug**: `ReadOnlyContext` was looking for memory indices in the document snapshot directory, which never contained memory indices because the worker doesn't manage memories.

## Decision

### Principle: Memory System is Process-Local to Main Process

Memory management is explicitly **not** part of the multiprocess worker architecture. The main process (MCP server) is the sole owner of memory state because:

1. Memory operations (create/update/delete) are user-facing and require immediate consistency
2. Memory indexing is lightweight compared to document indexing
3. No need for snapshot-based synchronization between processes

### Code Organization

| Component | Owner | Index Path | Notes |
|-----------|-------|------------|-------|
| Documents | Worker Process | `snapshot_base/vN/{vector,keyword,graph}` | Snapshot-based sync |
| Memories | Main Process | `memory_path/indices/{vector,keyword,graph}` | Direct persistence |
| Git Commits | Worker Process (indexing) / Main Process (query) | `index_path/git_commits.db` | SQLite, read-only in main |

### Implementation Requirements

1. **Path Resolution**: Memory index path is always `memory_path/indices/`, never `snapshot_base`
2. **Loading**: Both `ApplicationContext` and `ReadOnlyContext` must load memory indices from `memory_path/indices/`
3. **Persistence**: Memory manager persists directly, never through snapshot publisher

## Consequences

### Positive
- Clear separation of concerns between document and memory systems
- Simpler memory architecture (no cross-process sync needed)
- Memory operations have predictable latency

### Negative  
- Memory indexing happens in main process, consuming event loop time
- Memory indices not versioned (no snapshot rollback)

### Mitigation
- Memory files are typically small (<1KB), so indexing is fast
- Memory operations use `asyncio.to_thread()` to avoid blocking

## Testing Requirements

Regression tests must verify:
1. Memory indices load correctly in `ReadOnlyContext` (multiprocess mode)
2. Memory search returns results after context creation
3. New memories are detected via reconciliation
4. Memory and document systems are independent (document snapshots don't affect memory)

See `tests/regression/test_memory_multiprocess.py` for implementation.
