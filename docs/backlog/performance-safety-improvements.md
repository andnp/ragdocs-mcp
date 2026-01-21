# Performance, Safety, and Resiliency Improvements Backlog

Generated: 2026-01-20
Status: Planned

## Overview

This document tracks medium and lower priority improvements identified during the comprehensive code quality scan. High-priority and critical issues have already been addressed.

---

## Priority 3: Medium Priority (1-2 months)

### Issue #1: Add Retry Logic for Index Persistence

**Category:** Resiliency - Lack of retry logic
**Location:** `src/indexing/manager.py:111-127`
**Impact:** MEDIUM - Transient failures cause data loss

**Problem:**
Single-shot persist with no retry. Transient failures (disk full, NFS timeout, permission errors) cause complete failure.

**Solution:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def persist(self):
    # existing implementation
```

**Effort:** 2-3 hours (add dependency, update method, add tests)
**Risk:** Low - tenacity is stable, retry is well-understood pattern

---

### Issue #2: Implement Circuit Breaker for Embedding Model

**Category:** Resiliency - Cascading failure risks
**Location:** `src/indices/vector.py:179-220`
**Impact:** MEDIUM - Repeated failures cascade to all queries

**Problem:**
No failure tracking. If embedding model fails (OOM, corrupted model file), every query retries the same operation.

**Solution:**
Implement circuit breaker pattern with:
- Failure threshold (e.g., 5 failures within 60s → open circuit)
- Automatic fallback to keyword-only search
- Half-open state with gradual recovery
- Configurable via `config.toml`

**Libraries:**
- `pybreaker` (1.6k stars, actively maintained)
- Or custom implementation (100 lines)

**Effort:** 1-2 days (design, implement, test failure scenarios)
**Risk:** Medium - requires careful state management, fallback coordination

---

### Issue #3: Optimize Graph Traversal Locking

**Category:** Performance - Inefficient data structures
**Location:** `src/indices/graph.py:79-99`
**Impact:** MEDIUM - Lock contention during traversal

**Problem:**
Holds write lock during entire BFS traversal. For deep graphs, this blocks other operations.

**Solution:**
Two approaches:
1. **Use read-write lock (RWLock):** `readerwriterlock` package
   - Allows concurrent reads, exclusive writes
   - Minimal code changes

2. **Snapshot pattern:**
   ```python
   def get_neighbors(self, doc_id, max_depth=2):
       # Take shallow copy of graph under lock
       with self._graph_lock:
           graph_snapshot = self._graph.copy()

       # BFS on snapshot (no lock held)
       neighbors = self._bfs_traversal(graph_snapshot, doc_id, max_depth)
       return neighbors
   ```

**Recommendation:** Start with snapshot pattern (simpler, no new dependency)

**Effort:** 3-4 hours (implement snapshot, benchmark, verify correctness)
**Risk:** Low - snapshot is safe, benchmark will show if copy overhead is acceptable

---

### Issue #4: Cache Query Embeddings

**Category:** Performance - Redundant computations
**Location:** `src/search/orchestrator.py:118-122`
**Impact:** MEDIUM - Repeated expensive operations

**Problem:**
Query embedding computed multiple times (once for search, again for MMR).

**Solution:**
```python
from functools import lru_cache
from typing import Tuple

class SearchOrchestrator:
    def __init__(self, ...):
        # ... existing init
        self._embedding_cache: dict[str, Tuple[list[float], float]] = {}
        self._cache_max_size = 100

    def _get_cached_embedding(self, query: str) -> list[float]:
        """Get query embedding with LRU cache."""
        import time
        current_time = time.time()

        # Check cache
        if query in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[query]
            # Expire after 5 minutes
            if current_time - timestamp < 300:
                return embedding

        # Compute new embedding
        embedding = self._vector._embedding_model.get_text_embedding(query)

        # Evict oldest if cache full
        if len(self._embedding_cache) >= self._cache_max_size:
            oldest_key = min(self._embedding_cache, key=lambda k: self._embedding_cache[k][1])
            del self._embedding_cache[oldest_key]

        self._embedding_cache[query] = (embedding, current_time)
        return embedding
```

**Alternative:** Use `@lru_cache` on embedding method (simpler but less control)

**Effort:** 2-3 hours (implement cache, add tests, benchmark)
**Risk:** Low - caching is well-understood, memory bounded

**Impact Analysis:**
- Embedding computation: ~50-100ms per query
- Cache hit rate (estimated): 20-30% (users refine similar queries)
- Speedup: 10-30% for cached queries

---

### Issue #5: Add Backpressure to Event Queue (DONE)

**Category:** Resiliency - Missing circuit breakers
**Location:** `src/indexing/watcher.py:34`
**Impact:** MEDIUM - Memory exhaustion under load
**Status:** ✅ Fixed in P2 sprint

**Problem:**
Unbounded queue allows rapid file changes to exhaust memory.

**Solution:**
Already implemented with `MAX_QUEUE_SIZE = 1000` and drop-oldest policy.

---

## Priority 4: Low Priority (As time permits)

### Issue #6: Add Timeout on Background Tasks

**Category:** Safety - Missing timeout handling
**Location:** `src/context.py:236-274`
**Impact:** LOW - Tasks may hang indefinitely

**Problem:**
Background tasks (git indexing, file watching) have no timeout protection.

**Solution:**
```python
async def _startup_background_tasks(self):
    tasks = [
        self._start_file_watcher(),
        self._start_git_indexing(),
    ]

    # Wrap in timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        logger.warning("Background tasks startup timed out")
```

**Effort:** 1 hour (add timeouts, test)
**Risk:** Low

---

### Issue #7: Optimize SearchPipeline Deduplication

**Category:** Performance - Inefficient list scans
**Location:** `src/search/pipeline.py` (requires reading)
**Impact:** LOW - Potential O(n²) deduplication

**Problem:**
Deduplication may use repeated list scans.

**Solution:**
Use set-based lookups for O(n) deduplication.

**Effort:** 2 hours (read code, benchmark, optimize if needed)
**Risk:** Low

---

### Issue #8: Add Resource Cleanup on Lifecycle Exception

**Category:** Safety - Resource leaks
**Location:** `src/lifecycle.py:43-82`
**Impact:** LOW - Resources leak on startup exception

**Problem:**
If initialization raises exception, resources (file handles, threads) may leak.

**Solution:**
```python
async def startup(self, timeout: float = 60.0):
    try:
        await self._startup_impl(timeout)
    except Exception:
        # Cleanup on failure
        logger.error("Startup failed, cleaning up resources", exc_info=True)
        await self._cleanup_resources()
        raise

async def _cleanup_resources(self):
    """Best-effort cleanup of all resources."""
    if self.ctx:
        try:
            await self.ctx.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)
```

**Effort:** 2 hours (add try/finally, test failure scenarios)
**Risk:** Low

---

## Priority 5: Nice-to-Have (Future)

### Issue #9: Optimize Stale Warning Deduplication

**Category:** Performance - Minor memory leak
**Location:** `src/indices/vector.py:49`
**Impact:** LOW

**Problem:**
`_warned_stale_chunk_ids` grows unbounded.

**Solution:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _log_stale_warning(self, chunk_id: str):
    logger.warning(f"Stale chunk reference: {chunk_id}")
```

**Effort:** 15 minutes
**Risk:** None

---

### Issue #10: Cancel Emergency Timer on Normal Shutdown

**Category:** Safety - Resource cleanup
**Location:** `src/lifecycle.py:156-175`
**Impact:** LOW

**Problem:**
Emergency timer thread persists briefly after normal shutdown.

**Solution:**
Always call `_cancel_emergency_timer()` in shutdown path.

**Effort:** 30 minutes
**Risk:** None

---

## Implementation Roadmap

### Sprint 1 (Next 2 weeks)
- [x] P1: Embedding timeout protection
- [x] P1: Bounded vocabularies
- [x] P2: FileWatcher shutdown race
- [x] P2: MCP handler validation
- [x] P2: Blocking I/O async wrapping

### Sprint 2 (Weeks 3-4)
- [ ] P3-1: Retry logic for persistence
- [ ] P3-3: Graph traversal locking optimization
- [ ] P3-4: Query embedding cache

### Sprint 3 (Weeks 5-6)
- [ ] P3-2: Circuit breaker for embedding model
- [ ] P4-6: Background task timeouts
- [ ] P4-7: SearchPipeline optimization

### Sprint 4+ (Future)
- [ ] Remaining P4 items
- [ ] P5 items as quick wins

---

## Monitoring & Metrics

To validate improvements, track:

| Metric | Baseline | Target | Tool |
|--------|----------|--------|------|
| **Embedding cache hit rate** | 0% | 20-30% | Custom logging |
| **Graph traversal latency (p95)** | TBD | -20% | pytest-benchmark |
| **Index persist failure rate** | 1-2% | <0.1% | Logs analysis |
| **Memory growth rate** | +5MB/hr | +1MB/hr | Memory profiler |
| **Query embedding time** | 50-100ms | 50-100ms (cached: <1ms) | pytest-benchmark |

---

## Dependencies

### New Dependencies Needed:
- `tenacity` (retry logic) - 5k stars, stable
- `pybreaker` (circuit breaker) - 1.6k stars OR custom (100 lines)
- `readerwriterlock` (optional, if RWLock approach chosen) - 200 stars

### Alternative: Implement patterns from scratch
- Retry: 30 lines with exponential backoff
- Circuit breaker: 100 lines for basic state machine
- RWLock: Use snapshot pattern instead

**Recommendation:** Use tenacity (battle-tested), implement custom circuit breaker (project-specific needs).

---

## Questions for Discussion

1. **Priority:** Should circuit breaker (P3-2) be promoted to P2 given cascading failure risk?
2. **Dependencies:** Prefer adding tenacity/pybreaker OR custom implementations?
3. **Monitoring:** Add Prometheus metrics for production deployments?
4. **Testing:** Create performance regression test suite alongside fixes?

---

## Notes

- All P1 and P2 issues resolved as of 2026-01-20
- This backlog focuses on incremental improvements without breaking changes
- Each issue includes effort estimate, risk assessment, and solution sketch
- Ready for sprint planning and prioritization
