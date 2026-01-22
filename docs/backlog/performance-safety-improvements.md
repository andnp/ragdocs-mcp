# Performance, Safety, and Resiliency Improvements Backlog

Generated: 2026-01-20
Updated: 2026-01-22
Status: Nearly Complete

## Overview

This document tracks medium and lower priority improvements identified during the comprehensive code quality scan. High-priority and critical issues have already been addressed.

---

## Priority 3: Medium Priority (1-2 months)

### Issue #1: Add Retry Logic for Index Persistence (DONE)

**Category:** Resiliency - Lack of retry logic
**Location:** `src/indexing/manager.py:238-249`
**Impact:** MEDIUM - Transient failures cause data loss
**Status:** ✅ Fixed - `_persist_indices_with_retry()` uses tenacity `@retry` decorator

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

### Issue #2: Implement Circuit Breaker for Embedding Model (DONE)

**Category:** Resiliency - Cascading failure risks
**Location:** `src/indices/vector.py:47-52`, `src/utils/circuit_breaker.py`
**Impact:** MEDIUM - Repeated failures cascade to all queries
**Status:** ✅ Fixed - Custom `CircuitBreaker` class with configurable thresholds (failure_threshold=5, recovery_timeout=60s)

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

### Issue #3: Optimize Graph Traversal Locking (DONE)

**Category:** Performance - Inefficient data structures
**Location:** `src/indices/graph.py:79-99`
**Impact:** MEDIUM - Lock contention during traversal
**Status:** ✅ Fixed - Snapshot pattern implemented (shallow copy under lock, BFS without lock)

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

### Issue #4: Cache Query Embeddings (DONE)

**Category:** Performance - Redundant computations
**Location:** `src/search/orchestrator.py:45-48, 67-93`
**Impact:** MEDIUM - Repeated expensive operations
**Status:** ✅ Fixed - `_embedding_cache` dict with TTL (300s) and LRU eviction, `_get_cached_embedding()` method

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

### Issue #6: Add Timeout on Background Tasks (DONE)

**Category:** Safety - Missing timeout handling
**Location:** `src/context.py:286-297`
**Impact:** LOW - Tasks may hang indefinitely
**Status:** ✅ Fixed - `_index_git_commits_initial_with_timeout()` wraps git indexing with `asyncio.wait_for(timeout=30.0)`

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
**Location:** `src/search/dedup.py`
**Impact:** LOW - Potential O(n²) deduplication
**Status:** ⏳ Not completed - Still uses O(n²) pattern with nested loops

**Problem:**
Deduplication may use repeated list scans.

**Solution:**
Use set-based lookups for O(n) deduplication.

**Effort:** 2 hours (read code, benchmark, optimize if needed)
**Risk:** Low

---

### Issue #8: Add Resource Cleanup on Lifecycle Exception (DONE)

**Category:** Safety - Resource leaks
**Location:** `src/lifecycle.py:54-72`
**Impact:** LOW - Resources leak on startup exception
**Status:** ✅ Fixed - `start()` method has try/except that calls `await self._cleanup_resources()` on failure

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

### Issue #9: Optimize Stale Warning Deduplication (DONE)

**Category:** Performance - Minor memory leak
**Location:** `src/indices/vector.py:56-58, 286-293`
**Impact:** LOW
**Status:** ✅ Fixed - `OrderedDict` with LRU eviction (`_max_warned_chunks = 1000`)

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

### Issue #10: Cancel Emergency Timer on Normal Shutdown (DONE)

**Category:** Safety - Resource cleanup
**Location:** `src/lifecycle.py:117-125`
**Impact:** LOW
**Status:** ✅ Fixed - `_cancel_emergency_timer()` called at start of `shutdown()` and in `_cleanup_resources()`

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
- [x] P3-1: Retry logic for persistence
- [x] P3-3: Graph traversal locking optimization
- [x] P3-4: Query embedding cache

### Sprint 3 (Weeks 5-6)
- [x] P3-2: Circuit breaker for embedding model
- [x] P4-6: Background task timeouts
- [ ] P4-7: SearchPipeline optimization

### Sprint 4+ (Future)
- [x] P4-8: Resource cleanup on lifecycle exception
- [x] P5-9: Stale warning deduplication
- [x] P5-10: Cancel emergency timer

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

## Summary

**Completion Status:** 9 of 10 issues resolved (90%)

| Priority | Total | Completed | Remaining |
|----------|-------|-----------|-----------|
| P3 (Medium) | 5 | 5 | 0 |
| P4 (Low) | 3 | 2 | 1 (P4-7: SearchPipeline dedup) |
| P5 (Nice-to-have) | 2 | 2 | 0 |

**Last Updated:** 2026-01-22

---

## Notes

- All P1 and P2 issues resolved as of 2026-01-20
- P3, P4, P5 issues mostly resolved as of 2026-01-22
- Only P4-7 (SearchPipeline deduplication optimization) remains
- This backlog focuses on incremental improvements without breaking changes
