# Plan: Progressive Semantic Indexing Architecture

**Status:** Implemented and validated — 2026-08-01
**Date:** 2026-07-30
**Related:** `docs/architecture.md`, `searchkernel/context.py`, `searchkernel/indexing/bootstrap_session.py`, `searchkernel/indexing/manager.py`, `searchkernel/indexing/runtime_readiness.py`

## Executive summary

The first-time indexing path should become progressively useful instead of waiting for every embedding to finish before serving queries.

The main architectural change is to separate:

1. preparing document data
2. applying lexical and graph index stages
3. planning semantic work
4. reusing or generating embeddings
5. publishing readiness and progress

The design should reuse the existing `BootstrapSession`, task queue, readiness model, and index implementations. It should not create a second startup state machine or put cache and scheduling policy into `VectorIndex`.

The faster-model/backend track is intentionally out of scope for this plan.

The shared coordinator, checkpoint recovery, cache-aware semantic planning,
progressive readiness, and application bootstrap integration are complete.
Hosted performance benchmarks remain in the scheduled/manual CI job rather
than the mandatory correctness workflow.

## Evidence and motivation

The guarded fresh-index benchmark against `~/Projects/rlcore/obsidian` measured:

| Stage | Time |
| --- | ---: |
| Total startup | 517.8 s |
| Bulk indexing | 511.2 s |
| Vocabulary catch-up | 3.8 s |
| Persistence | 3.2 s |
| Reranker warmup | 1.3 s |
| File discovery | 0.05 s |

The corpus produced about 6,733 chunks and 6,309 unique embedding texts. A direct sample showed that embedding inference dominates bulk indexing. The same run peaked at about 1.05 GiB RSS under a 3 GiB process-group limit.

The watcher probe found one scheduled recursive root and one inotify descriptor, but 1,387 underlying Linux directory watch entries. Application-level root count and kernel watch count must therefore remain separate metrics.

## Design goals

- Make lexical and graph search useful as soon as those stages are complete.
- Make semantic indexing resumable, observable, and independently replaceable.
- Reuse one semantic work path for cache hits, deduplication, coarse indexing, and fine indexing.
- Keep memory bounded by processing prepared batches rather than the whole corpus.
- Preserve exact chunk identity, metadata, move detection, and existing search behavior once fully indexed.
- Make cache invalidation explicit and fail closed when the embedding configuration changes.
- Keep lifecycle orchestration in the existing bootstrap/runtime layer.

## Non-goals

- Choosing a faster embedding model or inference backend.
- Replacing FAISS, Whoosh, NetworkX, or the current persistence layer in the first implementation.
- Making partially indexed search indistinguishable from fully indexed search.
- Adding a second daemon, worker, or startup coordinator.

## Current coupling to remove

`IndexManager.index_documents()` currently prepares documents in batches and applies all indexing stages at once (keyword, graph, vector). This makes it difficult to:

- publish lexical/graph readiness before embeddings finish
- checkpoint semantic work separately from file work
- reuse embeddings across files or rebuilds
- bound memory by batch
- measure each stage independently

`VectorIndex` should continue to own vector storage and retrieval. It should not own embedding cache policy, bootstrap scheduling, or tier selection.

## Target data flow

```text
discover files
    │
    ▼
prepare bounded batches
    │
    ├── lexical stage ──┐
    ├── graph stage ────┼──► publish partial/queryable state
    │                   │
    └── semantic inputs ┘
             │
             ▼
      semantic work planner
             │
       ┌─────┴─────┐
       │           │
   cache hits   cache misses
                   │
             deduplicate requests
                   │
             bounded embedding
                   │
             vector materialization
                   │
             checkpoint + persist
```

## Proposed module boundaries

### Prepared batches

Extract a behavior-preserving preparation layer via `searchkernel/indexing/stages.py`.

```python
@dataclass(frozen=True)
class PreparedIndexBatch:
    documents: list[PreparedIndexDocument]
    chunks: list[Chunk]
    graph_nodes: list[tuple[str, dict]]
    graph_edges: list[tuple[str, str, str, str]]
    semantic_inputs: list[SemanticInput]
```

The batch builder should emit bounded batches of files/chunks. It should centralize construction of embedding text so cache keys and vector insertion use exactly the same bytes.

### Index stages

Introduce small stage contracts rather than a large inheritance tree via `searchkernel/indexing/stages.py`:

```python
class IndexStage(Protocol):
    name: str

    def apply(self, batch: PreparedIndexBatch) -> StageResult: ...
```

Initial stages:

- `KeywordStage`
- `GraphStage`
- `SemanticStage`
- existing manifest/hash finalization stage

`IndexBuildCoordinator` (future) should execute these stages over batches. Existing single-document and watcher paths can initially use the same stage helpers without switching all behavior at once.

### Semantic work

Use one semantic work model for every semantic optimization via `searchkernel/indexing/semantic.py`:

```python
@dataclass(frozen=True)
class SemanticInput:
    source_id: str
    text: str
    content_hash: str
    tier: Literal["coarse", "fine"]
    priority: int
```

The `SemanticWorkPlanner` should:

1. canonicalize the embedding text
2. compute the content hash
3. group identical inputs
4. look up cache hits in bulk
5. submit only unique misses to the encoder
6. write each resulting vector to all assigned source IDs
7. persist progress for the completed work unit

This keeps cache lookup, duplicate suppression, coarse/fine priority, and backfill scheduling in one reusable path.

## Embedding cache

Add an `EmbeddingCache` protocol with a persistent SQLite implementation via `searchkernel/indexing/embedding_cache.py`.

The cache namespace must include an encoder fingerprint containing at least:

- model name and version identifier
- normalization settings
- query/text instruction settings
- embedding dimension

The cache key must hash the canonical embedding text, including any header text that is actually passed to the model. The existing `ChunkHashStore` should not be extended for this purpose: it tracks chunk mutation identity, not reusable model output.

Cache requirements:

- bulk `get_many()` and `put_many()` operations
- atomic writes
- corruption recovery or safe cache discard
- namespace mismatch treated as a miss
- metrics for hits, misses, writes, and invalidations
- a persistence location that survives index rebuilds

## Partial readiness and backfill

Reuse `BootstrapSession` for discovery, checkpoint monitoring, task submission, and lifecycle ownership. Add semantic progress to the existing public state via `searchkernel/indexing/runtime_readiness.py` rather than creating a separate readiness system.

The runtime should expose capabilities such as:

```text
lexical: available | unavailable
graph: available | unavailable
semantic_coarse: available | backfilling | complete
semantic_fine: available | backfilling | complete
```

The existing distinction between `can_serve_queries()` and `is_fully_ready()` should remain:

- `can_serve_queries()` becomes true when the minimum lexical/graph serving contract is met.
- `is_fully_ready()` becomes true only when required semantic tiers and persistence are complete.

Search should receive a `SearchAvailability` value and adjust fusion based on available channels. This avoids scattered readiness checks and prevents an incomplete vector index from being presented as complete.

Semantic backfill should use the existing worker/task architecture, with bounded work units and durable checkpoints. A restart should reload completed lexical/graph work and resume only unfinished semantic work.

## Coarse-to-fine indexing

The first semantic tier should use deterministic, inexpensive representations such as:

- document title
- header hierarchy
- a bounded prefix or extract

The fine tier should use the existing full chunk embedding text.

Both tiers should flow through `SemanticWorkPlanner`. The implementation may use separate vector segments or tier metadata, but callers should see one semantic-index abstraction.

Coarse results must map back to document/chunk IDs without duplicating metadata or search-result formatting. Fine results should supersede coarse results when available.

This tier is a quality-sensitive optimization and should be enabled only after comparison against a fixed query-quality set.

## Persistence and recovery

Separate file completion from semantic completion via extended `searchkernel/indexing/bootstrap_checkpoint.py`. A file may be:

1. discovered
2. parsed and chunked
3. present in keyword/graph indices
4. pending semantic work
5. partially semantically indexed
6. fully indexed

The durable checkpoint model should record stage completion and the encoder namespace used for semantic work. A completed lexical stage must not imply completed vector work.

Batch persistence should be atomic from the perspective of readers. If a semantic batch fails, the system should retain the prior queryable state, report the failed batch, and retry or leave an explicit partial state rather than silently dropping vectors.

## Implementation sequence

### Phase 0 — architecture seams, no behavior change

- Extract bounded preparation via `stages.py`.
- Extract keyword, graph, and semantic stage helpers.
- Add stage-level timing and counters.
- Keep the existing synchronous ordering as the default execution mode.

### Phase 1 — semantic work planner

- Add `SemanticInput`, planner, and encoder/materializer contracts via `semantic.py`.
- Add in-batch and cross-batch duplicate grouping.
- Add tests with fake encoders and fake caches.

### Phase 2 — persistent cache

- Implement SQLite `EmbeddingCache` via `embedding_cache.py`.
- Add encoder namespace and canonical text hashing.
- Add cache hit/miss metrics and invalidation tests.
- Verify moved files and rebuilds reuse vectors.

### Phase 3 — progressive startup

- Commit lexical/graph batches before semantic completion.
- Publish partial readiness through the existing bootstrap state path.
- Schedule semantic backfill through the existing worker/task system.
- Make restart and checkpoint behavior explicit.

### Phase 4 — coarse/fine semantic tiers

- Add deterministic coarse representations.
- Prioritize coarse work before fine body chunks.
- Update search availability and fusion policy.
- Compare quality and latency before making the tier default.

### Phase 5 — hardening and rollout

- Test interruption, retry, deletion, moves, cache corruption, and encoder changes.
- Add bounded-memory and process-restart tests.
- Run guarded cold, warm-cache, partial-readiness, and fully-ready Obsidian benchmarks.
- Document runtime status and operator-visible progress.

Each phase should be a small conventional commit and remain independently testable.

## Success criteria

- Lexical/graph search becomes queryable without waiting for full embedding completion.
- Rebuilding unchanged content performs near-zero new embedding work.
- Duplicate embedding requests are removed without changing result identity or metadata.
- Backfill resumes after restart without rebuilding completed work.
- Partial semantic coverage is visible to operators and search policy.
- Peak RSS remains bounded by configured batch limits.
- Fully indexed search quality is unchanged on the existing regression suite.
- Cold and warm Obsidian benchmark results report stage times, cache behavior, readiness transitions, and watcher metrics.

## Open decisions

1. Whether the cache should live in a project-specific persistent database or a shared user cache keyed by project and encoder namespace.
2. Whether coarse and fine vectors should use separate FAISS segments or one tiered index.
3. Whether partial semantic search should normalize fusion weights dynamically or use explicit serving profiles.
4. What minimum coarse tier is sufficient for quality without creating a second expensive corpus.
5. Which existing task/checkpoint schema can be extended without breaking older persisted indices.
