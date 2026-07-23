# Plan: Promote ragdocs In-Place into the Search Kernel — Implementation Tranches

**Status:** In progress — 2026-07-23 (see §Implementation progress)
**Governing design:** `devkit/specs/07-unified-search-architecture.md` (read it for *why*; this doc is *how*)
**Repo scope:** `mcp-markdown-ragdocs` (becomes the `searchkernel` library + daemon)
**Related:** `docs/plans/00-ragdocs-v2-refactor-overview.md`, `.../09-global-daemon-multi-corpus-runtime-refactor.md`

## Audience note (for the orchestrator + implementation agents)

Each **workstream (W#)** below is a self-contained brief: *Goal · Depends on · Deliverables ·
Primary files · Tasks · Acceptance · Tests · Notes*. Hand one workstream to one implementation
agent. Do **not** start a workstream before its `Depends on` are merged. Every task is sized for a
single atomic commit (compiles, lints, tests green — see Conventions). When a brief says "port,"
the canonical signature is in **W1 §Ports**; do not invent variants.

## Executive summary

ragdocs already has the crown-jewel search pipeline, a global daemon/thin-client control plane, and
several small ABCs/Protocols. It lacks: a source-agnostic `Record`, a real `ContentSource` port
(git was bolted on by duplication), a library-usable package boundary (package is literally `src`,
`ApplicationContext.create()` mutates globals), and a unified store (FAISS files + separate SQLite).
This plan turns ragdocs into `searchkernel`: a pure framework layer (domain + ports + composable
query/ingestion stages) plus a runtime (daemon/app), on Postgres+pgvector, consumable both as a
library and a standalone app.

## Global conventions

- **Package rename** `src` → `searchkernel` is W1 and W1 only; nothing else changes in that commit.
- **Dependency rule:** `searchkernel/domain` and `searchkernel/ports` import nothing from
  `adapters/`, `runtime/`, or any concrete store/model. Dependencies point inward. CI should assert
  this (import-linter or a test).
- **Commit discipline** (Andy's): conventional commits, atomic, bisect-safe, no "and" in subjects,
  tests bundled with contract-breaking feats, separate test commits for additive feats.
- **Testing bar:** every new stage/port/adapter ships isolated unit tests; every workstream keeps
  document + git search green end-to-end.
- **No behavior change disguised as a refactor:** extraction commits preserve behavior; behavior
  changes are their own commits.

## Workstream dependency graph

```
W1 (packaging + ports + Record + pgvector store + composition root)
 ├─ W2  (git as first ContentSource; delete duplicated git stack)
 ├─ WP  (caching decorator + ANN + batching)          ← store from W1
 ├─ WE  (eval + observability)                          ← needs a runnable search
 ├─ W3  (embedding + LLM provider registries)
 │       └─ WM (reindex --model migration capability)   ← needs W3 + store
 └─ W4a (query + ingestion toolkit extraction; routing, graph-expand, RAG-fusion)
          └─ W4 (federation: SearchableSource, fan-out, retrieve-then-rerank-once)  ← needs W3+W4a
```
Parallelizable after W1: **WP, WE, W3, W4a** can run concurrently (mostly disjoint files). W2 is
best done right after W1 as the port proof. W4 and WM are the join points.

---

## Implementation progress

Snapshot **2026-07-23**. Commits are on `mcp-markdown-ragdocs` `main` (Andy reviews commits, so SHAs
are listed for traceability). Legend: ✅ complete · ◐ partial · ☐ not started · ⛔ blocked on a
dependency.

| WS | State | Detail |
|----|-------|--------|
| **W1** — packaging, ports, Record, pgvector, composition root | ✅ | rename `2ef01c1`; domain types + `Record` `ffa46f8`; ports `981165b`; inward-dependency test + import-linter `c5a4db2`; pgvector store (per-`(model,dim)` typed `vector(dim)` tables + HNSW cosine) `e02627c`; store-backend config `07aafb1`; pgvector recall/index tests `208cc99`; runtime-thread extraction out of library path `e212280`; composition root `dd54bfa`; composition test `cbd1687`. Real Postgres 17.9 / pgvector 0.8.2 substrate (docker) verified via `EXPLAIN ANALYZE` HNSW index scan. |
| **W2** — git as first `ContentSource` | ✅ | **W2a** port proof `500a812`/`ae7e0a1`. **W2b done (workflow 1, 2026-07-23, sonnet):** `source_filter` on `SearchOrchestrator.query` `a28d6a5`(+test `09918fd`); git commits ingested into the live `IndexManager` `e95eb54`(+test `a2697a0`); source metadata on `ChunkResult` `811bc17`(+test `5d84c36`); reroute the single `search_git_history` MCP caller (`daemon/request_router.py`) through the orchestrator `5d4baef`; delete the duplicated stack — `commit_indexer.py`/`commit_search.py`/`parallel_indexer.py` + 6 obsolete test files `8972338`. Live git search now flows through the unified pipeline on FAISS. |
| **WP** — caching, ANN, batching | ✅ | Additive net-new pieces: in-memory LRU CacheStore `480109f` + sqlite CacheStore `95dc8f1` (+ tests `4f9a9da`); epoch-aware `@cached` decorator `dc5ca70` (+ tests `48ef4bc`); per-source-timeout fan-out helper `7a6f79c` (+ tests `c3341bd`) — the fan-out primitive W4 federation will build on. Embedding-skip already existed (hash-gated in `indexing/manager.py`); locked in by tests `3d851f8`. ANN/batching already satisfied by the W1 pgvector HNSW store. |
| **WE** — eval + observability | ✅ | metrics (recall/nDCG/MRR/AP) `e1df576`; golden-set schema + seed `338230b`; `QueryTrace`/spans `14ca19a`; eval + a/b runner with p50/p95/p99 `9c348d5`; 64 eval tests `06854df`. |
| **W3** — embedding + LLM provider registries | ◐ | **In-process HF `EmbeddingProvider` done** (no Ollama): `HuggingFaceEmbeddingProvider` (Qwen3-Embedding-0.6B, native dim 1024, query-instruction-aware `embed_query`, MRL `truncate_dim`) — adapter `173d9be`, config `45dcad4`, unit tests `c7e8bff`, integration `218388b`; verified 5+1 tests pass on real model + pgvector, recall@1 correct. **Reranker done (workflow 1, haiku):** `Reranker` port `e8f1873`; `HuggingFaceReranker` (Qwen3-Reranker-0.6B, P(yes)-logit scoring) `b9d24d8`(+tests `5006f40`). **Copilot LLM done (workflow 1, haiku):** `CopilotLLMProvider` (shells the `copilot` CLI, model `gpt-5.6-luna`, import-safe when CLI absent) `2b7f6fe`(+mocked tests `a6fb8ad`). **Remaining:** Ollama adapter + a provider registry (deferred; copilot chosen as the first LLM per Andy). |
| **W4a** — query + ingestion toolkit extraction | ◐ | **Started (2026-07-23, sonnet):** import-linter was silently never running (`[tool.import-linter]` vs the installed tool's `[tool.importlinter]`) `b45c943` — fixed first so later contract checks are real. `VectorStore.search()` model-threading concern resolved: `model_name`/`dim` now explicit keyword args on the port + pgvector adapter, `_active_model` instance state removed `0882878` (verified against a live pgvector container). `SearchStage`/`SearchContext` contract `f89dd55`; declarative `PipelineSpec` `b5d1f35`. First two concrete query-toolkit stages, each a pure delegate to existing logic then wired into `SearchOrchestrator` in a separate commit: `FusionStage` (wraps `ScorePipeline`) `a5e3cd1`/`28bc600`; `DedupRerankStage` (wraps `SearchPipeline.process`) `814597f`/`d0f0032`. Document + git search verified green throughout. **Remaining:** retrieve/graph-expand/hydrate/provenance stages (retrieve is async, needs an async-aware stage shape); query routing + RAG-Fusion stages; ingestion toolkit (discover/chunk/embed/index/repair) untouched; no `PipelineSpec`-driven executor yet (spec is data-only). |
| **W4** — federation | ⛔ | Blocked on W3 + W4a. Open concern: async DB driver (current pgvector adapter is sync psycopg2); the WP fan-out helper is ready. |
| **WM** — `reindex --model` migration | ✅ | **Done (workflow 1, haiku):** build-then-swap reindex routine against the `VectorStore` port (old vectors survive until the flip, mixed-dim guards, rollback-before-flip) `8898884`; `reindex --model [--truncate-dim N]` CLI subcommand `4126b1d`; 18 state-machine tests `d476872`. Implemented against the port, so it works once the live store is cut over. |

**Open design items surfaced during implementation:**
- **pgvector is built but DARK (major gap).** W1 added the store adapter (`PGVectorStore`/`PGKeywordStore`/`PGGraphStore`/`PGCacheStore`) + a `StoreConfig.backend` flag + full tests, but **nothing in the live path consumes it**: `store.backend` is never read, and `PGVectorStore` is imported only by `adapters/stores/__init__` and tests. The live index/search path hardcodes FAISS (`context.py` → `faiss_index.bin`). Cutting the live `IndexManager.vector` over to the `VectorStore` port is unfinished work entangled with **W4a** (the live index exposes a richer interface than the narrow port). **Decision 2026-07-23:** proceed with W2b + corpus indexing on the current FAISS+sqlite store; treat the pgvector live-cutover as its own workstream done with/after W4a (re-indexing is cheap — the corpus is a throwaway).
- `VectorStore.search()` has **no `model_name` param** — it relies on instance "active model" state,
  which is not concurrency-safe. The composition root / W4a must resolve this (thread `(model,dim)`
  through the call, not instance state) before federation runs multi-model queries.
- Federation fan-out (W4) needs an **async DB driver**; today's pgvector adapter is sync.
- ~~Pre-existing pyrefly error: `tests/e2e/test_hyde_tool.py:145` passes `ctx=` to
  `MCPServer.__init__`.~~ **Resolved `363ea7c`** — `MCPServer()` (context flows via `HandlerContext`);
  the whole hyde module was red (TypeError) and is now green.
- **Workflow-1 gate findings (2026-07-23):** the haiku verifiers reported PASS but ran only their
  workstream's test *subset*, so two regressions slipped through, caught by an independent full-suite
  run in the repo's real xdist mode (`-n --dist worksteal`): (1) the W2b conftest daemon-leak
  teardown killed the pytest-xdist worker on the 8 in-process lifecycle tests (it stopped the
  runner's own PID) — fixed by a PID guard, amended into `44eea6e`; (2) W2b's `test_worker.py`
  rewrite exercised `register_tasks` against a `_FakeContext` missing `index_path`/`documents_roots`
  — fixed, amended into `8972338`. **Lesson: subset-scoped verification cannot catch
  cross-test/parallel regressions; the orchestrator must run the full suite in the repo's xdist mode
  at the gate** (but see the tmpfs warning below — run targeted files, not the whole suite at once).

**Verification methodology used throughout:** verify-don't-trust — inspect code / `EXPLAIN` output
rather than trust agent self-reports; diff failing tests against the parent commit to separate
regressions from pre-existing failures; reset + re-commit for clean atomic history. This caught real
defects agents had self-reported as successes (pgvector brute-force punt, composition-root ordering
regression, rename `src.config` stragglers, and a WP `adapters/cache/__init__.py` that declared
`__all__` without importing the names). **Do not run the full pytest suite** — it filled `/tmp` with
~16G of fixtures; use `pytest --co` for collection + run targeted files.

---

## W1 — Kernel packaging, ports, Record, pgvector store, composition root

**Goal:** make ragdocs importable as a library with clean seams and a single Postgres+pgvector
store, with **zero behavior change** to document/git search.

**Depends on:** nothing.

**Deliverables**
1. `src` → `searchkernel` package rename; `pyproject` console script + a library entrypoint
   (`searchkernel/__init__.py` exporting the public surface).
2. `searchkernel/domain/` — `Record`, `Chunk`, `SearchResult`, `ScoredRef` (pure, no I/O).
3. `searchkernel/ports/` — the port ABCs/Protocols (see §Ports).
4. `searchkernel/app/composition.py` — a composition root replacing `ApplicationContext.create()`;
   **no global env/singleton mutation** (move `OMP_NUM_THREADS` etc. to an explicit runtime-config
   step the *app* calls, not the library).
5. Postgres+pgvector store scaffolding behind `VectorStore`/`KeywordStore`/`GraphStore` ports; keep
   FAISS/SQLite adapters as legacy fallbacks selectable by config.

**Primary files**
- rename: all of `src/**` → `searchkernel/**`; fix imports repo-wide.
- new: `searchkernel/domain/models.py` (fold in `models.py` `Document`/`CommitResult` → also add
  `Record`), `searchkernel/ports/*.py`, `searchkernel/app/composition.py`,
  `searchkernel/adapters/stores/pgvector.py`, `.../stores/faiss_legacy.py`, `.../stores/sqlite_legacy.py`.
- edit: `context.py` (shrink toward composition root), `config.py` (store backend selection),
  `storage/db.py` (Postgres schema alongside SQLite).

**§Ports (canonical signatures — all other workstreams bind to these)**
```python
# ports/content_source.py
class ContentSource(Protocol):
    source_kind: str
    def iter_records(self, since: Cursor | None) -> Iterable[Record]: ...      # ingestible
    def change_signal(self) -> ChangeSignal: ...                               # watch or poll_interval
class SearchableSource(Protocol):
    source_kind: str
    async def search(self, query: str, k: int, filters: Filters | None) -> Iterable[ScoredRef]: ...
# ports/embedding.py
class EmbeddingProvider(Protocol):
    model_name: str; dim: int
    def embed(self, texts: list[str]) -> list[Vector]: ...                     # batched
# ports/llm.py
class LLMProvider(Protocol):
    async def complete(self, prompt: str, *, response_format=None, tier: Tier = Tier.FAST) -> str | dict: ...
# ports/stores.py
class VectorStore(Protocol): ...   # upsert(records, model_name, dim), search(vec, k, filters), delete(ids), epoch()
class KeywordStore(Protocol): ...  # index(records), search(query, k, filters)
class GraphStore(Protocol): ...    # upsert_edges, neighbors(id, edge_types, depth)
class CacheStore(Protocol): ...    # get(key), set(key, val, epoch), by-epoch invalidation
# ports/search.py  (driving)
class SearchAPI(Protocol):
    async def search_anything(self, query, *, sources=None, filters=None) -> list[SearchResult]: ...
```

**Tasks**
1. Mechanical `src`→`searchkernel` rename + import fixes (one commit, tests green).
2. Add `domain/` types incl. `Record`; make `Document`/`CommitResult` construct from/へ `Record`.
3. Add `ports/` ABCs (signatures above); wire `IndexProtocol`/`EmbeddingModel`/`IndexManagerLike`
   into them (reconcile the aspirational `IndexProtocol` with real signatures).
4. Extract composition root; delete global-state mutation from library path.
5. pgvector store adapter + Postgres schema (vectors/keyword via tsvector/graph tables/kv); config
   flag `store.backend = pgvector|faiss+sqlite`; legacy adapters kept.
6. import-linter CI rule enforcing inward dependencies.

**Acceptance**
- `from searchkernel import SearchKernel; kernel = SearchKernel.build(cfg, ...)` works with no daemon
  and no global mutation.
- Document + git search identical behavior on both `pgvector` and `faiss+sqlite` backends.
- `domain`/`ports` import nothing from adapters/runtime (CI-enforced).

**Tests:** rename regression (full suite green); composition-root unit test; pgvector store parity
test vs FAISS on a fixture corpus.

**Notes:** the rename is the highest-churn, lowest-intellectual-risk task — do it first, alone.

---

## W2 — Prove the port: git as the first `ContentSource`

**Goal:** reimplement git-commit support as a `ContentSource` over the unified store, deleting the
duplicated git stack. This is the open/closed proof.

**Depends on:** W1.

**Deliverables**
- `searchkernel/adapters/sources/git.py` implementing `ContentSource` (yields commits as `Record`s).
- Commits indexed into the **same** pgvector/keyword/graph store as documents.
- **Deleted:** `git/commit_indexer.py`'s `git_commits.db`, its in-Python cosine, and
  `git/commit_search.py::search_git_history`. Git search now goes through `SearchOrchestrator`.

**Primary files:** new `adapters/sources/git.py`; edit/remove `git/*`; edit `mcp/tools/*` and CLI
`search-commits` to route through the unified search; edit `models.py` (`CommitResult` → `Record`).

**Tasks**
1. Map a commit → `Record` (source_kind="git_commit", source_id="git:<sha>", body=message+diff summary,
   metadata=author/time/files, uri=commit link).
2. Implement `iter_records(since=last_sha/commit_time)` + change signal (reuse `git/watcher.py`).
3. Route commit indexing through `IndexManager` into the shared store.
4. Delete the duplicated storage + search; point `search_git_history` MCP tool at
   `SearchOrchestrator` with a `source_filter=["git_commit"]`.
5. Migration note: existing `git_commits.db` is a throwaway; reindex from git.

**Acceptance:** git history search returns equivalent-or-better results through the unified
pipeline; `git_commits.db` and `search_git_history` no longer exist; adding git touched **zero** core
files (only an adapter + wiring).

**Tests:** git adapter unit test; end-to-end "commit appears in search_anything with source filter."

---

## WP — Caching decorator, ANN, batching (performance foundation)

**Goal:** the perf wins that are mostly lift-and-share of existing code.

**Depends on:** W1 (store). Overlaps W2.

**Deliverables**
1. `@cached` stage decorator + `CacheStore` port + **index-epoch invalidation** (bind entries to
   manifest `spec_version` + a corpus-version counter; reindex/upsert bumps epoch).
2. Lift memory's query-embedding cache + ragdocs' `search/result_cache.py` under the decorator;
   add reranker-pair cache; add a **semantic** (similarity-keyed) variant behind a flag.
3. ANN everywhere: on `pgvector` this is HNSW index config (no custom code); ensure no
   pure-Python cosine path remains (removed in W2 for git).
4. Batched embedding (32–256), content-hash embedding cache (skip re-embed of unchanged chunks via
   `indices/hash_store.py`), per-source-timeout parallel fan-out helper (used later by W4).

**Primary files:** new `searchkernel/runtime/cache.py` (`@cached`, `CacheStore`),
`adapters/cache/{memory_lru,sqlite,postgres}.py`; edit `search/result_cache.py`, `indices/vector.py`
(batching), `indices/hash_store.py`.

**Acceptance:** measurable latency drop on repeated queries (via WE timing); zero brute-force cosine
in the codebase; embedding cost skipped for unchanged content (assert via counter in a test).

**Tests:** cache hit/miss + epoch-invalidation unit tests; batching test; "unchanged chunk not
re-embedded" test.

**Notes:** WP is measured by latency → needs only WE's **observability** half, not the golden set.

---

## WE — Evaluation & observability harness (the governor)

**Goal:** make retrieval quality and latency measurable; gate every quality-stage change.

**Depends on:** W1 (a runnable search). Start early.

**Deliverables**
1. Per-stage timing/provenance trace unifying `SearchResultProvenance` + stage timings into one
   structured trace emitted per query.
2. An eval runner: `eval run <PipelineSpec> <golden_set>` → recall@k, nDCG, MRR + latency percentiles;
   `eval ab <specA> <specB>` diff.
3. A golden-set format + a small seed set (hand-label from real ragdocs queries).

**Primary files:** new `searchkernel/eval/{runner.py,metrics.py,golden.py}`,
`searchkernel/runtime/trace.py`; edit search stages to emit trace spans.

**Acceptance:** `eval ab` reports metric + latency deltas between two specs; every later
quality-stage workstream (W4a stages, W4 rerank, WM validate) cites an eval delta.

**Tests:** metric-computation unit tests against known fixtures; runner smoke test.

---

## W3 — Provider registries: embedding + LLM

**Goal:** pluggable embedding + a tiered LLM facade; lock the embedding model.

**Depends on:** W1.

**Deliverables**
1. `EmbeddingProvider` registry: adapters for Ollama (`qwen3-embedding`, `embeddinggemma`) + HF.
   **Default model = `Qwen3-Embedding-0.6B` (1024-dim, MRL)**; `EmbeddingGemma-300m` alternative.
2. `LLMProvider` tiered facade: `complete(prompt, response_format, tier)` with fallback; adapters
   `gemma-api`, `ollama` (FAST), `claude-cli`, `codex-cli`, `copilot-cli` (SMART). Modeled on
   eng-analytics `AIProvider`; **distinct from** devkit `AgentBackend`.
3. Reranker adapter for `Qwen3-Reranker-0.6B` behind the `Reranker` seam (used by W4).

**Primary files:** new `adapters/embedding/{ollama,hf}.py`, `adapters/llm/{gemma_api,ollama,claude_cli,codex_cli,copilot_cli}.py`,
`adapters/rerank/qwen3.py`; edit `indices/vector.py` (remove hard-wired HF), `config.py`.

**Acceptance:** switching embedding or LLM provider/tier is config-only; a per-chunk FAST call never
routes through a CLI subprocess (guard/assert); Qwen3 embed + rerank both run locally via Ollama.

**Tests:** provider-registry selection tests; fallback test; tier-routing test.

---

## WM — `reindex --model` migration capability

**Goal:** safe, reversible, eval-gated embedding-model migration (generalize memory's
`embedding_repair_queue`). No data loss ever (embeddings are a derived cache).

**Depends on:** W3, W1 store.

**Deliverables:** durable task `reindex --model <new>` implementing expand → backfill → validate →
flip → contract; per-model isolated embeddings (`(model_name, dim)`); write-time **mixed-dimension
guard**; DB backup before contract.

**Primary files:** new `searchkernel/ingestion/migration.py`; edit store adapters (per-model
column/partition), `indexing/manifest.py` (active-model pointer + epoch).

**Acceptance:** migrate a fixture corpus 768→1024 model with old model serving throughout; rollback
before flip works; mixed-dim write rejected; eval gate blocks a regressing flip.

**Tests:** expand/backfill/flip/contract state-machine tests; mixed-dim rejection test;
rollback test.

---

## W4a — Toolkit extraction (query + ingestion stages)

**Goal:** turn the ~30 search modules + ingestion primitives into composable stages behind one
contract, assembled via a declarative `PipelineSpec`; add the nearly-free new stages.

**Depends on:** W1. Overlaps W3.

**Deliverables**
1. `SearchStage` contract (pure transform over `SearchContext`) + `PipelineSpec` (declarative,
   pinned per domain). ragdocs' `.query(pipeline_config=...)` generalizes into this.
2. **Query toolkit:** retrieve (vector/keyword) · graph one-hop expansion (reuse `GraphStore`) ·
   RRF fusion (variance-aware) · normalize/calibrate · recency/type scoring · threshold · dedup ·
   MMR · rerank (Reranker port) · hydrate · provenance. Plus **query routing** (extend
   `search/classifier.py`) and **RAG-Fusion** (behind a flag, off by default).
3. **Ingestion toolkit:** discover · chunk (`ChunkingStrategy`) · embed (batched, hash-gated) ·
   index · dedup/canonicalize · re-embed/repair (shared with WM).
4. Domain policy providers: `GraphProvider` (edge types+weights), timestamp selector, status→multiplier map.

**Primary files:** new `searchkernel/pipeline/{stage.py,spec.py}`, `searchkernel/stages/**` (lift
from `search/*` and ingestion), `searchkernel/ingestion/**`; keep behavior via a default
`PipelineSpec` matching today's `SearchOrchestrator`.

**Acceptance:** ragdocs search runs entirely off a default `PipelineSpec` with identical results
(eval parity); adding/removing a stage is a spec edit; routing + graph-expansion available as
stages; each stage has isolated tests.

**Tests:** per-stage unit tests; "default spec == legacy orchestrator" parity test via WE.

**Notes:** extraction criterion is **cohesion, not usage count** (see design doc); keep each stage's
contract narrow + versioned.

---

## W4 — Federation: `SearchableSource`, fan-out, retrieve-then-rerank-once

**Goal:** `search_anything` spanning the kernel index + federated sources.

**Depends on:** W3 (reranker/LLM), W4a (stages), WP (fan-out helper).

**Deliverables**
1. `SearchAPI.search_anything` fans out (routed subset) to the kernel index + registered
   `SearchableSource`s in parallel with per-source timeouts.
2. **Retrieve-then-rerank-once** as default fusion (gather candidate texts → one cross-encoder pass
   → single comparable score); RRF fallback when reranker unavailable.
3. Federated-source registry + health/status in `AdminAPI`.

**Primary files:** edit `search/orchestrator.py` (or new `search/federation.py`), `daemon/request_router.py`
(add `search_anything` path), `mcp/tools/*` (expose the tool).

**Acceptance:** `search_anything` returns one fused ranked list across kernel + ≥1 stub
`SearchableSource`; a slow source is dropped without corrupting ranking; rerank path is model-agnostic.

**Tests:** fan-out + timeout test; rerank-once vs RRF parity/fallback test; routing-subset test.

---

## Milestones

- **M1:** W1 merged — library-importable kernel on pgvector, search unchanged.
- **M2:** W2 + WP — git on the port, brute-force cosine gone, caching live.
- **M3:** W3 + WE — providers pluggable, Qwen3 default, eval harness gating.
- **M4:** W4a + W4 — composable pipeline + federation with rerank-once.
- **M5:** WM — model migration proven end-to-end.

## Acceptance criteria (repo-level)

1. `searchkernel` is a library (no daemon, no global mutation) **and** a standalone app.
2. Adding a source = new adapter, zero core edits (git is the proof).
3. One store (pgvector); no FAISS-files/SQLite split in the default path; no brute-force cosine.
4. Every domain pipeline is a pinned `PipelineSpec`; stage changes are eval-gated.
5. Embedding-model migration is reversible, eval-gated, and loses no data.
