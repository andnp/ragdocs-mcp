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
| **W4a** — query + ingestion toolkit extraction | ◐ | **Started (2026-07-23, sonnet):** import-linter was silently never running (`[tool.import-linter]` vs the installed tool's `[tool.importlinter]`) `b45c943` — fixed first so later contract checks are real. `VectorStore.search()` model-threading concern resolved: `model_name`/`dim` now explicit keyword args on the port + pgvector adapter, `_active_model` instance state removed `0882878` (verified against a live pgvector container). `SearchStage`/`SearchContext` contract `ca277b1`; declarative `PipelineSpec` `294068b`. Four concrete query-toolkit stages so far, each a pure delegate to existing logic then wired into `SearchOrchestrator` in a separate commit: `FusionStage` (wraps `ScorePipeline`) `adde2cc`/`4270a5c`; `DedupRerankStage` (wraps `SearchPipeline.process`) `2183d40`/`c926e17`. **Continued (2026-07-23, sonnet, second pass):** resolved the async-stage-shape design gap flagged by the prior gate -- added a parallel `AsyncSearchStage` protocol (same shape, coroutine `run`) alongside the sync `SearchStage` `551a3fd`; `RetrieveStage` (gathers vector+keyword retrieval concurrently, parameterized over the searcher callables so it stays monkeypatchable) `081b49e`/`2b67ac3`; `GraphExpandStage` (one-hop rank-neighbors + chunk-candidate expansion, same callable-injection pattern) `54e9c1f`/`5c26ee7`. Note: `f89dd55`/`b5d1f35`/`a5e3cd1`/`28bc600`/`814597f`/`d0f0032` from the first pass were rewritten to `ca277b1`/`294068b`/`adde2cc`/`4270a5c`/`2183d40`/`c926e17` by a rebase that reworded `f89dd55`'s subject (it contained "and", a hard-rule violation) -- no content changes. Document + git search (targeted suites, 51+44 passed) verified green throughout both passes. **Continued (2026-07-23, sonnet, third pass):** decomposed the two coarse wrappers into the plan's actual per-concern stages. Fusion toolkit: `RRFFuseStage` (wraps `ScorePipeline.fuse`) `ca453cd`; `CalibrateStage` (wraps `calibrate_results` at the exact threshold/steepness `ScorePipeline.calibrate` hardcodes; normalization stays skipped, matching the documented reason in `score_pipeline.py`) `1f7f936`; `RecencyBoostStage` (wraps `ScorePipeline.boost`, recency only -- no type-scoring exists in the codebase) `31c2a1b`; `FusionStage` rewired to compose these three in the same fuse→calibrate→(boost) order instead of delegating to whole-`ScorePipeline.run` `c56bc66`. Dedup/rerank toolkit: `ThresholdStage` (`filter_by_confidence`) `242dfea`; `ContentHashDedupStage`/`NgramDedupStage`/`SimilarityDedupStage` (the three `search/dedup.py` techniques, each its own stage rather than one bundled "dedup" stage, so `CompressionStats`' `after_content_dedup`/`after_ngram_dedup`/`after_dedup`/`clusters_merged` fields each map to one stage boundary) `ba6c772`/`1de9730`/`326173f`; `DocLimitStage` (wraps `limit_per_document`; named honestly -- the plan calls this slot "MMR" but no maximal-marginal-relevance reranker exists today) `6ba4e64`; `RerankStage` (wraps `ReRanker.rerank`) `8fc234a`; `DedupRerankStage` rewired to compose all six in SearchPipeline.process's original order, computing `CompressionStats` from each stage's before/after candidate-count delta (verified algebraically equal to the original inline bookkeeping, including `clusters_merged`) instead of delegating to whole-`SearchPipeline.process` `ae84295`. Query routing: `RoutingStage` (wraps `classify_query`/`get_adaptive_weights`) `efc1b33`; orchestrator's two separate classifier call-sites merged into one early `RoutingStage.run` call (adaptive-weight computation moved earlier -- pure function of `query_type` + config constants, so its result is unchanged) `f7e68df`. Document + git search (pipeline/orchestrator unit+integration suites + git e2e reroute, 137/137) verified green after every wiring commit. **Remaining (as of third pass):** RAG-Fusion stage (flagged off, not started); the entire ingestion toolkit (discover/chunk/embed/index/dedup-canonicalize/re-embed-repair) untouched; domain policy providers (`GraphProvider`, timestamp selector, status→multiplier map) not started; a `PipelineSpec`-driven executor that actually walks a spec against a stage registry (spec is still inert data -- nothing consumes it, so "adding/removing a stage is a spec edit" is not yet true); the "default spec == legacy orchestrator" eval-parity test via WE; hydrate/provenance are not yet extracted as standalone stages (they remain entangled in orchestrator methods -- `ChunkHydrator`, `_build_result_provenance`, `_materialize_chunk_results` -- pulling them out cleanly needs its own pass since provenance is threaded through community-boost/project-uplift/parent-expansion too, not just dedup/rerank). **Continued (2026-07-24, sonnet, fourth pass -- the executor keystone):** built the piece that makes `PipelineSpec` non-inert. `StageDeps` + `DEFAULT_QUERY_STAGE_REGISTRY` (maps a `StageSpec.name` to a factory resolving orchestrator-bound callables for `retrieve`/`graph_expand`, plain config for `routing`/`fusion`/`dedup_rerank`) `c70f49b`; `PipelineExecutor` (`run_stage` resolves+runs one named stage, awaiting only if the resolved stage is async; `run` walks a whole spec threading `SearchContext`) `ab7425c`; `DEFAULT_QUERY_SPEC` pinning today's routing→retrieve→graph_expand→fusion→dedup_rerank order as data `346a8e3`; `SearchOrchestrator`'s five hardcoded stage-construction call sites (`_route`/`_retrieve`/`_graph_expand`/`_apply_score_pipeline`/`_get_pipeline`+`_resolve_pipeline`) rewired to resolve through `PipelineExecutor`+the registry instead of importing stage classes directly `f157896` -- so swapping or disabling one of these five now is a registry/spec edit, not an orchestrator edit. **Scope note:** this covers the query-toolkit-proper chain only, not the whole of `query()` -- tag expansion, community-boost, project-uplift/filter, source-filter, parent-expansion, provenance-build and materialize still run as hardcoded orchestrator glue *between* those five executor-driven slots (unchanged from before this pass), because collapsing them into spec-driven stages is W4a's still-open hydrate/provenance item (see above) and was deliberately left to that pass rather than rushed here. **Design constraint surfaced:** `dedup_rerank`'s stage cannot be freshly built-and-run through the generic executor loop on every query -- `RerankStage` lazily loads (and caches) a cross-encoder model on first `.run()`, so the orchestrator's existing `self._pipeline` cache (reuse across queries unless a custom `pipeline_config`/`disable_reranking` is given) had to be preserved exactly; only the stage's *construction* was moved behind the registry, its cached *instance* is still reused as before. Any future work that tries to fold `dedup_rerank` into one contiguous multi-stage `executor.run(full_spec, ...)` call must account for this or it will reload the reranker model every query. Verified via targeted suites only (per the tmpfs warning): the seven protected behavior-test files (105 tests) + orchestrator/graph-expansion/classifier/daemon-router/git-e2e/txt-indexing unit suites (65 tests) green; `test_query_execution_context.py`'s `DedupRerankStage` monkeypatch target moved from `orchestrator.DedupRerankStage` to `registry.DedupRerankStage` (construction moved modules, same behavior asserted) since it white-box-patches the construction seam. One pre-existing flake (`test_move_detection_with_git_rename`, unrelated to this pass -- reproduces identically on unmodified HEAD, fails only when run alongside its sibling tests in the same file, passes in isolation) confirmed not a regression. **Remaining after fourth pass:** RAG-Fusion stage; the ingestion toolkit; domain policy providers; hydrate/provenance extraction (still needed both for its own sake and to let the executor absorb the rest of `query()`'s glue); the eval-parity test. **Continued (2026-07-24, sonnet, fifth pass):** four of the five remaining items advanced. **Hydrate/provenance:** `ProvenanceStage` (wraps `_build_result_provenance`'s per-strategy rank/raw-score bookkeeping) `895d2b1`/`954cd00`; `HydrateStage` (wraps `_materialize_chunk_results`; reports failed-hydration chunk ids via `context.metadata["missing_chunk_ids"]` rather than calling the reindex side-effect itself, so the stage stays pure and the orchestrator still owns queuing reindex exactly as before) `24f9e85`/`52a413a`. Neither is registered in `DEFAULT_QUERY_STAGE_REGISTRY`/`DEFAULT_QUERY_SPEC` yet -- they run earlier (`provenance`, right after tag expansion) and later (`hydrate`, after parent-expansion) than the five-slot contiguous sub-chain the spec pins, so folding them in requires the spec to grow to cover the *whole* `query()` shape, not just be a registry addition; that whole-pipeline-spec work is still open. **RAG-Fusion:** `RAGFusionStage` (query-variant generation + per-variant retrieval + cross-variant RRF fuse via `ScorePipeline.fuse`, treating each variant as an equally-weighted synthetic "strategy") added and registered as `"rag_fusion"` in the registry, `RAGFusionConfig.enabled=False` by default so it is a no-op passthrough and not part of `DEFAULT_QUERY_SPEC` -- registering it changes nothing for existing callers `6965d71`. **Domain policy providers:** `GraphProvider` (edge-type -> weight policy, lifted out of `search/edge_types.py` which now delegates to it) `cbdff4c`/`c306578` -- this one is real, live-path behavior (verified via `test_edge_types.py`/`test_graph_store.py`/`test_graph_expansion.py` green); `StatusMultiplierPolicy` (`RecordStatus` -> score-multiplier map) `1517bac` and `TimestampSelector` (picks `Record.updated_at` over `created_at` for recency, since `updated_at` is the field `Record`'s own docstring names as the incremental-sync watermark) `5a2a494` are both new pure domain scaffolding, **not wired into any live scoring path** -- `RecencyBoostStage`'s `time_scoring_mode` is never set in the live config today, so there is no existing status/timestamp-driven scoring behavior to lift; these exist so future scoring work has one policy object to consume instead of reinventing the mapping per call site. **Ingestion toolkit:** started (not finished) -- `DiscoverStage` (wraps `ApplicationContext.discover_files`'s single-root-vs-multi-root branch, delegating to `searchkernel.indexing.discovery`'s existing functions) `32b08b7`/`3fb14a9` is the first of the six ingestion phases (discover→chunk→embed→index→dedup/canonicalize→re-embed/repair); chunk/embed/index/dedup-canonicalize/re-embed-repair are still untouched in `indexing/manager.py` and friends. **Eval-parity test:** still not attempted. Deliberately skipped rather than rushed: today `PipelineExecutor` only drives 5 of `query()`'s stages individually (each via its own `run_stage` call from orchestrator glue, per the fourth-pass scope note) -- there is no independent "spec-driven pipeline" implementation distinct from the orchestrator to compare against yet, so a literal "`DEFAULT_QUERY_SPEC`-driven search vs. legacy orchestrator" WE-harness parity test would either be vacuous (both paths are the same code) or require first building the full-`query()`-as-one-spec executor run, which is a bigger lift than fits this pass. Verified via targeted suites only (per the tmpfs warning): all seven protected behavior-test files + the full `tests/unit/pipeline/` tree + `test_edge_types.py`/`test_graph_store.py`/`test_graph_expansion.py`/`test_file_discovery.py`/`test_context.py`/`test_multi_project_context.py`/`test_bootstrap_session.py` green after every wiring commit (224+ tests). Re-verified the four pre-existing `indexing/manager.py` `_persist_indices` `AttributeError` failures (`test_index_manager_batching.py` x2, `test_delta_indexing.py`, `test_multi_index.py`) are unchanged/untouched by this pass -- `indexing/manager.py` itself was not edited except via `context.py`'s `discover_files` delegation, unrelated to persist. **Remaining after fifth pass:** the rest of the ingestion toolkit (chunk/embed/index/dedup-canonicalize/re-embed-repair); folding `provenance`/`hydrate` into a whole-`query()` spec (currently just callable stages, not registry/spec members); wiring `StatusMultiplierPolicy`/`TimestampSelector` into an actual scoring stage if that behavior is ever wanted; the eval-parity test (blocked on the whole-`query()`-as-one-spec executor work above). **Continued (2026-07-24, sonnet, sixth pass):** closed both gate-flagged gaps from the fifth pass plus two more ingestion phases. `ProvenanceStage`/`HydrateStage` registered in `DEFAULT_QUERY_STAGE_REGISTRY` (`StageDeps.hydrate_chunk_result` added for the latter's orchestrator-bound callable) `b6a893e`; `SearchOrchestrator._build_result_provenance`/`_materialize_chunk_results` rewired to construct both through the registry instead of importing the stage classes directly, matching the existing `_build_dedup_rerank_stage` pattern for stages that must stay synchronously callable outside the async `PipelineExecutor` (both methods are called synchronously from a protected behavior test, so they cannot go through `PipelineExecutor.run_stage`'s coroutine path without breaking it) `184fd7e`. **Ingestion toolkit:** `ChunkStage` (wraps `ChunkingStrategy.chunk_document`, parameterized over the chunker instance like `RetrieveStage` over its searchers) `a265a01`, wired into all three of `IndexManager`'s `chunk_document` call sites (`index_document`/`index_record`/`reconcile_indices`'s move-detection parse) via one `_chunk_document` helper `8ddb32f`; `IndexStage` (wraps the identical `vector.add_chunks`+`keyword.add_chunks`+per-chunk `graph.add_node` block `_update_chunks` and `_full_reindex_document` each inlined -- embedding itself happens inside `VectorIndex.add_chunks`, hash-gated per WP, so this is also where embedding is triggered) `c80afb4`, wired into both via a `_index_chunks` helper `c5aafee`. Stale `PipelineSpec` module docstring (said building an executor was follow-on work, predating the executor) refreshed `b1e50d6`. Verified via targeted suites only (per the tmpfs warning): all seven protected behavior-test files + full `tests/unit/pipeline/` tree + `test_index_manager_record.py`/`test_index_manager_reindex.py`/`test_index_manager_batching.py`/`test_indexing_tasks.py`/`test_delta_indexing.py`/`test_multi_index.py`/`test_keyword_index.py`/`test_vector_index.py` green after every wiring commit; `lint-imports` kept green throughout. Same four pre-existing `IndexManager._persist_indices` `AttributeError` failures re-confirmed unrelated (untouched call path). **Remaining after sixth pass:** ingestion toolkit is 3 of 6 phases done (discover/chunk/index) -- embed has no separate extraction point (it lives inside `VectorIndex.add_chunks`, already covered by `IndexStage`), but dedup/canonicalize (the `_detect_file_moves`/`_apply_file_move` move-detection pair) and re-embed/repair (`reindex_document`'s missing-chunk recovery path) are still untouched, inlined `IndexManager` methods; `provenance`/`hydrate` are registry members now but `DEFAULT_QUERY_SPEC` still only pins the original contiguous five-stage chain -- the rest of `query()`'s glue (tag expansion, community-boost, project-uplift/filter, source-filter, parent-expansion) is still hardcoded orchestrator code between registry-driven stages, so a literal whole-`query()`-as-one-spec walk still does not exist; the eval-parity test remains blocked on that for the same reason as the fifth pass; `StatusMultiplierPolicy`/`TimestampSelector` remain unwired scaffolding (no live behavior to preserve). **Continued (2026-07-24, sonnet, seventh pass):** hygiene fix + closed the ingestion toolkit. Reworded two commit subjects (`b6a893e`->`f08ecd7`, `184fd7e`->`8d36fbe`) via `git rebase -i` to drop the literal word "and" (a hard-rule violation flagged by the prior gate) without touching their content -- both were already-correct paired feat/refactor commits, only the wording changed. Re-verified the four pre-existing `IndexManager._persist_indices` `AttributeError` failures: `git log -S _persist_indices` shows the method was removed by `6c40e75`, which `git merge-base --is-ancestor` confirms is already on `origin/main`, well before any W4a commit -- out of scope, untouched. **Ingestion toolkit, closed:** dedup/canonicalize -- `DetectMovesStage` (wraps `_detect_file_moves`'s content-hash-overlap comparison) `cde6c2e`/`99850ef`; `ApplyMoveStage` (wraps `_apply_file_move`'s vector/keyword/graph rename + hash-store rebind, leaving persistence/logging/state-version bookkeeping at the call site per `IndexStage`'s precedent) `0a2a4c2`/`747d04d`. Re-embed/repair -- `RepairStage` (wraps `reindex_document`'s multi-root-with-single-root-fallback doc_id -> path resolution) `b078a39`/`936f250`; the re-embedding itself (`remove_document` + `index_document(force=True)`) was already covered transitively by `ChunkStage`/`IndexStage`, so this stage -- like `DiscoverStage` before it -- extracts only the root-selection branch, not the surrounding side effects. This closes all six named ingestion phases (embed has no separate extraction point, folded into `IndexStage`, as noted above). Verified via targeted suites only: `test_file_move_detection.py`/`test_move_detection.py`/`test_move_detection_integration.py` (two pre-existing order-dependent flakes -- `test_move_detection_with_git_rename`, `test_move_with_partial_edit_above_threshold` -- reproduce identically on pre-change HEAD via `git stash`, confirmed unrelated), `test_index_manager_record.py`/`test_index_manager_reindex.py`/`test_indexing_tasks.py`/`test_delta_detection.py`, and all seven protected behavior-test files, green throughout; `lint-imports` green throughout. **Remaining after seventh pass:** only the eval-parity test, still blocked on the same whole-`query()`-as-one-spec gap identified in the fifth/sixth passes (tag expansion, community-boost, project-uplift/filter, source-filter, parent-expansion still hardcoded orchestrator glue between registry-driven slots) -- folding that glue into the spec is a bigger lift than fits a single pass and was not attempted here to avoid an artificial/vacuous parity test; `StatusMultiplierPolicy`/`TimestampSelector` remain unwired scaffolding (no live behavior to preserve, unchanged from prior passes). |
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
