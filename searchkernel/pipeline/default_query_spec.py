"""The default query `PipelineSpec`: reproduces `SearchOrchestrator.query`'s
full hand-wired stage order -- routing -> effective_top_k -> retrieve ->
seed_bookkeeping -> tag_expansion -> graph_expand -> strategy_results ->
provenance -> fusion -> community_boost -> project_uplift ->
project_filter -> source_filter -> dedup_rerank -> parent_expansion ->
hydrate -- as data, for `PipelineExecutor` to walk against
`DEFAULT_QUERY_STAGE_REGISTRY`. `effective_top_k`/`seed_bookkeeping`/
`strategy_results` are the bookkeeping stages that make a generic walk of
this spec possible at all: without them, `tag_expansion` (needs
`chunk_id_to_doc_id`/`all_doc_ids`/`top_k`), `graph_expand` (needs
`seed_scores`) and `provenance`/`fusion` (need `context.strategy_results`)
would each be missing required `context.metadata` keys on the first
generic run.

Per-query config (weights, top_k, min_confidence, ...) is resolved by the
orchestrator at call time and threaded through `context.metadata`/
per-stage `run_stage` config rather than baked into this spec's (static)
`StageSpec.config`, since some of it (e.g. `fusion`'s `strategy_weights`)
is itself the output of an earlier stage (`routing`) in the same walk.
`SearchOrchestrator.query` walks this spec directly -- a `for stage_spec
in DEFAULT_QUERY_SPEC.stages` loop driving `PipelineExecutor.run_stage`
-- so adding, removing or reordering a stage here is a spec edit, not an
orchestrator edit. `dedup_rerank` is the one documented exception: the
orchestrator special-cases it to call a cached `DedupRerankStage`
instance directly instead of `run_stage`, so the reranker's lazily-loaded
cross-encoder model is reused across queries instead of rebuilt (and
reloaded) on every one.
"""

from __future__ import annotations

from searchkernel.pipeline.spec import PipelineSpec, StageSpec

DEFAULT_QUERY_SPEC = PipelineSpec(
    name="default_query",
    stages=(
        StageSpec(name="routing"),
        StageSpec(name="effective_top_k"),
        StageSpec(name="retrieve"),
        StageSpec(name="seed_bookkeeping"),
        StageSpec(name="tag_expansion"),
        StageSpec(name="graph_expand"),
        StageSpec(name="strategy_results"),
        StageSpec(name="provenance"),
        StageSpec(name="fusion"),
        StageSpec(name="community_boost"),
        StageSpec(name="project_uplift"),
        StageSpec(name="project_filter"),
        StageSpec(name="source_filter"),
        StageSpec(name="dedup_rerank"),
        StageSpec(name="parent_expansion"),
        StageSpec(name="hydrate"),
    ),
)
