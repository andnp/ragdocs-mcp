"""The default query `PipelineSpec`: reproduces `SearchOrchestrator.query`'s
full hand-wired stage order -- routing -> retrieve -> tag_expansion ->
graph_expand -> provenance -> fusion -> community_boost -> project_uplift
-> project_filter -> source_filter -> dedup_rerank -> parent_expansion ->
hydrate -- as data, for `PipelineExecutor` to walk against
`DEFAULT_QUERY_STAGE_REGISTRY`.

Per-query config (weights, top_k, min_confidence, ...) is resolved by the
orchestrator at call time and passed as each `StageSpec.config`; this
spec only pins stage *names* and their *order*. `SearchOrchestrator.query`
does not (yet) walk this spec via a single `PipelineExecutor.run` call --
it invokes each slot through its own helper method, all of which resolve
the concrete stage from `DEFAULT_QUERY_STAGE_REGISTRY` exactly as this
spec pins -- so the spec and the orchestrator's actual call order are two
independently-verified descriptions of the same order (see
`tests/unit/pipeline/test_default_query_spec.py`), the same relationship
that already held for the original five-stage core chain before this
spec grew to cover the whole pipeline. `dedup_rerank` is included for
completeness of the pinned order even though the orchestrator invokes it
via a cached stage instance rather than a fresh `run_stage` call, to
preserve the reranker model cache across queries.
"""

from __future__ import annotations

from searchkernel.pipeline.spec import PipelineSpec, StageSpec

DEFAULT_QUERY_SPEC = PipelineSpec(
    name="default_query",
    stages=(
        StageSpec(name="routing"),
        StageSpec(name="retrieve"),
        StageSpec(name="tag_expansion"),
        StageSpec(name="graph_expand"),
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
