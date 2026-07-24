"""The default query `PipelineSpec`: reproduces `SearchOrchestrator.query`'s
hand-wired stage order (routing -> retrieve -> graph_expand -> fusion ->
dedup_rerank) as data, for `PipelineExecutor` to walk against
`DEFAULT_QUERY_STAGE_REGISTRY`.

Per-query config (weights, top_k, min_confidence, ...) is resolved by the
orchestrator at call time and passed as each `StageSpec.config`; this
spec only pins stage *names* and their *order*.
"""

from __future__ import annotations

from searchkernel.pipeline.spec import PipelineSpec, StageSpec

DEFAULT_QUERY_SPEC = PipelineSpec(
    name="default_query",
    stages=(
        StageSpec(name="routing"),
        StageSpec(name="retrieve"),
        StageSpec(name="graph_expand"),
        StageSpec(name="fusion"),
        StageSpec(name="dedup_rerank"),
    ),
)
