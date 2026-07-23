"""DedupRerankStage: the confidence-filter/dedup/doc-limit/rerank query stage.

Lifted from SearchOrchestrator's direct use of SearchPipeline behind the
SearchStage contract. Delegates straight to SearchPipeline.process -- same
inputs, same outputs -- so wiring the orchestrator through this stage is a
pure extraction with no behavior change.

Per-query callables (embedding/content lookups) and top_n don't fit a pure
SearchContext value, so they travel in context.metadata rather than as
extra run() parameters, keeping the SearchStage.run(context) -> context
contract strict.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.pipeline import SearchPipeline, SearchPipelineConfig

_GET_EMBEDDING_KEY = "get_embedding"
_GET_CONTENT_KEY = "get_content"
_TOP_N_KEY = "top_n"
_COMPRESSION_STATS_KEY = "compression_stats"


class DedupRerankStage:
    """Filter/dedup/doc-limit/rerank a fused candidate list (context.candidates).

    Expects `context.metadata` to carry `get_embedding`, `get_content`
    (both `Callable[[str], ...]`) and `top_n` (int); writes the resulting
    `CompressionStats` back to `context.metadata["compression_stats"]`.
    """

    name = "dedup_rerank"

    def __init__(self, config: SearchPipelineConfig):
        self._pipeline = SearchPipeline(config)

    def run(self, context: SearchContext) -> SearchContext:
        get_embedding = context.metadata[_GET_EMBEDDING_KEY]
        get_content = context.metadata[_GET_CONTENT_KEY]
        top_n = context.metadata[_TOP_N_KEY]

        final, stats = self._pipeline.process(
            context.candidates,
            get_embedding,
            get_content,
            context.query,
            top_n,
        )

        metadata = dict(context.metadata)
        metadata[_COMPRESSION_STATS_KEY] = stats
        return replace(context, candidates=final, metadata=metadata)
