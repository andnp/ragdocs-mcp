"""DedupRerankStage: composes the threshold/dedup/doc-limit/rerank stages.

Lifted from SearchOrchestrator's direct use of SearchPipeline behind the
SearchStage contract. Composes the finer-grained ThresholdStage,
ContentHashDedupStage, NgramDedupStage, SimilarityDedupStage,
DocLimitStage and RerankStage in the same order
SearchPipeline.process runs its threshold/dedup/doc-limit/rerank steps
-- same inputs, same outputs -- so this stays a pure extraction with no
behavior change. Each finer stage's before/after candidate-count delta
reproduces the CompressionStats SearchPipeline.process computed inline.

Per-query callables (embedding/content lookups) and top_n don't fit a pure
SearchContext value, so they travel in context.metadata rather than as
extra run() parameters, keeping the SearchStage.run(context) -> context
contract strict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from searchkernel.domain import CompressionStats
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.dedup_content_hash import ContentHashDedupStage
from searchkernel.pipeline.stages.dedup_ngram import NgramDedupStage
from searchkernel.pipeline.stages.dedup_similarity import SimilarityDedupStage
from searchkernel.pipeline.stages.doc_limit import DocLimitStage
from searchkernel.pipeline.stages.rerank import RerankStage
from searchkernel.pipeline.stages.threshold import ThresholdStage
from searchkernel.search.pipeline import SearchPipelineConfig

_GET_EMBEDDING_KEY = "get_embedding"
_GET_CONTENT_KEY = "get_content"
_TOP_N_KEY = "top_n"
_COMPRESSION_STATS_KEY = "compression_stats"
_NGRAM_DEDUP_THRESHOLD = 0.7


class DedupRerankStage:
    """Filter/dedup/doc-limit/rerank a fused candidate list (context.candidates).

    Expects `context.metadata` to carry `get_embedding`, `get_content`
    (both `Callable[[str], ...]`) and `top_n` (int); writes the resulting
    `CompressionStats` back to `context.metadata["compression_stats"]`.
    """

    name = "dedup_rerank"

    def __init__(self, config: SearchPipelineConfig):
        self._threshold = ThresholdStage(config.min_confidence)
        self._content_hash_dedup = ContentHashDedupStage()
        self._ngram_dedup = NgramDedupStage(_NGRAM_DEDUP_THRESHOLD)
        self._similarity_dedup = SimilarityDedupStage(config.dedup_threshold)
        self._doc_limit = DocLimitStage(config.max_chunks_per_doc)
        self._rerank = RerankStage(
            enabled=config.reranking_enabled, top_n=config.rerank_top_n
        )

    def _build_cached_content_provider(
        self,
        get_content: Callable[[str], str | None],
    ) -> Callable[[str], str | None]:
        content_cache: dict[str, str | None] = {}

        def cached_get_content(chunk_id: str) -> str | None:
            if chunk_id in content_cache:
                return content_cache[chunk_id]

            content = get_content(chunk_id)
            content_cache[chunk_id] = content
            return content

        return cached_get_content

    def run(self, context: SearchContext) -> SearchContext:
        get_embedding = context.metadata[_GET_EMBEDDING_KEY]
        get_content = context.metadata[_GET_CONTENT_KEY]
        top_n = context.metadata[_TOP_N_KEY]

        cached_get_content = self._build_cached_content_provider(get_content)
        original_count = len(context.candidates)

        stage_context = SearchContext(
            query=context.query,
            candidates=context.candidates,
            metadata={
                "get_content": cached_get_content,
                "get_embedding": get_embedding,
            },
        )

        stage_context = self._threshold.run(stage_context)
        after_threshold = len(stage_context.candidates)

        stage_context = self._content_hash_dedup.run(stage_context)
        after_content_dedup = len(stage_context.candidates)

        stage_context = self._ngram_dedup.run(stage_context)
        after_ngram_dedup = len(stage_context.candidates)

        before_similarity = len(stage_context.candidates)
        stage_context = self._similarity_dedup.run(stage_context)
        after_dedup = len(stage_context.candidates)
        clusters_merged = before_similarity - after_dedup

        stage_context = self._doc_limit.run(stage_context)
        after_doc_limit = len(stage_context.candidates)

        stage_context = self._rerank.run(stage_context)

        final = [
            (chunk_id, max(0.0, min(1.0, score)))
            for chunk_id, score in stage_context.candidates[:top_n]
        ]

        stats = CompressionStats(
            original_count=original_count,
            after_threshold=after_threshold,
            after_content_dedup=after_content_dedup,
            after_ngram_dedup=after_ngram_dedup,
            after_dedup=after_dedup,
            after_doc_limit=after_doc_limit,
            clusters_merged=clusters_merged,
        )

        metadata = dict(context.metadata)
        metadata[_COMPRESSION_STATS_KEY] = stats
        return replace(context, candidates=final, metadata=metadata)
