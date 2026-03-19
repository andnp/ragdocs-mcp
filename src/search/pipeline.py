from collections.abc import Callable
from dataclasses import dataclass

from src.models import CompressionStats
from src.search.dedup import (
    deduplicate_by_content_hash,
    deduplicate_by_ngram,
    deduplicate_by_similarity,
)
from src.search.filters import filter_by_confidence, limit_per_document
from src.search.reranker import ReRanker


@dataclass
class SearchPipelineConfig:
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 0
    dedup_threshold: float = 0.85
    reranking_enabled: bool = True
    rerank_top_n: int = 10


class SearchPipeline:
    def __init__(self, config: SearchPipelineConfig):
        self._config = config
        self._reranker: ReRanker | None = None

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

    def process(
        self,
        fused_results: list[tuple[str, float]],
        get_embedding: Callable[[str], list[float] | None],
        get_content: Callable[[str], str | None],
        query: str,
        top_n: int = 5,
    ) -> tuple[list[tuple[str, float]], CompressionStats]:
        if not fused_results:
            return [], CompressionStats(
                original_count=0,
                after_threshold=0,
                after_content_dedup=0,
                after_ngram_dedup=0,
                after_dedup=0,
                after_doc_limit=0,
                clusters_merged=0,
            )

        cached_get_content = self._build_cached_content_provider(get_content)

        # Scores are already calibrated by orchestrator
        original_count = len(fused_results)

        filtered = filter_by_confidence(fused_results, self._config.min_confidence)
        after_threshold = len(filtered)

        # Content hash dedup (exact text match)
        content_deduped, _ = deduplicate_by_content_hash(filtered, cached_get_content)
        after_content_dedup = len(content_deduped)

        # N-gram dedup (fast character-level similarity pre-filter)
        ngram_deduped = content_deduped
        after_ngram_dedup = after_content_dedup
        if len(content_deduped) > 1:
            ngram_deduped, _ = deduplicate_by_ngram(
                content_deduped,
                cached_get_content,
                0.7,
            )
            after_ngram_dedup = len(ngram_deduped)

        # Semantic dedup
        clusters_merged = 0
        dedup_results = ngram_deduped
        if len(ngram_deduped) > 1:
            dedup_results, clusters_merged = deduplicate_by_similarity(
                ngram_deduped,
                get_embedding,
                self._config.dedup_threshold,
            )
        after_dedup = len(dedup_results)

        # Doc limit (after dedup to maximize diversity)
        limited = limit_per_document(dedup_results, self._config.max_chunks_per_doc)
        after_doc_limit = len(limited)

        if limited and self._config.reranking_enabled:
            reranker = self._get_reranker()
            limited = reranker.rerank(
                query,
                limited,
                cached_get_content,
                self._config.rerank_top_n,
            )

        final = [
            (chunk_id, max(0.0, min(1.0, score))) for chunk_id, score in limited[:top_n]
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

        return final, stats

    def _get_reranker(self) -> ReRanker:
        if self._reranker is None:
            self._reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker
