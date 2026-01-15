from collections.abc import Callable
from dataclasses import dataclass

from src.models import CompressionStats
from src.search.dedup import deduplicate_by_content_hash, deduplicate_by_ngram, deduplicate_by_similarity
from src.search.diversity import select_mmr
from src.search.filters import filter_by_confidence, limit_per_document
from src.search.reranker import ReRanker


@dataclass
class SearchPipelineConfig:
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 0
    dedup_enabled: bool = True
    dedup_threshold: float = 0.85
    ngram_dedup_enabled: bool = True
    ngram_dedup_threshold: float = 0.7
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    parent_retrieval_enabled: bool = False
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10


class SearchPipeline:
    def __init__(self, config: SearchPipelineConfig):
        self._config = config
        self._reranker: ReRanker | None = None

    def process(
        self,
        fused_results: list[tuple[str, float]],
        get_embedding: Callable[[str], list[float] | None],
        get_content: Callable[[str], str | None],
        query: str,
        top_n: int = 5,
        query_embedding: list[float] | None = None,
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

        # Scores are already calibrated by orchestrator
        original_count = len(fused_results)

        filtered = filter_by_confidence(fused_results, self._config.min_confidence)
        after_threshold = len(filtered)

        # Content hash dedup (exact text match)
        content_deduped, _ = deduplicate_by_content_hash(filtered, get_content)
        after_content_dedup = len(content_deduped)

        # N-gram dedup (fast character-level similarity pre-filter)
        ngram_deduped = content_deduped
        after_ngram_dedup = after_content_dedup
        if self._config.ngram_dedup_enabled and len(content_deduped) > 1:
            ngram_deduped, _ = deduplicate_by_ngram(
                content_deduped,
                get_content,
                self._config.ngram_dedup_threshold,
            )
            after_ngram_dedup = len(ngram_deduped)

        # MMR or Semantic dedup (mutually exclusive)
        clusters_merged = 0
        diversity_results = ngram_deduped
        if self._config.mmr_enabled and query_embedding is not None and len(ngram_deduped) > 1:
            diversity_results = select_mmr(
                query_embedding,
                ngram_deduped,
                get_embedding,
                self._config.mmr_lambda,
                top_n=len(ngram_deduped),
            )
        elif self._config.dedup_enabled and len(ngram_deduped) > 1:
            diversity_results, clusters_merged = deduplicate_by_similarity(
                ngram_deduped,
                get_embedding,
                self._config.dedup_threshold,
            )
        after_dedup = len(diversity_results)

        # Doc limit (after dedup to maximize diversity)
        limited = limit_per_document(diversity_results, self._config.max_chunks_per_doc)
        after_doc_limit = len(limited)

        if self._config.rerank_enabled and limited:
            reranker = self._get_reranker()
            limited = reranker.rerank(
                query,
                limited,
                get_content,
                self._config.rerank_top_n,
            )

        final = limited[:top_n]

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
            self._reranker = ReRanker(model_name=self._config.rerank_model)
        return self._reranker
