from collections.abc import Callable
from dataclasses import dataclass

from src.models import CompressionStats
from src.search.dedup import deduplicate_by_similarity
from src.search.filters import filter_by_confidence, limit_per_document
from src.search.fusion import normalize_scores
from src.search.reranker import ReRanker


@dataclass
class SearchPipelineConfig:
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 0
    dedup_enabled: bool = True
    dedup_threshold: float = 0.85
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
    ) -> tuple[list[tuple[str, float]], CompressionStats]:
        if not fused_results:
            return [], CompressionStats(
                original_count=0,
                after_threshold=0,
                after_doc_limit=0,
                after_dedup=0,
                clusters_merged=0,
            )

        normalized = normalize_scores(fused_results)
        original_count = len(normalized)

        filtered = filter_by_confidence(normalized, self._config.min_confidence)
        after_threshold = len(filtered)

        limited = limit_per_document(filtered, self._config.max_chunks_per_doc)
        after_doc_limit = len(limited)

        clusters_merged = 0
        if self._config.dedup_enabled and len(limited) > 1:
            limited, clusters_merged = deduplicate_by_similarity(
                limited,
                get_embedding,
                self._config.dedup_threshold,
            )
        after_dedup = len(limited)

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
            after_doc_limit=after_doc_limit,
            after_dedup=after_dedup,
            clusters_merged=clusters_merged,
        )

        return final, stats

    def _get_reranker(self) -> ReRanker:
        if self._reranker is None:
            self._reranker = ReRanker(model_name=self._config.rerank_model)
        return self._reranker
