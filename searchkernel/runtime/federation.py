"""search_anything: the federation entrypoint fusing local + federated sources.

Fans out concurrently to registered SearchableSources (per-source timeout via
runtime.fanout.gather_with_timeout, so one slow or failing source yields no
results rather than crashing the whole query), then applies exactly one
model-agnostic rerank pass over the merged candidate set (retrieve-then-
rerank-once) rather than trusting each source's own scores. Because the
reranker scores candidate text, not vectors, sources never need to share an
embedding space.
"""

import logging
from dataclasses import replace as dataclass_replace
from typing import Any

from searchkernel.domain import ScoredRef
from searchkernel.ports.rerank import Reranker
from searchkernel.runtime.fanout import gather_with_timeout
from searchkernel.runtime.registry import SourceRegistry

logger = logging.getLogger(__name__)

DEFAULT_PER_SOURCE_K = 10
DEFAULT_PER_SOURCE_TIMEOUT_S = 5.0


async def search_anything(
    query: str,
    *,
    registry: SourceRegistry,
    reranker: Reranker,
    sources: list[str] | None = None,
    top_n: int = 10,
    per_source_k: int = DEFAULT_PER_SOURCE_K,
    per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S,
    filters: dict[str, Any] | None = None,
) -> list[ScoredRef]:
    """Fuse the registered sources into one reranked list of ScoredRefs.

    Args:
        query: The search query string.
        registry: SourceRegistry holding the candidate SearchableSources.
        reranker: Scores the merged candidate texts once, cross-source.
        sources: Optional subset of source_kind names to fan out to.
                 If None, every registered source is queried.
        top_n: Maximum number of results to return.
        per_source_k: How many candidates to request from each source.
        per_source_timeout_s: Timeout applied independently to each source.
        filters: Optional source-specific filters (opaque to core).

    Returns:
        A single ranked list of ScoredRefs, ordered by the reranker's score
        descending, truncated to top_n.
    """
    selected = registry.select(sources)
    if not selected:
        return []

    per_source_results = await gather_with_timeout(
        [source.search(query, per_source_k, filters) for source in selected],
        per_timeout_s=per_source_timeout_s,
    )

    candidates: list[ScoredRef] = []
    for results in per_source_results:
        if results:
            candidates.extend(results)

    if not candidates:
        return []

    texts = [candidate.metadata.get("text", "") for candidate in candidates]
    rerank_scores = reranker.rerank(query, texts)

    reranked = [
        dataclass_replace(
            candidate,
            score=rerank_score,
            metadata={**candidate.metadata, "source_score": candidate.score},
        )
        for candidate, rerank_score in zip(candidates, rerank_scores)
    ]
    reranked.sort(key=lambda ref: ref.score, reverse=True)
    return reranked[:top_n]
