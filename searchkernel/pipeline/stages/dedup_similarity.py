"""SimilarityDedupStage: the embedding-cosine-similarity dedup query stage.

One of the per-concern stages the dedup toolkit decomposes into (see the
W4a plan). Delegates straight to deduplicate_by_similarity -- same
inputs, same outputs -- so it is a pure extraction with no behavior
change.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.dedup import deduplicate_by_similarity

_GET_EMBEDDING_KEY = "get_embedding"


class SimilarityDedupStage:
    """Drop candidates whose embedding cosine-matches an already-kept one.

    Expects `context.metadata["get_embedding"]`
    (`Callable[[str], list[float] | None]`). A no-op below two
    candidates, matching SearchPipeline.process.
    """

    name = "dedup_similarity"

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold

    def run(self, context: SearchContext) -> SearchContext:
        if len(context.candidates) <= 1:
            return context

        get_embedding = context.metadata[_GET_EMBEDDING_KEY]
        deduped, _clusters_merged = deduplicate_by_similarity(
            context.candidates, get_embedding, self._threshold
        )
        return replace(context, candidates=deduped)
