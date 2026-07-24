"""RerankStage: the cross-encoder rerank query stage.

One of the per-concern stages the dedup/rerank toolkit decomposes into
(see the W4a plan). Delegates straight to ReRanker.rerank -- same
inputs, same outputs -- so it is a pure extraction with no behavior
change. A no-op when disabled or given no candidates, matching
SearchPipeline.process.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.reranker import ReRanker

_GET_CONTENT_KEY = "get_content"


class RerankStage:
    """Rerank candidates against the query with a cross-encoder.

    Expects `context.metadata["get_content"]` (`Callable[[str], str | None]`).
    """

    name = "rerank"

    def __init__(
        self,
        enabled: bool = True,
        top_n: int = 10,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._enabled = enabled
        self._top_n = top_n
        self._reranker: ReRanker | None = None
        self._model_name = model_name

    def _get_reranker(self) -> ReRanker:
        if self._reranker is None:
            self._reranker = ReRanker(model_name=self._model_name)
        return self._reranker

    def run(self, context: SearchContext) -> SearchContext:
        if not context.candidates or not self._enabled:
            return context

        get_content = context.metadata[_GET_CONTENT_KEY]
        reranked = self._get_reranker().rerank(
            context.query, context.candidates, get_content, self._top_n
        )
        return replace(context, candidates=reranked)
