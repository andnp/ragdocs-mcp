"""DocLimitStage: the per-document result-diversity cap query stage.

One of the per-concern stages the dedup/rerank toolkit decomposes into
(see the W4a plan). Delegates straight to limit_per_document -- same
inputs, same outputs -- so it is a pure extraction with no behavior
change.

Note: the plan lists this concern as "MMR"; today's implementation is a
simple per-document chunk cap rather than a true maximal-marginal-
-relevance reranker, so this stage is named for what actually exists.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.filters import limit_per_document


class DocLimitStage:
    """Cap the number of candidates kept per source document."""

    name = "doc_limit"

    def __init__(self, max_chunks_per_doc: int = 0):
        self._max_chunks_per_doc = max_chunks_per_doc

    def run(self, context: SearchContext) -> SearchContext:
        limited = limit_per_document(context.candidates, self._max_chunks_per_doc)
        return replace(context, candidates=limited)
