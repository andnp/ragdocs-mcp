"""NgramDedupStage: the character-ngram-similarity dedup query stage.

One of the per-concern stages the dedup toolkit decomposes into (see the
W4a plan). Delegates straight to deduplicate_by_ngram -- same inputs,
same outputs -- so it is a pure extraction with no behavior change.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.dedup import deduplicate_by_ngram

_GET_CONTENT_KEY = "get_content"


class NgramDedupStage:
    """Drop candidates whose content n-gram-overlaps an already-kept one.

    Expects `context.metadata["get_content"]` (`Callable[[str], str | None]`).
    A no-op below two candidates, matching SearchPipeline.process.
    """

    name = "dedup_ngram"

    def __init__(self, threshold: float = 0.7):
        self._threshold = threshold

    def run(self, context: SearchContext) -> SearchContext:
        if len(context.candidates) <= 1:
            return context

        get_content = context.metadata[_GET_CONTENT_KEY]
        deduped, _removed = deduplicate_by_ngram(
            context.candidates, get_content, self._threshold
        )
        return replace(context, candidates=deduped)
