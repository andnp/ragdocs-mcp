"""ThresholdStage: the min-confidence filter query stage.

One of the per-concern stages the dedup/rerank toolkit decomposes into
(see the W4a plan). Delegates straight to filter_by_confidence -- same
inputs, same outputs -- so it is a pure extraction with no behavior
change.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.filters import filter_by_confidence


class ThresholdStage:
    """Drop candidates scoring below a minimum confidence."""

    name = "threshold"

    def __init__(self, min_confidence: float = 0.0):
        self._min_confidence = min_confidence

    def run(self, context: SearchContext) -> SearchContext:
        filtered = filter_by_confidence(context.candidates, self._min_confidence)
        return replace(context, candidates=filtered)
