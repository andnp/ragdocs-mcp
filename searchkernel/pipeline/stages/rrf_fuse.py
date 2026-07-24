"""RRFFuseStage: the multi-strategy RRF-fuse query stage.

One of the per-concern stages the fusion toolkit decomposes into (see
the W4a plan). Delegates straight to ScorePipeline.fuse -- same inputs,
same outputs -- so it is a pure extraction with no behavior change.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.score_pipeline import ScorePipeline, ScorePipelineConfig


class RRFFuseStage:
    """Combine per-strategy candidate lists (context.strategy_results)
    into a single reciprocal-rank-fused candidate list."""

    name = "rrf_fuse"

    def __init__(self, config: ScorePipelineConfig | None = None):
        self._pipeline = ScorePipeline(config)

    def run(self, context: SearchContext) -> SearchContext:
        fused = self._pipeline.fuse(context.strategy_results)
        return replace(context, candidates=fused)
