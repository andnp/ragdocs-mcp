"""RecencyBoostStage: the time-based score-boost query stage.

One of the per-concern stages the fusion toolkit decomposes into (see
the W4a plan). Delegates straight to ScorePipeline.boost -- same
inputs, same outputs -- so it is a pure extraction with no behavior
change. A no-op when no timestamps are supplied or no time-scoring mode
is configured, matching ScorePipeline.run.

Note: the plan pairs this with "type scoring"; no type-based scoring
exists in the current codebase, so this stage covers recency only.
Per-query timestamps don't fit a pure SearchContext value, so `run`
takes them as an explicit parameter (mirroring FusionStage) rather than
threading them through context.metadata.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.score_pipeline import ScorePipeline, ScorePipelineConfig


class RecencyBoostStage:
    """Apply a configured time-decay/tier boost to calibrated scores."""

    name = "recency_boost"

    def __init__(self, config: ScorePipelineConfig | None = None):
        self._pipeline = ScorePipeline(config)

    def run(
        self,
        context: SearchContext,
        timestamps: dict[str, datetime] | None = None,
    ) -> SearchContext:
        if timestamps is None or self._pipeline.config.time_scoring_mode is None:
            return context

        boosted = self._pipeline.boost(context.candidates, timestamps)
        return replace(context, candidates=boosted)
