"""FusionStage: the RRF-fuse + calibrate (+ optional time-boost) query stage.

Lifted from SearchOrchestrator's direct use of ScorePipeline behind the
SearchStage contract. Delegates straight to ScorePipeline.run -- same
inputs, same outputs -- so wiring the orchestrator through this stage is
a pure extraction with no behavior change.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.score_pipeline import ScorePipeline, ScorePipelineConfig


class FusionStage:
    """Fuse multi-strategy candidate lists (context.strategy_results) into
    a single scored, calibrated candidate list (context.candidates)."""

    name = "fusion"

    def __init__(self, config: ScorePipelineConfig | None = None):
        self._pipeline = ScorePipeline(config)

    def run(
        self,
        context: SearchContext,
        timestamps: dict[str, datetime] | None = None,
    ) -> SearchContext:
        fused = self._pipeline.run(context.strategy_results, timestamps)
        return replace(context, candidates=fused)
