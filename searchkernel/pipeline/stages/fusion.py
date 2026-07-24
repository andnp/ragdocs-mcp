"""FusionStage: composes the fuse/calibrate/recency-boost query stages.

Lifted from SearchOrchestrator's direct use of ScorePipeline behind the
SearchStage contract. Composes the finer-grained RRFFuseStage,
CalibrateStage and RecencyBoostStage in the same order
ScorePipeline.run runs its fuse/calibrate/boost steps -- same inputs,
same outputs -- so this stays a pure extraction with no behavior
change.
"""

from __future__ import annotations

from datetime import datetime

from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.calibrate import CalibrateStage
from searchkernel.pipeline.stages.recency_boost import RecencyBoostStage
from searchkernel.pipeline.stages.rrf_fuse import RRFFuseStage
from searchkernel.search.score_pipeline import ScorePipelineConfig


class FusionStage:
    """Fuse multi-strategy candidate lists (context.strategy_results) into
    a single scored, calibrated candidate list (context.candidates)."""

    name = "fusion"

    def __init__(self, config: ScorePipelineConfig | None = None):
        self._fuse = RRFFuseStage(config)
        self._calibrate = CalibrateStage()
        self._boost = RecencyBoostStage(config)

    def run(
        self,
        context: SearchContext,
        timestamps: dict[str, datetime] | None = None,
    ) -> SearchContext:
        context = self._fuse.run(context)
        context = self._calibrate.run(context)
        return self._boost.run(context, timestamps)
