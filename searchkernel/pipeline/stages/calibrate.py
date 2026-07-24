"""CalibrateStage: the sigmoid confidence-calibration query stage.

One of the per-concern stages the fusion toolkit decomposes into (see
the W4a plan). Delegates straight to calibrate_results with the exact
threshold/steepness ScorePipeline.calibrate hardcodes -- same inputs,
same outputs -- so it is a pure extraction with no behavior change.

Note: the plan pairs this with "normalize"; ScorePipeline.run
deliberately skips normalization before calibration (normalizing first
destroys discrimination -- see score_pipeline.py's docstring), so this
stage only calibrates, matching current behavior.
"""

from __future__ import annotations

from dataclasses import replace

from searchkernel.pipeline.stage import SearchContext
from searchkernel.search.calibration import calibrate_results

_THRESHOLD = 0.02
_STEEPNESS = 150.0


class CalibrateStage:
    """Convert raw fused scores to [0, 1] calibrated confidence scores."""

    name = "calibrate"

    def run(self, context: SearchContext) -> SearchContext:
        calibrated = calibrate_results(
            context.candidates, threshold=_THRESHOLD, steepness=_STEEPNESS
        )
        return replace(context, candidates=calibrated)
