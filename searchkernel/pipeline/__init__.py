"""Composable query/ingestion pipeline framework (W4a).

`SearchStage`/`SearchContext` define the contract a query-toolkit stage
implements; `PipelineSpec` declares an ordered, pinned sequence of stages.
Concrete stages live under `searchkernel.pipeline.stages`.
"""

from searchkernel.pipeline.spec import PipelineSpec, StageSpec
from searchkernel.pipeline.stage import SearchContext, SearchStage

__all__ = ["PipelineSpec", "SearchContext", "SearchStage", "StageSpec"]
