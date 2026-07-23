"""Composable query/ingestion pipeline framework (W4a).

`SearchStage`/`SearchContext` define the contract a query-toolkit stage
implements. Concrete stages live under `searchkernel.pipeline.stages`.
"""

from searchkernel.pipeline.stage import SearchContext, SearchStage

__all__ = ["SearchContext", "SearchStage"]
