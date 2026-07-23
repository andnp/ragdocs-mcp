"""Concrete SearchStage implementations lifted from searchkernel.search.*.

Each stage wraps existing, already-tested logic behind the SearchStage
contract; wiring a stage in swaps the orchestrator's call site to go
through the stage rather than the underlying helper directly, with no
change to results.
"""

from searchkernel.pipeline.stages.dedup_rerank import DedupRerankStage
from searchkernel.pipeline.stages.fusion import FusionStage

__all__ = ["DedupRerankStage", "FusionStage"]
