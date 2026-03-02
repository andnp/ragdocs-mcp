from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.search.orchestrator import SearchOrchestrator
    from src.search.pipeline import SearchPipeline

__all__ = [
    "SearchOrchestrator",
    "SearchPipeline",
]


def __getattr__(name: str):
    if name == "SearchOrchestrator":
        from src.search.orchestrator import SearchOrchestrator

        return SearchOrchestrator
    if name == "SearchPipeline":
        from src.search.pipeline import SearchPipeline

        return SearchPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
