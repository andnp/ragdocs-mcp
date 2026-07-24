"""SearchStage contract: composable pure transforms over a SearchContext.

A query pipeline is a sequence of stages threading a `SearchContext`
through retrieve -> graph-expand -> fuse -> dedup/rerank -> hydrate, etc.
Each stage is a narrow, independently-testable unit; stages compose by
returning a new context rather than mutating the one they receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchContext:
    """State threaded through a query pipeline.

    Stages must not mutate a context in place; `run()` returns a new
    `SearchContext` (e.g. via `dataclasses.replace`) reflecting the
    stage's output, so a pipeline run is a strict left-to-right fold.
    """

    query: str
    candidates: list[tuple[str, float]] = field(default_factory=list)
    strategy_results: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SearchStage(Protocol):
    """A single composable step of a query (or ingestion) pipeline.

    Implementations must be pure with respect to `context`: given the same
    input they produce the same output, with no hidden mutation of shared
    state between calls (aside from stage-local caches/instrumentation).
    """

    name: str

    def run(self, context: SearchContext) -> SearchContext: ...


@runtime_checkable
class AsyncSearchStage(Protocol):
    """An I/O-bound composable pipeline step (e.g. retrieval).

    Same contract as `SearchStage` -- pure with respect to `context`,
    returns a new context rather than mutating -- but `run` is a
    coroutine. Stages that must await (index lookups, network calls)
    implement this instead of `SearchStage`; a pipeline executor awaits
    `AsyncSearchStage`s and calls `SearchStage`s directly.
    """

    name: str

    async def run(self, context: SearchContext) -> SearchContext: ...
