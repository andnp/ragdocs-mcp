"""Local search source: adapts SearchOrchestrator as a federated SearchableSource.

Wrapping the kernel's own index behind the SearchableSource port makes the
local corpus just one federated source among others from search_anything's
perspective, fused the same way as any external source.
"""

from collections.abc import Iterable
from typing import Any

from searchkernel.domain import ScoredRef

from mcp_markdown_ragdocs.models import ChunkResult


class LocalSearchSource:
    """SearchableSource wrapping the kernel's own SearchOrchestrator.

    SearchOrchestrator.query is already a coroutine, but the store it drives
    (e.g. the pgvector adapter's psycopg2 driver) is synchronous, so a query
    can block the event loop under the hood. To keep that possibility from
    ever stalling sibling sources in a fan-out, the query runs in its own
    thread (with its own event loop) via asyncio.to_thread, isolating any
    blocking I/O from the caller's loop. If that per-call thread/loop
    overhead ever becomes the bottleneck, the real fix is an async store
    driver (future work) rather than more threading here.
    """

    source_kind = "local"

    def __init__(self, orchestrator: Any):
        self._orchestrator = orchestrator

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        if hasattr(self._orchestrator, "search"):
            source_filter = (filters or {}).get("source_filter")
            canonical_filters = dict(filters or {})
            if source_filter is not None:
                canonical_filters["source_kinds"] = list(source_filter)
            outcome = await self._orchestrator.search(
                query,
                limit=k,
                filters=canonical_filters,
            )
            return [
                ScoredRef(
                    source_id=result.record.source_id,
                    score=result.score,
                    source_kind=self.source_kind,
                    workspace_id=result.record.workspace_id,
                    metadata={
                        **result.record.metadata,
                        "text": result.record.body,
                        "storage_key": result.record.storage_key,
                        "provenance": result.provenance.to_dict(),
                    },
                )
                for result in outcome.results
            ]

        source_filter = (filters or {}).get("source_filter")
        chunk_results, _stats, _strategy_stats = await self._orchestrator.query(
            query,
            top_k=k,
            top_n=k,
            source_filter=source_filter,
        )
        return [self._to_scored_ref(result) for result in chunk_results]

    @staticmethod
    def _to_scored_ref(result: ChunkResult) -> ScoredRef:
        return ScoredRef(
            source_id=result.chunk_id,
            score=result.score,
            source_kind="local",
            metadata={
                "text": result.content,
                "doc_id": result.doc_id,
                "file_path": result.metadata.get("file_path", ""),
                "header_path": result.metadata.get("header_path", ""),
            },
        )
