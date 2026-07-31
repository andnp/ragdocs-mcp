"""Application adapter for the canonical searchkernel record pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from searchkernel import RecordSearchPipeline
from searchkernel.domain import (
    CompressionStats,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    SearchResultProvenance,
    SearchStrategyStats,
)
from mcp_markdown_ragdocs.models import ChunkResult


class _IndexManager(Protocol):
    vector: Any
    keyword: Any


class _LegacyKeywordStore:
    """Expose the existing chunk index through the record-store contract."""

    def __init__(self, manager: _IndexManager) -> None:
        self._manager = manager

    def index(self, records: list[Record]) -> None:
        del records

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RecordHit | tuple[str, float]]:
        return await asyncio.to_thread(self._search_sync, query, k, filters)

    def _search_sync(
        self, query: str, k: int, filters: dict[str, Any] | None
    ) -> list[RecordHit | tuple[str, float]]:
        filters = filters or {}
        source_kinds = set(filters.get("source_kinds") or ())
        workspace_id = filters.get("workspace_id")
        hits: list[RecordHit | tuple[str, float]] = []
        candidates = list(self._manager.keyword.search(query, top_k=k))
        vector_search = getattr(self._manager.vector, "search", None)
        if callable(vector_search):
            vector_results = vector_search(query, top_k=k)
            if isinstance(vector_results, list):
                candidates.extend(vector_results)
        seen: set[str] = set()
        for result in candidates:
            chunk_id = result["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = self._manager.vector.get_chunk_by_id(result["chunk_id"])
            metadata = (chunk or {}).get("metadata", {})
            source_kind = str(metadata.get("source_kind", "note"))
            project_id = metadata.get(
                "workspace_id", metadata.get("project_id")
            )
            if source_kinds and source_kind not in source_kinds:
                continue
            if workspace_id is not None and project_id != workspace_id:
                continue
            hits.append(
                RecordHit(
                    RecordIdentity(
                        str(project_id) if project_id is not None else None,
                        source_kind,
                        result["chunk_id"],
                    ),
                    float(result["score"]),
                )
            )
        return hits


class _LegacyRecordHydrator:
    def __init__(self, manager: _IndexManager) -> None:
        self._manager = manager

    async def hydrate_record(self, identity: RecordIdentity) -> Record | None:
        chunk = self._manager.vector.get_chunk_by_id(identity.source_id)
        if chunk is None:
            return None
        metadata = dict(chunk.get("metadata") or {})
        metadata.update(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": metadata.get(
                    "canonical_source_id", chunk["doc_id"]
                ),
                "chunk_index": chunk.get("chunk_index", 0),
                "header_path": metadata.get("header_path", ""),
                "file_path": metadata.get("file_path", ""),
            }
        )
        now = datetime.now(UTC)
        return Record(
            workspace_id=identity.workspace_id,
            source_kind=identity.source_kind,
            source_id=identity.source_id,
            title=str(metadata.get("title") or metadata["doc_id"]),
            body=str(chunk.get("content", "")),
            created_at=now,
            updated_at=now,
            metadata=metadata,
            uri=str(metadata.get("file_path") or "") or None,
            status=RecordStatus.ACTIVE,
        )


class CanonicalSearchAdapter:
    """Preserve ragdocs' query tuple while using RecordSearchPipeline."""

    def __init__(
        self,
        manager: _IndexManager,
        *,
        documents_path: Path | None = None,
    ) -> None:
        config = getattr(manager, "_config", None)
        indexing = getattr(config, "indexing", None)
        self.documents_path = documents_path or Path(
            getattr(indexing, "documents_path", ".")
        )
        self._vector = manager.vector
        self._keyword = manager.keyword
        self._pipeline = RecordSearchPipeline(
            hydrator=_LegacyRecordHydrator(manager),
            keyword_store=_LegacyKeywordStore(manager),
        )
        self.last_query_execution_stats: dict[str, object] = {}

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ):
        return await self._pipeline.async_search(
            query,
            limit=limit,
            filters=dict(filters or {}),
        )

    async def query(
        self,
        query: str,
        *,
        top_k: int,
        top_n: int,
        pipeline_config: object | None = None,
        project_filter: Sequence[str] | None = None,
        source_filter: Sequence[str] | None = None,
        project_context: str | None = None,
        excluded_files: set[str] | None = None,
        **_: object,
    ) -> tuple[list[ChunkResult], CompressionStats, SearchStrategyStats]:
        del pipeline_config, project_context
        filters: dict[str, object] = {}
        if source_filter:
            filters["source_kinds"] = list(source_filter)
        outcome = await self.search(query, limit=max(top_k, top_n), filters=filters)
        filtered_results = [
            result
            for result in outcome.results
            if (
                not project_filter
                or result.record.metadata.get("project_id") in project_filter
            )
            and not self._is_excluded(result.record.metadata, excluded_files)
        ]
        selected_results = filtered_results[:top_n]
        maximum_score = max(
            (result.score for result in selected_results),
            default=0.0,
        )
        results = [
            self._to_chunk_result(
                result.record,
                result.score / maximum_score if maximum_score else result.score,
                result.provenance,
            )
            for result in selected_results
        ]
        self.last_query_execution_stats = {
            "degraded": outcome.degraded,
            "failures": [failure.message for failure in outcome.failures],
        }
        count = len(results)
        return (
            results,
            CompressionStats(count, count, count, count, count, count, 0),
            SearchStrategyStats(keyword_count=count),
        )

    async def query_with_hypothesis(
        self,
        query: str,
        *,
        top_k: int,
        top_n: int,
        pipeline_config: object | None = None,
        project_filter: Sequence[str] | None = None,
        source_filter: Sequence[str] | None = None,
        project_context: str | None = None,
        excluded_files: set[str] | None = None,
    ):
        return await self.query(
            query,
            top_k=top_k,
            top_n=top_n,
            pipeline_config=pipeline_config,
            project_filter=project_filter,
            source_filter=source_filter,
            project_context=project_context,
            excluded_files=excluded_files,
        )

    async def drain_reindex(self) -> None:
        return None

    @staticmethod
    def _to_chunk_result(
        record: Record,
        score: float,
        provenance: SearchResultProvenance,
    ) -> ChunkResult:
        metadata = record.metadata
        return ChunkResult(
            chunk_id=str(metadata.get("chunk_id", record.source_id)),
            doc_id=str(metadata.get("doc_id", record.source_id)),
            score=score,
            header_path=str(metadata.get("header_path", "")),
            file_path=str(metadata.get("file_path", "")),
            project_id=metadata.get("project_id"),
            content=record.body,
            parent_chunk_id=metadata.get("parent_chunk_id"),
            provenance=provenance,
            metadata=metadata,
        )

    def _is_excluded(
        self,
        metadata: dict[str, object],
        excluded_files: set[str] | None,
    ) -> bool:
        if not excluded_files:
            return False
        file_path = str(metadata.get("file_path", ""))
        path = Path(file_path)
        candidates = {
            file_path,
            path.name,
            path.stem,
            str(path.with_suffix("")),
        }
        try:
            relative = path.resolve().relative_to(self.documents_path)
            candidates.add(str(relative))
            candidates.add(str(relative.with_suffix("")))
        except ValueError:
            pass
        return bool(candidates & excluded_files)


__all__ = ["CanonicalSearchAdapter"]
