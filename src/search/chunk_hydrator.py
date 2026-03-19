from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import ChunkResult
from src.search.path_utils import extract_doc_id_from_chunk_id, resolve_doc_path


@dataclass(frozen=True)
class HydratedChunk:
    chunk_id: str
    doc_id: str
    header_path: str
    file_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    project_id: str | None = None


class ChunkHydrator:
    def __init__(
        self,
        vector: VectorIndex,
        keyword: KeywordIndex,
        documents_path: Path,
        queue_reindex: Callable[[list[str], str], None] | None = None,
    ) -> None:
        self._vector = vector
        self._keyword = keyword
        self._documents_path = documents_path
        self._queue_reindex = queue_reindex

    def hydrate_chunk_result(self, chunk_id: str, score: float) -> ChunkResult | None:
        hydrated = self.get_chunk_data(
            chunk_id,
            reason="chunk hydration fell back to keyword metadata during result assembly",
        )
        if hydrated is None:
            return None

        metadata = hydrated.metadata
        parent_chunk_id = (
            metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None
        )
        parent_content = None
        if parent_chunk_id:
            parent_content = self._vector.get_parent_content(parent_chunk_id)

        return ChunkResult(
            chunk_id=hydrated.chunk_id,
            doc_id=hydrated.doc_id,
            score=score,
            header_path=hydrated.header_path,
            file_path=hydrated.file_path,
            project_id=hydrated.project_id,
            content=hydrated.content,
            parent_chunk_id=(str(parent_chunk_id) if parent_chunk_id is not None else None),
            parent_content=(str(parent_content) if parent_content is not None else None),
        )

    def get_content(self, chunk_id: str) -> str | None:
        hydrated = self.get_chunk_data(
            chunk_id,
            reason="chunk hydration fell back to keyword metadata during content fetch",
        )
        if hydrated is None:
            return None
        return hydrated.content

    def get_chunk_data(self, chunk_id: str, reason: str) -> HydratedChunk | None:
        vector_chunk = self._vector.get_chunk_by_id(chunk_id)
        if vector_chunk is not None:
            hydrated = self._from_vector_chunk(chunk_id, vector_chunk)
            if self._has_missing_core_fields(hydrated):
                self._queue_if_configured([chunk_id], reason)
                return self._enrich_vector_chunk(chunk_id, hydrated)
            return hydrated

        if "_parent_" in chunk_id:
            fallback_chunk = self._get_child_chunk_for_parent(chunk_id)
            if fallback_chunk is not None:
                self._queue_if_configured(
                    [chunk_id],
                    "parent chunk hydration failed during result assembly",
                )
                return fallback_chunk

        keyword_chunk = self._keyword.get_chunk_by_id(chunk_id)
        if keyword_chunk is not None:
            self._queue_if_configured([chunk_id], reason)
            return self._from_keyword_chunk(chunk_id, keyword_chunk)

        return None

    def _from_vector_chunk(
        self, chunk_id: str, chunk_data: dict[str, Any]
    ) -> HydratedChunk:
        metadata = chunk_data.get("metadata", {})
        doc_id = str(chunk_data.get("doc_id", extract_doc_id_from_chunk_id(chunk_id)))
        project_id = chunk_data.get("project_id")
        if project_id is None and isinstance(metadata, dict):
            project_id = metadata.get("project_id")

        return HydratedChunk(
            chunk_id=str(chunk_data.get("chunk_id", chunk_id)),
            doc_id=doc_id,
            header_path=str(chunk_data.get("header_path", "")),
            file_path=self._resolve_file_path(doc_id, chunk_data.get("file_path")),
            content=str(chunk_data.get("content", "")),
            metadata=metadata if isinstance(metadata, dict) else {},
            project_id=str(project_id) if project_id is not None else None,
        )

    def _enrich_vector_chunk(
        self, chunk_id: str, chunk_data: HydratedChunk
    ) -> HydratedChunk:
        keyword_chunk = self._keyword.get_chunk_by_id(chunk_id)
        if keyword_chunk is None:
            return chunk_data

        fallback_chunk = self._from_keyword_chunk(chunk_id, keyword_chunk)
        return self._merge_hydrated_chunks(chunk_data, fallback_chunk)

    def _from_keyword_chunk(
        self, chunk_id: str, chunk_data: dict[str, Any]
    ) -> HydratedChunk:
        doc_id = str(chunk_data.get("doc_id", extract_doc_id_from_chunk_id(chunk_id)))
        raw_header_path = chunk_data.get("headers")
        title = chunk_data.get("title")
        metadata = {
            "title": str(title) if title is not None else "",
            "tags": self._parse_tags(chunk_data.get("tags")),
        }

        return HydratedChunk(
            chunk_id=str(chunk_data.get("chunk_id", chunk_id)),
            doc_id=doc_id,
            header_path=self._resolve_header_path(raw_header_path, title),
            file_path=self._resolve_file_path(
                doc_id,
                chunk_data.get("source_file"),
            ),
            content=str(chunk_data.get("content", "")),
            metadata=metadata,
            project_id=None,
        )

    def _get_child_chunk_for_parent(self, parent_chunk_id: str) -> HydratedChunk | None:
        doc_id = extract_doc_id_from_chunk_id(parent_chunk_id)
        for child_chunk_id in self._vector.get_chunk_ids_for_document(doc_id):
            if child_chunk_id == parent_chunk_id:
                continue

            child_chunk = self._vector.get_chunk_by_id(child_chunk_id)
            if child_chunk is None:
                continue

            metadata = child_chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            if metadata.get("parent_chunk_id") == parent_chunk_id:
                return self._from_vector_chunk(child_chunk_id, child_chunk)

        return None

    def _has_missing_core_fields(self, chunk_data: HydratedChunk) -> bool:
        return any(
            not value
            for value in (
                chunk_data.chunk_id,
                chunk_data.doc_id,
                chunk_data.file_path,
                chunk_data.content,
            )
        )

    def _merge_hydrated_chunks(
        self, primary: HydratedChunk, fallback: HydratedChunk
    ) -> HydratedChunk:
        return HydratedChunk(
            chunk_id=self._pick_preferred_value(primary.chunk_id, fallback.chunk_id),
            doc_id=self._pick_preferred_value(primary.doc_id, fallback.doc_id),
            header_path=self._pick_preferred_value(
                primary.header_path, fallback.header_path
            ),
            file_path=self._pick_preferred_value(primary.file_path, fallback.file_path),
            content=self._pick_preferred_value(primary.content, fallback.content),
            metadata=self._merge_metadata(primary.metadata, fallback.metadata),
            project_id=self._pick_preferred_value(primary.project_id, fallback.project_id),
        )

    def _merge_metadata(
        self, primary: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(fallback)
        for key, value in primary.items():
            if not self._is_missing_value(value):
                merged[key] = value
        return merged

    def _pick_preferred_value(self, primary: Any, fallback: Any) -> Any:
        if not self._is_missing_value(primary):
            return primary
        return fallback

    def _is_missing_value(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) == 0
        return False

    def _resolve_header_path(self, raw_header_path: object, title: object) -> str:
        if isinstance(raw_header_path, str) and raw_header_path.strip():
            return raw_header_path
        if isinstance(title, str) and title.strip():
            return title
        return ""

    def _resolve_file_path(self, doc_id: str, raw_file_path: object) -> str:
        if isinstance(raw_file_path, str) and raw_file_path:
            candidate = Path(raw_file_path)
            if candidate.is_absolute():
                return str(candidate)
            if candidate.suffix:
                resolved_candidate = (self._documents_path / candidate).resolve()
                if resolved_candidate.exists():
                    return str(resolved_candidate)
                return raw_file_path

        resolved = resolve_doc_path(doc_id, self._documents_path)
        if resolved is not None:
            return str(resolved)

        if isinstance(raw_file_path, str):
            return raw_file_path
        return ""

    def _parse_tags(self, raw_tags: object) -> list[str]:
        if not isinstance(raw_tags, str) or not raw_tags:
            return []
        return [tag for tag in raw_tags.split(",") if tag]

    def _queue_if_configured(self, chunk_ids: list[str], reason: str) -> None:
        if self._queue_reindex is None:
            return
        self._queue_reindex(chunk_ids, reason)