"""Structured response helpers for MCP document tools."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mcp_markdown_ragdocs.models import ChunkResult


@dataclass(frozen=True)
class QueryDocumentsResultEnvelopeItem:
    chunk_id: str
    doc_id: str
    file_path: str
    header_path: str
    score: float
    content: str
    project_id: str | None = None
    parent_chunk_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "header_path": self.header_path,
            "score": self.score,
            "content": self.content,
        }
        if self.project_id is not None:
            result["project_id"] = self.project_id
        if self.parent_chunk_id is not None:
            result["parent_chunk_id"] = self.parent_chunk_id
        return result


@dataclass(frozen=True)
class QueryDocumentsResponseEnvelope:
    status: str
    results: tuple[QueryDocumentsResultEnvelopeItem, ...]
    message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "results": [result.to_dict() for result in self.results],
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def render_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def build_query_documents_response_envelope(
    *,
    results: list[ChunkResult],
) -> QueryDocumentsResponseEnvelope:
    result_items = tuple(
        QueryDocumentsResultEnvelopeItem(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            file_path=result.file_path,
            header_path=result.header_path,
            score=result.score,
            content=result.content,
            project_id=result.project_id,
            parent_chunk_id=result.parent_chunk_id,
        )
        for result in results
    )

    return QueryDocumentsResponseEnvelope(
        status="ok",
        results=result_items,
    )


def build_query_documents_status_envelope(
    *,
    status: str,
    payload: dict[str, object],
) -> QueryDocumentsResponseEnvelope:
    message = payload.get("message")
    if not isinstance(message, str):
        message = payload.get("details")
    if not isinstance(message, str):
        message = None

    return QueryDocumentsResponseEnvelope(
        status=status,
        results=(),
        message=message,
        error=str(payload["error"]) if isinstance(payload.get("error"), str) else None,
    )


def build_query_documents_validation_error(
    *, query: str, message: str
) -> QueryDocumentsResponseEnvelope:
    return QueryDocumentsResponseEnvelope(
        status="error",
        results=(),
        message=message,
        error="validation_error",
    )


def build_compact_document_results_response(
    results: list[ChunkResult],
) -> str:
    """Render the shared compact result contract used by document searches."""
    items = [
        QueryDocumentsResultEnvelopeItem(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            file_path=result.file_path,
            header_path=result.header_path,
            score=result.score,
            content=result.content,
            project_id=result.project_id,
            parent_chunk_id=result.parent_chunk_id,
        ).to_dict()
        for result in results
    ]
    return json.dumps(
        {"status": "ok", "results": items},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_compact_error_response(message: str) -> str:
    return json.dumps(
        {"status": "error", "error": "validation_error", "message": message},
        ensure_ascii=False,
        separators=(",", ":"),
    )
