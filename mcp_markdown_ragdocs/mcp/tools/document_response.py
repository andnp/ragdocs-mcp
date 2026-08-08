"""Structured response helpers for MCP document tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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
    diagnostics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "results": [result.to_dict() for result in self.results],
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.error is not None:
            payload["error"] = self.error
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        return payload

    def render_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def build_query_documents_response_envelope(
    *,
    results: list[ChunkResult],
    diagnostics: dict[str, object] | None = None,
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
        diagnostics=diagnostics,
    )


def build_query_documents_status_envelope(
    *,
    status: str,
    payload: dict[str, object],
    include_diagnostics: bool = False,
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
        diagnostics=(build_query_documents_diagnostics(payload) if include_diagnostics else None),
    )


_MAX_DIAGNOSTIC_FAILURES = 8
_MAX_DIAGNOSTIC_VALUE_LENGTH = 256
_MAX_DIAGNOSTIC_LIST_ITEMS = 32


def _bounded_value(value: Any, *, depth: int = 0) -> object:
    if depth >= 4:
        return str(value)[:_MAX_DIAGNOSTIC_VALUE_LENGTH]
    if isinstance(value, str):
        return value[:_MAX_DIAGNOSTIC_VALUE_LENGTH]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key)[:64]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_DIAGNOSTIC_LIST_ITEMS]
        ]
    return str(value)[:_MAX_DIAGNOSTIC_VALUE_LENGTH]


def build_query_documents_diagnostics(
    payload: dict[str, object],
    *,
    compression_stats: Any = None,
    strategy_stats: Any = None,
    query_execution_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    execution = query_execution_stats or {}
    compression = compression_stats.to_dict() if compression_stats is not None else {}
    outcome_counts = execution.get("candidate_counts")
    if isinstance(outcome_counts, dict):
        candidate_counts = _bounded_value(outcome_counts)
    else:
        candidate_counts = {
            key: compression[key]
            for key in (
                "original_count",
                "after_threshold",
                "after_content_dedup",
                "after_ngram_dedup",
                "after_dedup",
                "after_doc_limit",
            )
            if key in compression
        }
    failures = execution.get("failures", payload.get("failures", []))
    if not failures and payload.get("status") == "error":
        failures = [payload.get("details") or payload.get("error") or "search failed"]
    if isinstance(failures, (tuple, set)):
        failures = list(failures)
    elif not isinstance(failures, list):
        failures = [str(failures)] if failures else []
    failures = [
        str(item)[:_MAX_DIAGNOSTIC_VALUE_LENGTH]
        for item in failures[:_MAX_DIAGNOSTIC_FAILURES]
    ]
    degraded = bool(execution.get("degraded", payload.get("degraded", False)))
    degraded = degraded or payload.get("status") in {"error", "partial"}
    return {
        "candidate_counts": candidate_counts,
        "pipeline_diagnostics": _bounded_value(execution),
        "degraded": degraded,
        "failures": failures,
        "final_strategy_counts": _bounded_value(
            strategy_stats.to_dict() if strategy_stats is not None else {}
        ),
    }


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
