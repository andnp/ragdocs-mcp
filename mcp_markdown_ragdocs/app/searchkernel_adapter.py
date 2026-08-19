"""Adapt SearchKernel outcomes to application-owned search contracts."""

from __future__ import annotations

from searchkernel.api import Record, RecordSearchOutcome, RecordSearchResult

from mcp_markdown_ragdocs.models import ChunkResult


def build_search_diagnostics(outcome: RecordSearchOutcome) -> dict[str, object]:
    failures = getattr(outcome, "failures", ())
    failure_messages = [
        getattr(failure, "message", str(failure)) for failure in failures
    ]
    diagnostics: dict[str, object] = {
        "degraded": bool(
            getattr(outcome, "degraded", False)
            or failure_messages
            or getattr(outcome, "missing_record_ids", ())
        ),
        "failures": failure_messages,
        "missing_record_ids": list(getattr(outcome, "missing_record_ids", ())),
        "diagnostics": list(getattr(outcome, "diagnostics", ())),
        "cache_diagnostics": list(getattr(outcome, "cache_diagnostics", ())),
        "candidate_count": int(getattr(outcome, "candidate_count", 0)),
        "candidate_counts": dict(getattr(outcome, "candidate_counts", {})),
        "stage_timings_ms": dict(getattr(outcome, "stage_timings_ms", {})),
    }
    trace = getattr(outcome, "trace", None)
    if trace is not None and callable(getattr(trace, "to_dict", None)):
        diagnostics["trace"] = trace.to_dict()
    return diagnostics


def map_kernel_result(result: RecordSearchResult) -> ChunkResult:
    """Map one canonical kernel result to the application result contract."""
    record = result.record
    metadata = dict(record.metadata)
    metadata.setdefault("record_id", record.storage_key)
    metadata.setdefault("title", record.title)
    metadata.setdefault("workspace_id", record.workspace_id)
    metadata.setdefault("source_kind", record.source_kind)
    metadata.setdefault("source_id", record.source_id)
    project_id = _record_project_id(record)
    metadata.setdefault("project_id", project_id)
    file_path = str(metadata.get("file_path") or "")
    if not file_path and record.uri:
        file_path = record.uri.removeprefix("file://")
    return ChunkResult(
        chunk_id=str(metadata.get("chunk_id", record.source_id)),
        doc_id=str(metadata.get("doc_id", record.source_id)),
        score=result.score,
        header_path=str(metadata.get("header_path") or record.title or ""),
        file_path=file_path,
        project_id=project_id,
        content=record.body,
        parent_chunk_id=metadata.get("parent_chunk_id"),
        provenance=result.provenance,
        metadata=metadata,
    )


__all__ = ["build_search_diagnostics", "map_kernel_result"]


def _record_project_id(record: Record) -> str | None:
    workspace_id = record.workspace_id
    if workspace_id is not None:
        return workspace_id
    project_id = record.metadata.get("project_id")
    return project_id if isinstance(project_id, str) else None
