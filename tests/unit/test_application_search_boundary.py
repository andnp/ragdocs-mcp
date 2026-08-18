"""Focused contracts for the application-owned search boundary."""

from datetime import UTC, datetime
from collections.abc import Mapping

import pytest
from searchkernel.api import RecordSearchOutcome, RecordSearchResult, SearchResultProvenance
from searchkernel.domain import Record

from mcp_markdown_ragdocs.app.search import (
    ApplicationSearchUseCase,
    SearchRequest,
    map_kernel_result,
)


def _record() -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id="source-1",
        title="Boundary",
        body="Boundary content",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "doc-1", "chunk_id": "chunk-1", "file_path": "docs/a.md"},
    )


def test_map_kernel_result_preserves_application_result_fields() -> None:
    """Map canonical record data without exposing the kernel result object.

    The application result should retain the fields used by transports.
    """
    mapped = map_kernel_result(
        RecordSearchResult(
            record=_record(),
            score=0.42,
            provenance=SearchResultProvenance(strategies=("keyword",)),
        )
    )

    assert mapped.chunk_id == "chunk-1"
    assert mapped.doc_id == "doc-1"
    assert mapped.file_path == "docs/a.md"
    assert mapped.content == "Boundary content"
    assert mapped.score == 0.42


@pytest.mark.asyncio
async def test_use_case_accepts_injected_diagnostics_port() -> None:
    """Allow diagnostics policy to vary without changing search behavior.

    Diagnostics are supplied by a port while result execution stays stable.
    """
    async def execute(*_args, **_kwargs):
        return RecordSearchOutcome()

    class SearchKernel:
        async def async_search(
            self, query: str, *, limit: int, filters: Mapping[str, object]
        ) -> RecordSearchOutcome:
            return await execute(query, limit=limit, filters=filters)

    use_case = ApplicationSearchUseCase(
        SearchKernel(),
        documents_roots=(),
        diagnostics=lambda outcome: {"source": "test"},
    )

    execution = await use_case.execute(SearchRequest(query="boundary", top_n=1))

    assert execution.query_execution_stats == {"source": "test"}
    assert execution.results == []
