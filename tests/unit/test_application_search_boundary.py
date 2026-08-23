"""Focused contracts for the application-owned search boundary."""

from dataclasses import replace
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


class _CapturingSearchKernel:
    def __init__(self) -> None:
        self.filters: Mapping[str, object] | None = None

    async def async_search(
        self, query: str, *, limit: int, filters: Mapping[str, object]
    ) -> RecordSearchOutcome:
        self.filters = dict(filters)
        return RecordSearchOutcome()


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

    assert execution.query_execution_stats == {
        "source": "test",
        "lexical_query": False,
    }
    assert execution.results == []


@pytest.mark.asyncio
async def test_use_case_scopes_ordinary_queries_to_document_sources() -> None:
    """Exclude Git history before the SearchKernel performs retrieval.

    Ordinary document search includes Markdown and Drive records while keeping
    pathless and path-addressable Git history out of the retrieval candidate set.
    """
    kernel = _CapturingSearchKernel()

    await ApplicationSearchUseCase(kernel, documents_roots=()).execute(
        SearchRequest(query="ordinary documents", top_n=1)
    )

    assert kernel.filters is not None
    assert kernel.filters["source_kinds"] == ["note", "gdrive"]


@pytest.mark.asyncio
async def test_use_case_keeps_git_history_queries_git_scoped() -> None:
    """Keep explicit Git-history requests restricted to commit records.

    The performance routing default must not widen the dedicated history path.
    """
    kernel = _CapturingSearchKernel()

    await ApplicationSearchUseCase(kernel, documents_roots=()).execute(
        SearchRequest(
            query="commit history",
            top_n=1,
            source_filter=("git_commit",),
        )
    )

    assert kernel.filters is not None
    assert kernel.filters["source_kinds"] == ["git_commit"]


@pytest.mark.asyncio
async def test_use_case_preserves_explicit_multi_source_filters() -> None:
    """Pass caller-selected source combinations through unchanged.

    Explicit filters remain caller policy, even when they include more than
    the ordinary document sources.
    """
    kernel = _CapturingSearchKernel()
    source_filter = ("note", "gdrive", "git_commit")

    await ApplicationSearchUseCase(kernel, documents_roots=()).execute(
        SearchRequest(
            query="all sources",
            top_n=1,
            source_filter=source_filter,
        )
    )

    assert kernel.filters is not None
    assert kernel.filters["source_kinds"] == list(source_filter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_lexical_query"),
    [
        ("app/search.py", True),
        ("get_user_id", True),
        ("FOO-123", True),
        ("how do I search for boundaries", False),
    ],
)
async def test_use_case_reports_lexical_classification_in_diagnostics(
    query: str, expected_lexical_query: bool
) -> None:
    """Carry the lexical classifier's verdict into query diagnostics.

    Downstream consumers of query_execution_stats need to see whether a
    query was recognized as an exact identifier/path/ticket-ID shape.
    """
    async def execute(*_args, **_kwargs):
        return RecordSearchOutcome()

    class SearchKernel:
        async def async_search(
            self, query: str, *, limit: int, filters: Mapping[str, object]
        ) -> RecordSearchOutcome:
            return await execute(query, limit=limit, filters=filters)

    use_case = ApplicationSearchUseCase(SearchKernel(), documents_roots=())

    execution = await use_case.execute(SearchRequest(query=query, top_n=1))

    assert execution.query_execution_stats["lexical_query"] is expected_lexical_query


@pytest.mark.asyncio
async def test_use_case_keeps_exact_single_token_matches() -> None:
    """Return a document when one meaningful query token matches its body.

    Exact lexical matches are user-visible search evidence even when the
    hybrid rank score is intentionally small.
    """
    record = _record()
    record = replace(record, body="API authentication using tokens")
    outcome = RecordSearchOutcome(
        results=(
            RecordSearchResult(
                record=record,
                score=0.01,
                provenance=SearchResultProvenance(strategies=("keyword", "vector")),
            ),
        )
    )

    class SearchKernel:
        async def async_search(
            self, query: str, *, limit: int, filters: Mapping[str, object]
        ) -> RecordSearchOutcome:
            return outcome

    execution = await ApplicationSearchUseCase(
        SearchKernel(), documents_roots=()
    ).execute(SearchRequest(query="authentication", top_n=1))

    assert [result.content for result in execution.results] == [
        "API authentication using tokens"
    ]


@pytest.mark.asyncio
async def test_use_case_excludes_relationship_query_own_target() -> None:
    """A relationship query must not return its own subject as a result.

    "what links to target" reduces to a single meaningful token
    ("target"), which would otherwise let target.md pass credibility on
    nothing more than its own title mentioning "target" -- even though
    it is the subject of the query, not a graph neighbor of it.
    """
    record = replace(
        _record(),
        title="Target",
        body="Root-relative graph target content.",
        metadata={"doc_id": "target", "file_path": "docs/target.md"},
    )
    outcome = RecordSearchOutcome(
        results=(
            RecordSearchResult(
                record=record,
                score=0.03,
                provenance=SearchResultProvenance(strategies=("keyword", "vector")),
            ),
        )
    )

    class SearchKernel:
        async def async_search(
            self, query: str, *, limit: int, filters: Mapping[str, object]
        ) -> RecordSearchOutcome:
            return outcome

    execution = await ApplicationSearchUseCase(
        SearchKernel(), documents_roots=()
    ).execute(SearchRequest(query="what links to target", top_n=1))

    assert execution.results == []


@pytest.mark.asyncio
async def test_use_case_normalizes_trailing_punctuation_for_matches() -> None:
    """Treat sentence punctuation as a separator during lexical matching.

    A body token at the end of a sentence must match the same token in a
    punctuation-free query.
    """
    record = _record()
    outcome = RecordSearchOutcome(
        results=(
            RecordSearchResult(
                record=record,
                score=0.01,
                provenance=SearchResultProvenance(strategies=("keyword", "vector")),
            ),
        )
    )

    class SearchKernel:
        async def async_search(
            self, query: str, *, limit: int, filters: Mapping[str, object]
        ) -> RecordSearchOutcome:
            return outcome

    execution = await ApplicationSearchUseCase(
        SearchKernel(), documents_roots=()
    ).execute(SearchRequest(query="Boundary content", top_n=1))

    assert [result.doc_id for result in execution.results] == ["doc-1"]
