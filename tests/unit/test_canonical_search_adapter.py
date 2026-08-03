from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from searchkernel.domain import Record

from mcp_markdown_ragdocs.search import CanonicalSearchAdapter


@pytest.fixture
def adapter(record_manager) -> CanonicalSearchAdapter:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id="doc-a",
            title="Authentication",
            body="Authentication guidance",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={
                "doc_id": "doc-a",
                "chunk_id": "chunk-a",
                "project_id": "project-a",
                "file_path": "a.md",
                "header_path": "Auth",
            },
        ),
        Record(
            source_kind="git_commit",
            source_id="doc-b",
            title="Authentication fix",
            body="Git commit authentication fix",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={
                "doc_id": "doc-b",
                "chunk_id": "chunk-b",
                "project_id": "project-b",
                "file_path": "repo",
                "header_path": "Commit",
            },
        ),
    ]
    assert record_manager.index_records(records) is True
    return CanonicalSearchAdapter(record_manager)


@pytest.mark.asyncio
async def test_query_preserves_chunk_result_semantics(adapter):
    results, stats, strategy = await adapter.query(
        "authentication",
        top_k=10,
        top_n=1,
    )

    assert [result.chunk_id for result in results] == ["chunk-a"]
    assert results[0].doc_id == "doc-a"
    assert results[0].file_path == "a.md"
    assert stats.original_count == 2
    assert strategy.keyword_count == 1


@pytest.mark.asyncio
async def test_query_forwards_contract_filters_and_preserves_scores(adapter, monkeypatch):
    captured = {}

    async def fake_search(query, *, limit, filters):
        captured.update(query=query, limit=limit, filters=filters)
        return SimpleNamespace(results=(), failures=(), degraded=False)

    monkeypatch.setattr(adapter, "search", fake_search)

    await adapter.query(
        "authentication",
        top_k=3,
        top_n=2,
        project_filter=["project-a"],
        source_filter=["note"],
        excluded_files={"private.md"},
        min_score=0.4,
        similarity_threshold=0.9,
        max_chunks_per_doc=1,
    )

    assert captured == {
        "query": "authentication",
        "limit": 8,
        "filters": {
            "source_kinds": ["note"],
            "project_ids": ["project-a"],
            "excluded_files": ["private.md"],
            "min_score": 0.4,
            "similarity_threshold": 0.9,
            "max_chunks_per_doc": 1,
        },
    }


@pytest.mark.asyncio
async def test_one_per_document_overfetches_and_limits_chunks(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=2,
        top_n=2,
        max_chunks_per_doc=1,
    )

    assert len(results) == 2
    assert len({result.doc_id for result in results}) == 2


@pytest.mark.asyncio
async def test_query_applies_canonical_source_filter(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        source_filter=["git_commit"],
    )

    assert [result.chunk_id for result in results] == ["chunk-b"]


@pytest.mark.asyncio
async def test_query_applies_project_filter_without_legacy_orchestrator(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        project_filter=["project-b"],
    )

    assert [result.chunk_id for result in results] == ["chunk-b"]


@pytest.mark.asyncio
async def test_empty_query_returns_no_results(adapter):
    results, _, _ = await adapter.query("", top_k=10, top_n=10)

    assert results == []
