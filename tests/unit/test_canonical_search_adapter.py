from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from searchkernel.domain import Record

from mcp_markdown_ragdocs.app.search import SearchQuery
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter


@pytest.fixture
def adapter(record_manager) -> CanonicalSearchAdapter:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            workspace_id="project-a",
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
            workspace_id="project-b",
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
        Record(
            workspace_id="project-c",
            source_kind="note",
            source_id="doc-c",
            title="Workspace authentication",
            body="Workspace authentication guidance",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "doc-c", "chunk_id": "chunk-c"},
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
    assert results[0].metadata["source_kind"] == "note"
    assert results[0].metadata["source_id"] == "doc-a"
    assert results[0].metadata["workspace_id"] == "project-a"
    assert stats.original_count == 3
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
        "limit": 20,
        "filters": {
            "source_kinds": ["note"],
            "workspace_id": "project-a",
            "excluded_files": ["private.md"],
            "min_score": 0.4,
            "similarity_threshold": 0.9,
            "max_chunks_per_doc": 1,
        },
    }


@pytest.mark.asyncio
async def test_query_forwards_multiple_project_scopes(adapter, monkeypatch):
    captured = {}

    async def fake_search(query, *, limit, filters):
        captured.update(query=query, limit=limit, filters=filters)
        return SimpleNamespace(results=(), failures=(), degraded=False)

    monkeypatch.setattr(adapter, "search", fake_search)

    await adapter.query(
        "authentication",
        top_k=3,
        top_n=2,
        project_filter=["project-a", "project-b"],
    )

    assert captured["filters"]["project_ids"] == ["project-a", "project-b"]


@pytest.mark.asyncio
async def test_application_use_case_overfetches_without_explicit_top_k(
    adapter,
    monkeypatch,
):
    captured = {}

    async def fake_search(query, *, limit, filters):
        captured["limit"] = limit
        return SimpleNamespace(results=(), failures=(), degraded=False)

    monkeypatch.setattr(adapter.search_use_case._pipeline, "async_search", fake_search)

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=1, min_score=0.5)
    )

    assert captured["limit"] == 20
    assert execution.results == []


@pytest.mark.asyncio
async def test_hypothesis_query_requests_semantic_only_mode(adapter, monkeypatch):
    calls = []

    async def fake_search(query, *, limit, filters):
        calls.append((query, limit, filters))
        return SimpleNamespace(results=(), failures=(), degraded=False)

    monkeypatch.setattr(adapter, "search", fake_search)

    await adapter.query("ordinary", top_k=5, top_n=2)
    await adapter.query_with_hypothesis("hypothesis", top_k=5, top_n=2)

    assert "retrieval_mode" not in calls[0][2]
    assert calls[1][0] == "hypothesis"
    assert calls[1][2]["retrieval_mode"] == "semantic_only"


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
async def test_exclusions_ignore_records_without_file_paths(adapter):
    assert adapter._is_excluded({}, {"private.md"}) is False
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        source_filter=["git_commit"],
        excluded_files={"private.md"},
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
async def test_query_applies_workspace_identity_without_metadata_project_id(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        project_filter=["project-c"],
    )

    assert [result.chunk_id for result in results] == ["chunk-c"]


@pytest.mark.asyncio
async def test_empty_query_returns_no_results(adapter):
    results, _, _ = await adapter.query("", top_k=10, top_n=10)

    assert results == []


@pytest.mark.asyncio
async def test_application_use_case_applies_scope_filters(adapter):
    execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="authentication",
            top_n=10,
            project_filter=("project-b",),
            source_filter=("git_commit",),
        )
    )

    assert [result.chunk_id for result in execution.results] == ["chunk-b"]


@pytest.mark.asyncio
async def test_application_use_case_preserves_raw_rrf_order(adapter):
    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=10)
    )
    scores = [result.score for result in execution.results]

    assert scores == sorted(scores, reverse=True)
    assert scores
    assert scores[0] <= 1.0


@pytest.mark.asyncio
async def test_application_use_case_returns_empty_for_unmatched_query(adapter):
    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=10, min_score=1.0)
    )

    assert execution.results == []
