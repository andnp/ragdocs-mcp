from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from searchkernel.api import SearchResultProvenance
from searchkernel.domain import Record, RecordIdentity

from mcp_markdown_ragdocs.app.search import SearchQuery, build_record_search_policy
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter
from tests.integration._canonical import make_search_adapter


@pytest.mark.asyncio
async def test_search_expands_hyphenated_terms_in_query(adapter, monkeypatch):
    captured = {}

    async def fake_search(query, *, limit, filters):
        captured["query"] = query
        return SimpleNamespace(results=(), failures=(), degraded=False)

    monkeypatch.setattr(adapter._search_kernel._pipeline, "async_search", fake_search)

    await adapter.search_use_case.execute(
        SearchQuery(query="multi-process architecture", top_n=2),
        search=adapter.search,
    )

    assert captured["query"] == "multi-process multiprocess architecture"


@pytest.mark.asyncio
async def test_search_preserves_canonical_outcome_diagnostics(adapter, monkeypatch):
    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=(),
            failures=(),
            degraded=False,
            missing_record_ids=("missing-record",),
            cache_diagnostics=("cache miss",),
            diagnostics=("graph skipped",),
            candidate_count=4,
            candidate_counts={"keyword": 3, "vector": 2},
            stage_timings_ms={"fusion": 1.5},
            trace=None,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="diagnostics", top_n=2),
        search=fake_search,
    )

    assert execution.query_execution_stats == {
        "degraded": True,
        "failures": [],
        "missing_record_ids": ["missing-record"],
        "diagnostics": ["graph skipped"],
        "cache_diagnostics": ["cache miss"],
        "candidate_count": 4,
        "candidate_counts": {"keyword": 3, "vector": 2},
        "stage_timings_ms": {"fusion": 1.5},
        "lexical_query": False,
    }


@pytest.mark.asyncio
async def test_search_penalizes_duplicate_header_paths(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=f"doc-{index}",
            title=f"Doc {index}",
            body=f"Configuration details {index}",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": f"doc-{index}", "file_path": f"spec_{index}.md", "header_path": "# Configuration"},
        )
        for index in range(3)
    ]
    records.append(
        Record(
            source_kind="note",
            source_id="doc-diverse",
            title="Doc Diverse",
            body="Unique configuration architecture details",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "doc-diverse", "file_path": "arch.md", "header_path": "# Architecture Overview"},
        )
    )

    outcome = SimpleNamespace(
        results=[
            SimpleNamespace(record=records[0], score=1.0, provenance=SimpleNamespace(strategies=("keyword",))),
            SimpleNamespace(record=records[1], score=0.99, provenance=SimpleNamespace(strategies=("keyword",))),
            SimpleNamespace(record=records[2], score=0.98, provenance=SimpleNamespace(strategies=("keyword",))),
            SimpleNamespace(record=records[3], score=0.95, provenance=SimpleNamespace(strategies=("keyword",))),
        ],
        failures=(),
        degraded=False,
    )


    async def fake_search(*_args, **_kwargs):
        return outcome

    await adapter.search_use_case.execute(
        SearchQuery(query="configuration", top_n=3, max_chunks_per_doc=0),
        search=fake_search,
    )


@pytest.mark.asyncio
async def test_default_match_keeps_typo_variations(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="typo-target",
        title="Reciprocal Rank Fusion",
        body="Details on reciprocal rank fusion ranking algorithm",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "typo-target", "file_path": "fusion.md", "header_path": "Fusion"},
    )
    outcome = SimpleNamespace(
        results=[
            SimpleNamespace(
                record=record,
                score=0.15,
                provenance=SimpleNamespace(strategies=("vector",)),
            )
        ],
        failures=(),
        degraded=False,
    )

    async def fake_search(*_args, **_kwargs):
        return outcome

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="reciporcal rank fusin", top_n=5),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["typo-target"]







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
    return make_search_adapter(record_manager)


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
async def test_query_forwards_project_context_as_ranking_scope(adapter, monkeypatch):
    captured = {}

    async def fake_search(query, *, limit, filters):
        captured.update(query=query, limit=limit, filters=filters)
        return SimpleNamespace(results=(), failures=(), degraded=False)

    monkeypatch.setattr(adapter, "search", fake_search)

    await adapter.query(
        "authentication",
        top_k=3,
        top_n=2,
        project_context="project-b",
    )

    assert captured["filters"]["ranking_workspace_id"] == "project-b"
    assert captured["filters"]["project_uplift_multiplier"] == pytest.approx(1.2)


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
async def test_document_search_defaults_to_one_chunk_with_multiple_override(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=f"doc-a-chunk-{index}",
            title="Authentication",
            body=f"Chunk {index}",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "doc-a", "chunk_id": f"chunk-{index}"},
        )
        for index in range(2)
    ]
    outcome = SimpleNamespace(
        results=[
            SimpleNamespace(record=record, score=1.0 - index * 0.1, provenance=SimpleNamespace(strategies=()))
            for index, record in enumerate(records)
        ],
        failures=(),
        degraded=False,
    )

    async def fake_search(*_args, **_kwargs):
        return outcome

    default = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=2),
        search=fake_search,
    )
    multiple = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=2, max_chunks_per_doc=0),
        search=fake_search,
    )

    assert len(default.results) == 1
    assert len(multiple.results) == 2


@pytest.mark.asyncio
async def test_abstention_threshold_keeps_relevant_raw_scores_only(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    scores = (0.08, 0.015, 0.004)
    outcome = SimpleNamespace(
        results=[
            SimpleNamespace(
                record=Record(
                    source_kind="note",
                    source_id=f"score-{index}",
                    title="Score fixture",
                    body=f"Fixture {index}",
                    created_at=timestamp,
                    updated_at=timestamp,
                    metadata={"doc_id": f"doc-{index}", "chunk_id": f"chunk-{index}"},
                ),
                score=score,
                provenance=SimpleNamespace(strategies=()),
            )
            for index, score in enumerate(scores)
        ],
        failures=(),
        degraded=False,
    )

    async def fake_search(*_args, **_kwargs):
        return outcome

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="fixture", top_n=5, min_score=0.01),
        search=fake_search,
    )

    assert [result.score for result in execution.results] == [0.08, 0.015]


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

    assert results == []


def test_project_policy_uplifts_only_matching_workspace():
    from searchkernel.search.record_pipeline import (
        RecordSearchCandidate,
        RecordSearchQueryContext,
    )

    policy = build_record_search_policy(object(), project_uplift_multiplier=1.5)
    assert policy is not None
    adjust = policy.query_score_adjuster
    assert adjust is not None
    context = RecordSearchQueryContext(
        "authentication",
        {"ranking_workspace_id": "project-b"},
        20,
    )
    matching = RecordSearchCandidate(
        RecordIdentity("project-b", "note", "doc-b"),
        0.8,
        SearchResultProvenance(),
    )
    unrelated = RecordSearchCandidate(
        RecordIdentity("project-a", "note", "doc-a"),
        0.8,
        SearchResultProvenance(),
    )

    assert adjust(matching, context) == 1.0
    assert adjust(unrelated, context) == 0.8


@pytest.mark.asyncio
async def test_preferred_project_does_not_retrieve_git_history(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=3,
        project_context="project-b",
        max_chunks_per_doc=0,
    )

    assert {result.project_id for result in results} == {"project-a", "project-c"}


@pytest.mark.asyncio
async def test_query_applies_workspace_identity_without_metadata_project_id(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        project_filter=["project-c"],
    )

    assert [result.chunk_id for result in results] == ["chunk-c"]
    assert results[0].project_id == "project-c"


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
async def test_application_use_case_preserves_raw_rrf_scores_and_order(adapter):
    """Expose the pipeline's raw scores in its existing ranked order.

    Search consumers use score magnitude and ordering as the observable
    ranking contract, so the adapter must not normalize or reorder results.
    """
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=doc_id,
            title=doc_id,
            body="Authentication guidance",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": doc_id, "file_path": f"{doc_id}.md"},
        )
        for doc_id in ("first", "second")
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(record=records[0], score=0.42, provenance=None),
                SimpleNamespace(record=records[1], score=0.17, provenance=None),
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=2, max_chunks_per_doc=0),
        search=fake_search,
    )

    assert [(result.doc_id, result.score) for result in execution.results] == [
        ("first", 0.42),
        ("second", 0.17),
    ]


@pytest.mark.asyncio
async def test_application_use_case_returns_empty_for_unmatched_query(adapter):
    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=10, min_score=1.0)
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_drops_unrelated_low_score_records(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="unrelated",
        title="Deployment",
        body="Container rollout instructions",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "unrelated", "file_path": "deploy.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.004,
                    provenance=SimpleNamespace(strategies=("vector",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement", top_n=5),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_keeps_exact_title_low_score(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="exact-title",
        title="Quantum Entanglement",
        body="Reference notes",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "exact-title", "file_path": "quantum.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.004,
                    provenance=SimpleNamespace(strategies=("vector",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement", top_n=5),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["exact-title"]
    assert execution.results[0].metadata["title"] == "Quantum Entanglement"


@pytest.mark.asyncio
async def test_explicit_zero_score_override_keeps_low_score_record(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="override",
        title="Deployment",
        body="Container rollout instructions",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "override", "file_path": "deploy.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.004,
                    provenance=SimpleNamespace(strategies=("vector",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement", top_n=5, min_score=0.0),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["override"]


@pytest.mark.asyncio
async def test_one_per_document_uses_file_path_when_chunk_doc_id_is_missing(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=f"chunk-{index}",
            title=f"Heading {index}",
            body=f"Body {index}",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"file_path": "guide.md", "header_path": f"Heading {index}"},
        )
        for index in range(2)
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.9 - index * 0.1,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                )
                for index, record in enumerate(records)
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="guide.md", top_n=5),
        search=fake_search,
    )

    assert len(execution.results) == 1
    assert execution.results[0].file_path == "guide.md"


@pytest.mark.asyncio
async def test_filename_query_preserves_score_order(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id="generic",
            title="Other",
            body="Generic content",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "generic", "file_path": "other.md"},
        ),
        Record(
            source_kind="note",
            source_id="target",
            title="Target",
            body="Symbol details",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "target", "file_path": "target.md"},
        ),
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=records[0],
                    score=0.9,
                    provenance=SimpleNamespace(strategies=("vector",)),
                ),
                SimpleNamespace(
                    record=records[1],
                    score=0.2,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                ),
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="target.md", top_n=2, max_chunks_per_doc=0),
        search=fake_search,
    )

    assert [result.file_path for result in execution.results] == ["other.md", "target.md"]


@pytest.mark.asyncio
async def test_document_search_prefers_notes_over_pathless_git_records(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id="note",
            title="Authentication guide",
            body="Documentation",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "note", "file_path": "auth.md"},
        ),
        Record(
            source_kind="git_commit",
            source_id="git:commit",
            title="Authentication fix",
            body="Commit history",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "git:commit", "commit_id": "git:commit"},
        ),
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=records[1],
                    score=0.9,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                ),
                SimpleNamespace(
                    record=records[0],
                    score=0.2,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                ),
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=2, max_chunks_per_doc=0),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["note"]


@pytest.mark.asyncio
async def test_default_documents_exclude_self_referential_pathless_git_results(
    adapter,
):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="git_commit",
        source_id="git:abstention",
        title="Fix search regression",
        body="quantum entanglement recipe for Mars",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "doc_id": "git:abstention",
            "commit_id": "git:abstention",
            "project_id": "mcp-markdown-ragdocs",
        },
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.049,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=1, raw_score=49.0),
                            "vector": SimpleNamespace(rank=1, raw_score=0.28),
                        },
                    ),
                )
            ],
            failures=(),
            degraded=False,
        )

    default_execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement recipe for Mars", top_n=5),
        search=fake_search,
    )
    git_execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="quantum entanglement recipe for Mars",
            top_n=5,
            source_filter=("git_commit",),
        ),
        search=fake_search,
    )

    assert default_execution.results == []
    assert [result.doc_id for result in git_execution.results] == ["git:abstention"]


@pytest.mark.asyncio
async def test_artifact_queries_keep_matching_pathless_git_records(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="git_commit",
        source_id="git:artifact",
        title="Update document tools",
        body=(
            "diff --git a/mcp_markdown_ragdocs/mcp/tools/document_tools.py "
            "b/mcp_markdown_ragdocs/mcp/tools/document_tools.py\n"
            "+ handle_query_documents"
        ),
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "doc_id": "git:artifact",
            "commit_id": "git:artifact",
            "project_id": "mcp-markdown-ragdocs",
        },
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.01,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="document_tools.py handle_query_documents",
            top_n=5,
        ),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["git:artifact"]


@pytest.mark.asyncio
async def test_default_abstention_drops_medium_vector_only_match(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="unrelated",
        title="Deployment",
        body="Container rollout instructions",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "unrelated", "file_path": "deploy.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.028,
                    provenance=SimpleNamespace(strategies=("vector",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement", top_n=5),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_keeps_low_score_lexical_match(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="lexical",
        title="Reference",
        body="Quantum notes",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "lexical", "file_path": "quantum.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.004,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement", top_n=5),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["lexical"]


@pytest.mark.asyncio
async def test_default_abstention_drops_unrelated_hybrid_matches(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=f"unrelated-{index}",
            title="Deployment",
            body="Container rollout instructions",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": f"unrelated-{index}", "file_path": "deploy.md"},
        )
        for index in range(3)
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=score,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=rank, raw_score=1e-6),
                            "vector": SimpleNamespace(rank=rank, raw_score=0.2),
                        },
                    ),
                )
                for rank, (record, score) in enumerate(
                    zip(records, (0.0236, 0.0201, 0.0178), strict=True),
                    start=1,
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement recipe for Mars", top_n=5),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_drops_weak_ranked_hybrid_with_generic_overlap(
    adapter,
):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="cartpole-protocol",
        title="001-cartpole-ppo-structured-missingness",
        body=(
            "Secondary metrics. Evaluation protocol. Use a fixed seed set across "
            "all baselines."
        ),
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "doc_id": "cartpole-protocol",
            "file_path": "001-cartpole-ppo-structured-missingness.md",
            "header_path": "Metrics and Evaluation Protocol",
        },
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.02804878048780488,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=22, raw_score=6.861064460761092),
                            "vector": SimpleNamespace(rank=22, raw_score=0.24118660390377045),
                        },
                    ),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum banana protocol for lunar goats", top_n=5),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_drops_weak_mixed_vocabulary_hybrid(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="live-operations",
        title="Operations",
        body="Live deployment guidance for lunar operations.",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "live-operations", "file_path": "operations.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.031,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=1, raw_score=0.08),
                            "vector": SimpleNamespace(rank=1, raw_score=0.24),
                        },
                    ),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="live lunar-goat quantum", top_n=5),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_keeps_multiple_meaningful_hybrid_tokens(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="goat-protocol",
        title="Evaluation notes",
        body="Protocol evaluation for lunar goats",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "goat-protocol", "file_path": "goats.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.028,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=22, raw_score=1e-6),
                            "vector": SimpleNamespace(rank=22, raw_score=0.2),
                        },
                    ),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum banana protocol for lunar goats", top_n=5),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["goat-protocol"]


@pytest.mark.asyncio
async def test_default_abstention_drops_scoped_unrelated_hybrid_matches(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="scoped-unrelated",
        title="Deployment",
        body="Container rollout instructions",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "doc_id": "scoped-unrelated",
            "file_path": "deploy.md",
            "project_id": "mcp-markdown-ragdocs",
        },
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.0236,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=1, raw_score=1e-6),
                            "vector": SimpleNamespace(rank=1, raw_score=0.2),
                        },
                    ),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="quantum entanglement recipe for Mars",
            top_n=5,
            project_filter=("mcp-markdown-ragdocs",),
        ),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_drops_global_vector_and_stopword_matches(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=source_id,
            title="Deployment",
            body="Container rollout instructions",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": source_id, "file_path": f"{source_id}.md"},
        )
        for source_id in ("vector-a", "vector-b", "keyword-a", "keyword-b")
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=records[0],
                    score=0.0213,
                    provenance=SimpleNamespace(strategies=("vector",)),
                ),
                SimpleNamespace(
                    record=records[1],
                    score=0.0203,
                    provenance=SimpleNamespace(strategies=("vector",)),
                ),
                SimpleNamespace(
                    record=records[2],
                    score=0.0147,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                ),
                SimpleNamespace(
                    record=records[3],
                    score=0.0137,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                ),
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="quantum entanglement recipe for Mars", top_n=5),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_drops_scoped_stopword_only_matches(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id=f"scoped-{index}",
            title="Deployment",
            body="Container rollout instructions",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={
                "doc_id": f"scoped-{index}",
                "file_path": f"scoped-{index}.md",
                "project_id": "mcp-markdown-ragdocs",
            },
        )
        for index in range(5)
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=score,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                )
                for record, score in zip(
                    records,
                    (0.0147, 0.0144, 0.0141, 0.0139, 0.0137),
                    strict=True,
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="quantum entanglement recipe for Mars",
            top_n=5,
            project_filter=("mcp-markdown-ragdocs",),
        ),
        search=fake_search,
    )

    assert execution.results == []


@pytest.mark.asyncio
async def test_default_abstention_keeps_semantic_paraphrase_with_keyword_signal(
    adapter,
):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="token-lifecycle",
        title="Token Lifecycle",
        body="Refresh tokens expire after 30 days. Access tokens expire after 15 minutes.",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "token-lifecycle", "file_path": "tokens.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.018,
                    provenance=SimpleNamespace(
                        strategies=("keyword", "vector"),
                        strategy_details={
                            "keyword": SimpleNamespace(rank=1, raw_score=0.9915),
                            "vector": SimpleNamespace(rank=1, raw_score=0.1755),
                        },
                    ),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="how frequently must access credentials be renewed",
            top_n=1,
        ),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["token-lifecycle"]


@pytest.mark.asyncio
async def test_artifact_symbol_query_preserves_score_order(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            source_kind="note",
            source_id="generic",
            title="Other",
            body="Generic content",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"doc_id": "generic", "file_path": "other.md"},
        ),
        Record(
            source_kind="note",
            source_id="symbol",
            title="Search",
            body="Application search implementation",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={
                "doc_id": "symbol",
                "file_path": "src/search.py",
                "header_path": "ApplicationSearchUseCase",
            },
        ),
    ]

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=records[0],
                    score=0.9,
                    provenance=SimpleNamespace(strategies=("vector",)),
                ),
                SimpleNamespace(
                    record=records[1],
                    score=0.2,
                    provenance=SimpleNamespace(strategies=("keyword",)),
                ),
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(
            query="ApplicationSearchUseCase",
            top_n=2,
            max_chunks_per_doc=0,
        ),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == ["generic", "symbol"]


@pytest.mark.asyncio
async def test_graph_provenance_survives_result_mapping(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="linked",
        title="Linked note",
        body="Related content",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "linked", "file_path": "linked.md"},
    )
    provenance = SimpleNamespace(strategies=("graph",))

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(record=record, score=0.02, provenance=provenance)
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="relationship", top_n=1),
        search=fake_search,
    )

    assert execution.results[0].provenance is provenance
    assert execution.strategy_stats.graph_count == 1


@pytest.mark.asyncio
async def test_default_abstention_keeps_valid_low_score_relationship_result(adapter):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="relationship-low-score",
        title="Unrelated title",
        body="Unrelated body",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "relationship-low-score", "file_path": "link.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.001,
                    provenance=SimpleNamespace(strategies=("graph",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="which pages link to target", top_n=1),
        search=fake_search,
    )

    assert [result.doc_id for result in execution.results] == [
        "relationship-low-score"
    ]


@pytest.mark.asyncio
async def test_default_abstention_rejects_low_score_graph_result_for_ordinary_query(
    adapter,
):
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="ordinary-low-score",
        title="Unrelated title",
        body="Unrelated body",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": "ordinary-low-score", "file_path": "link.md"},
    )

    async def fake_search(*_args, **_kwargs):
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    record=record,
                    score=0.001,
                    provenance=SimpleNamespace(strategies=("graph",)),
                )
            ],
            failures=(),
            degraded=False,
        )

    execution = await adapter.search_use_case.execute(
        SearchQuery(query="authentication", top_n=1),
        search=fake_search,
    )

    assert execution.results == []
