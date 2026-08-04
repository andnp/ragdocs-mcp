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
