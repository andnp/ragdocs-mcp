"""Tests for SearchOrchestrator.query's source_filter parameter.

source_filter restricts results to chunks whose metadata carries a
matching source_kind (e.g. "git_commit"), letting callers scope a query
to one ContentSource without a separate storage/search stack per source.
"""

import pytest
from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from tests.integration._canonical import make_record, make_record_index_manager, make_search_adapter


@pytest.fixture
def config(tmp_path):
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index_data"),
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0),
        llm=LLMConfig(embedding_model="local"),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def manager(config):
    return make_record_index_manager(config)


@pytest.fixture
def orchestrator(config, manager):
    return make_search_adapter(manager, config)


def _make_record(source_id: str, content: str, source_kind: str):
    return make_record(
        source_id,
        content,
        source_kind=source_kind,
        metadata={"doc_id": source_id, "source_kind": source_kind},
    )


def _seed_mixed_sources(manager):
    note_chunk = _make_record(
        "note_doc",
        "Authentication documentation covers OAuth flows for the public API.",
        "note",
    )
    commit_chunk = _make_record(
        "git:abc123",
        "Fix authentication bug in the public API login handler.",
        "git_commit",
    )
    assert manager.index_records([note_chunk, commit_chunk])


@pytest.mark.asyncio
async def test_source_filter_restricts_results_to_matching_source_kind(
    manager, orchestrator
):
    _seed_mixed_sources(manager)

    results, _, _ = await orchestrator.query(
        "authentication public API",
        top_k=10,
        top_n=10,
        source_filter=["git_commit"],
    )

    assert results
    assert all(result.doc_id == "git:abc123" for result in results)


@pytest.mark.asyncio
async def test_source_filter_none_excludes_pathless_git_sources(manager, orchestrator):
    _seed_mixed_sources(manager)

    results, _, _ = await orchestrator.query(
        "authentication public API",
        top_k=10,
        top_n=10,
    )

    doc_ids = {result.doc_id for result in results}
    assert "note_doc" in doc_ids
    assert "git:abc123" not in doc_ids


@pytest.mark.asyncio
async def test_source_filter_excluding_all_sources_returns_no_results(
    manager, orchestrator
):
    _seed_mixed_sources(manager)

    results, _, _ = await orchestrator.query(
        "authentication public API",
        top_k=10,
        top_n=10,
        source_filter=["jira"],
    )

    assert results == []
