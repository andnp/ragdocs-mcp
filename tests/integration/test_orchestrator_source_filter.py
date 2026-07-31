"""Tests for SearchOrchestrator.query's source_filter parameter.

source_filter restricts results to chunks whose metadata carries a
matching source_kind (e.g. "git_commit"), letting callers scope a query
to one ContentSource without a separate storage/search stack per source.
"""

from datetime import UTC, datetime

import pytest
from searchkernel.domain import Chunk
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.search.orchestrator import SearchOrchestrator

from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.indexing.manager import IndexManager


def _with_hash(chunk):
    """Finalize a freshly-built domain.Chunk (test helper).

    domain.Chunk, unlike the legacy models.Chunk, does not auto-compute
    content_hash in __post_init__, and its metadata dict must stay JSON
    serializable (it flows into index/docstore persistence), so a raw
    datetime `modified_time` is normalized to ISO text.
    """
    if not chunk.content_hash:
        chunk.content_hash = chunk.compute_content_hash()
    modified_time = chunk.metadata.get("modified_time")
    if hasattr(modified_time, "isoformat"):
        chunk.metadata["modified_time"] = modified_time.isoformat()
    return chunk



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
def indices(shared_embedding_model):
    return {
        "vector": VectorIndex(embedding_model=shared_embedding_model),
        "keyword": KeywordIndex(),
        "graph": GraphStore(),
    }


@pytest.fixture
def manager(config, indices):
    return IndexManager(config, indices["vector"], indices["keyword"], indices["graph"])


@pytest.fixture
def orchestrator(config, indices, manager):
    return SearchOrchestrator(
        indices["vector"], indices["keyword"], indices["graph"], config, manager
    )


def _make_chunk(chunk_id: str, doc_id: str, content: str, source_kind: str) -> Chunk:
    return _with_hash(Chunk(chunk_id=chunk_id, record_id=doc_id, content=content, metadata={"source_kind": source_kind, "header_path": "", "start_pos": 0, "end_pos": len(content), "file_path": "", "modified_time": datetime.now(UTC)}, chunk_index=0))


def _seed_mixed_sources(indices):
    note_chunk = _make_chunk(
        "note_doc_chunk_0",
        "note_doc",
        "Authentication documentation covers OAuth flows for the public API.",
        "note",
    )
    commit_chunk = _make_chunk(
        "git:abc123_chunk_0",
        "git:abc123",
        "Fix authentication bug in the public API login handler.",
        "git_commit",
    )
    indices["vector"].add_chunk(note_chunk)
    indices["vector"].add_chunk(commit_chunk)
    indices["keyword"].add_chunk(note_chunk)
    indices["keyword"].add_chunk(commit_chunk)


@pytest.mark.asyncio
async def test_source_filter_restricts_results_to_matching_source_kind(
    indices, orchestrator
):
    _seed_mixed_sources(indices)

    results, _, _ = await orchestrator.query(
        "authentication public API",
        top_k=10,
        top_n=10,
        source_filter=["git_commit"],
    )

    assert results
    assert all(result.record_id == "git:abc123" for result in results)


@pytest.mark.asyncio
async def test_source_filter_none_returns_all_sources(indices, orchestrator):
    _seed_mixed_sources(indices)

    results, _, _ = await orchestrator.query(
        "authentication public API",
        top_k=10,
        top_n=10,
    )

    doc_ids = {result.record_id for result in results}
    assert "note_doc" in doc_ids
    assert "git:abc123" in doc_ids


@pytest.mark.asyncio
async def test_source_filter_excluding_all_sources_returns_no_results(
    indices, orchestrator
):
    _seed_mixed_sources(indices)

    results, _, _ = await orchestrator.query(
        "authentication public API",
        top_k=10,
        top_n=10,
        source_filter=["jira"],
    )

    assert results == []
