"""Tests for LocalSearchSource: SearchOrchestrator adapted to SearchableSource.

Verifies the port-conformance mapping (SearchOrchestrator's ChunkResults ->
ScoredRefs carrying candidate text) that the federation rerank step depends
on, over a real orchestrator against in-memory FAISS-backed indices.
"""

from datetime import UTC, datetime

import pytest
from searchkernel.domain import Chunk, ScoredRef
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.ports import SearchableSource

from mcp_markdown_ragdocs.adapters.sources.local import LocalSearchSource
from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.indexing.manager import IndexManager
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter


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
    return CanonicalSearchAdapter(manager)


@pytest.fixture
def source(orchestrator):
    return LocalSearchSource(orchestrator)


def _seed(indices):
    chunk = _with_hash(Chunk(chunk_id="note_doc_chunk_0", record_id="note_doc", content="Authentication documentation covers OAuth flows for the public API.", metadata={"source_kind": "note", "header_path": "Auth", "start_pos": 0, "end_pos": 10, "file_path": "note_doc.md", "modified_time": datetime.now(UTC)}, chunk_index=0))
    indices["vector"].add_chunk(chunk)
    indices["keyword"].add_chunk(chunk)


def test_implements_searchable_source_protocol(source):
    assert isinstance(source, SearchableSource)


def test_source_kind_is_local(source):
    assert source.source_kind == "local"


@pytest.mark.asyncio
async def test_search_maps_chunk_results_to_scored_refs(indices, source):
    _seed(indices)

    results = list(await source.search("authentication public API", k=5))

    assert results
    assert all(isinstance(r, ScoredRef) for r in results)
    top = results[0]
    assert top.source_kind == "local"
    assert top.source_id == "note_doc_chunk_0"
    assert "Authentication documentation covers OAuth flows for the public API." in (
        top.metadata["text"]
    )
    assert top.metadata["doc_id"] == "note_doc"


@pytest.mark.asyncio
async def test_search_no_match_returns_empty(indices, source):
    _seed(indices)

    results = list(await source.search("completely unrelated banana query", k=5))

    assert isinstance(results, list)
