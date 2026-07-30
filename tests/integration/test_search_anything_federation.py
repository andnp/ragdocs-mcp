"""End-to-end test of search_anything fusing the local kernel source with a
federated source, through the SourceRegistry -- the W4 acceptance surface:
one fused ranked list spanning the local corpus and >=1 external source.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import pytest

from searchkernel.adapters.sources.local import LocalSearchSource
from searchkernel.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from searchkernel.domain import ScoredRef
from searchkernel.indexing.manager import IndexManager
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.models import Chunk
from searchkernel.runtime.federation import search_anything
from searchkernel.runtime.registry import SourceRegistry
from searchkernel.search.orchestrator import SearchOrchestrator


class _StubExternalSource:
    """A federated source standing in for a real external system (e.g. git)."""

    source_kind = "external"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        return [
            ScoredRef(
                source_id="external-1",
                score=0.5,
                source_kind="external",
                metadata={"text": "External system notes on OAuth token refresh."},
            )
        ]


class _StubReranker:
    """Deterministic reranker: score = position of the text in a fixed table."""

    model_name = "stub-reranker"

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores_by_text = scores_by_text

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores_by_text.get(doc, 0.0) for doc in documents]


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


@pytest.fixture
def registry(orchestrator):
    registry = SourceRegistry()
    registry.register(LocalSearchSource(orchestrator))
    registry.register(_StubExternalSource())
    return registry


def _seed(indices):
    chunk = Chunk(
        chunk_id="note_doc_chunk_0",
        doc_id="note_doc",
        content="Authentication documentation covers OAuth flows for the public API.",
        metadata={"source_kind": "note"},
        chunk_index=0,
        header_path="Auth",
        start_pos=0,
        end_pos=10,
        file_path="note_doc.md",
        modified_time=datetime.now(UTC),
    )
    indices["vector"].add_chunk(chunk)
    indices["keyword"].add_chunk(chunk)


@pytest.mark.asyncio
async def test_search_anything_fuses_local_and_external_sources(indices, registry):
    _seed(indices)
    reranker = _StubReranker(
        {
            "Authentication documentation covers OAuth flows for the public API.": 0.9,
            "External system notes on OAuth token refresh.": 0.8,
        }
    )

    results = await search_anything(
        "OAuth authentication",
        registry=registry,
        reranker=reranker,
    )

    source_kinds = {r.source_kind for r in results}
    assert "local" in source_kinds
    assert "external" in source_kinds
    assert len(results) == 2
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
