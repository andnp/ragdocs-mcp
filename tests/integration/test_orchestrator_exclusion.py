from datetime import datetime
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, SearchConfig, ChunkingConfig, LLMConfig
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.indices.graph import GraphStore
from src.indexing.manager import IndexManager
from src.models import Chunk
from src.search.orchestrator import SearchOrchestrator


@pytest.fixture
def config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        search=SearchConfig(
            semantic_weight=1.0,
            keyword_weight=1.0,
            rrf_k_constant=60,
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def indices():
    vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def orchestrator(config, indices):
    vector, keyword, graph = indices
    manager = IndexManager(config, vector, keyword, graph)
    return SearchOrchestrator(vector, keyword, graph, config, manager)


@pytest.mark.asyncio
async def test_orchestrator_query_without_exclusions(orchestrator, config):
    docs_path = Path(config.indexing.documents_path)

    chunk1 = Chunk(
        chunk_id="docs/api_chunk_0",
        doc_id="docs/api",
        content="API authentication using tokens",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_path / "docs" / "api.md"),
        modified_time=datetime.now(),
    )

    orchestrator._vector.add_chunk(chunk1)
    orchestrator._keyword.add_chunk(chunk1)

    results, stats = await orchestrator.query("authentication", top_k=5, top_n=5)

    assert len(results) > 0
    assert any("api" in r.chunk_id for r in results)


@pytest.mark.asyncio
async def test_orchestrator_query_with_exclusions(orchestrator, config):
    docs_path = Path(config.indexing.documents_path)

    chunk1 = Chunk(
        chunk_id="docs/api_chunk_0",
        doc_id="docs/api",
        content="API authentication using tokens",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_path / "docs" / "api.md"),
        modified_time=datetime.now(),
    )

    chunk2 = Chunk(
        chunk_id="docs/guide_chunk_0",
        doc_id="docs/guide",
        content="Authentication guide for users",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_path / "docs" / "guide.md"),
        modified_time=datetime.now(),
    )

    orchestrator._vector.add_chunk(chunk1)
    orchestrator._vector.add_chunk(chunk2)
    orchestrator._keyword.add_chunk(chunk1)
    orchestrator._keyword.add_chunk(chunk2)

    excluded = {"docs/api"}
    results, stats = await orchestrator.query("authentication", top_k=5, top_n=5, excluded_files=excluded)

    assert len(results) > 0
    assert not any("api" in r.chunk_id for r in results)
    assert any("guide" in r.chunk_id for r in results)


@pytest.mark.asyncio
async def test_orchestrator_query_compression_stats_with_exclusions(orchestrator, config):
    docs_path = Path(config.indexing.documents_path)

    chunks = []
    for i in range(5):
        chunk = Chunk(
            chunk_id=f"docs/file{i}_chunk_0",
            doc_id=f"docs/file{i}",
            content=f"API documentation topic {i}",
            metadata={},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=50,
            file_path=str(docs_path / "docs" / f"file{i}.md"),
            modified_time=datetime.now(),
        )
        chunks.append(chunk)
        orchestrator._vector.add_chunk(chunk)
        orchestrator._keyword.add_chunk(chunk)

    excluded = {"docs/file0", "docs/file1"}
    results, stats = await orchestrator.query("API documentation", top_k=10, top_n=5, excluded_files=excluded)

    assert stats.original_count >= 0
    for result in results:
        assert "file0" not in result.chunk_id
        assert "file1" not in result.chunk_id


@pytest.mark.asyncio
async def test_orchestrator_query_multiple_exclusions(orchestrator, config):
    docs_path = Path(config.indexing.documents_path)

    chunks = []
    for name in ["api", "guide", "tutorial", "reference"]:
        chunk = Chunk(
            chunk_id=f"docs/{name}_chunk_0",
            doc_id=f"docs/{name}",
            content=f"Documentation for {name}",
            metadata={},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=50,
            file_path=str(docs_path / "docs" / f"{name}.md"),
            modified_time=datetime.now(),
        )
        chunks.append(chunk)
        orchestrator._vector.add_chunk(chunk)
        orchestrator._keyword.add_chunk(chunk)

    excluded = {"docs/api", "docs/guide"}
    results, stats = await orchestrator.query("documentation", top_k=10, top_n=5, excluded_files=excluded)

    for result in results:
        assert "api" not in result.chunk_id
        assert "guide" not in result.chunk_id
