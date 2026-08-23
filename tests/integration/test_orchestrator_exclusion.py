from pathlib import Path

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
        ),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def manager(config):
    return make_record_index_manager(config)


@pytest.fixture
def orchestrator(config, manager):
    return make_search_adapter(manager, config)


def _seed(manager, source_id: str, content: str, file_path: str):
    assert manager.index_record(
        make_record(
            source_id,
            content,
            metadata={"doc_id": source_id, "file_path": file_path},
        )
    )


@pytest.mark.asyncio
async def test_orchestrator_query_without_exclusions(orchestrator, manager, config):
    docs_path = Path(config.indexing.documents_path)
    _seed(manager, "docs/api", "API authentication using tokens", str(docs_path / "docs/api.md"))

    results, _stats, _ = await orchestrator.query("authentication", top_k=5, top_n=5)

    assert len(results) > 0
    assert any("api" in r.chunk_id for r in results)


@pytest.mark.asyncio
async def test_orchestrator_query_with_exclusions(orchestrator, manager, config):
    docs_path = Path(config.indexing.documents_path)
    _seed(manager, "docs/api", "API authentication using tokens", str(docs_path / "docs/api.md"))
    _seed(manager, "docs/guide", "Authentication guide for users", str(docs_path / "docs/guide.md"))

    excluded = {"docs/api"}
    results, _stats, _ = await orchestrator.query(
        "authentication", top_k=5, top_n=5, excluded_files=excluded
    )

    assert len(results) > 0
    assert not any("api" in r.chunk_id for r in results)
    assert any("guide" in r.chunk_id for r in results)


@pytest.mark.asyncio
async def test_orchestrator_query_compression_stats_with_exclusions(
    orchestrator, manager, config
):
    docs_path = Path(config.indexing.documents_path)

    for i in range(5):
        _seed(manager, f"docs/file{i}", f"API documentation topic {i}", str(docs_path / "docs" / f"file{i}.md"))

    excluded = {"docs/file0", "docs/file1"}
    results, stats, _ = await orchestrator.query(
        "API documentation", top_k=10, top_n=5, excluded_files=excluded
    )

    assert stats.original_count >= 0
    for result in results:
        assert "file0" not in result.chunk_id
        assert "file1" not in result.chunk_id


@pytest.mark.asyncio
async def test_orchestrator_query_multiple_exclusions(orchestrator, manager, config):
    docs_path = Path(config.indexing.documents_path)

    for name in ["api", "guide", "tutorial", "reference"]:
        _seed(manager, f"docs/{name}", f"Documentation for {name}", str(docs_path / "docs" / f"{name}.md"))

    excluded = {"docs/api", "docs/guide"}
    results, _stats, _ = await orchestrator.query(
        "documentation", top_k=10, top_n=5, excluded_files=excluded
    )

    for result in results:
        assert "api" not in result.chunk_id
        assert "guide" not in result.chunk_id


@pytest.mark.asyncio
async def test_orchestrator_document_limit_uses_real_chunks(orchestrator, manager, config):
    file_path = str(Path(config.indexing.documents_path) / "docs" / "auth.md")
    assert manager.index_records(
        [
            make_record(
                "docs/auth#overview",
                "Authentication overview for API clients",
                metadata={
                    "doc_id": "docs/auth",
                    "chunk_id": "overview",
                    "file_path": file_path,
                },
            ),
            make_record(
                "docs/auth#tokens",
                "Authentication token rotation details",
                metadata={
                    "doc_id": "docs/auth",
                    "chunk_id": "tokens",
                    "file_path": file_path,
                },
            ),
        ]
    )

    limited, _, _ = await orchestrator.query(
        "authentication",
        top_k=10,
        top_n=5,
        max_chunks_per_doc=1,
    )
    expanded, _, _ = await orchestrator.query(
        "authentication",
        top_k=10,
        top_n=5,
        max_chunks_per_doc=0,
    )

    assert len(limited) == 1
    assert len(expanded) == 2
    assert {result.doc_id for result in expanded} == {"docs/auth"}
    assert all(result.provenance is not None for result in expanded)
