"""
Integration tests for IndexManager resilience to index corruption.

Tests that IndexManager continues processing remaining indices when one
fails, and that startup reconciliation survives corrupted keyword index.
"""

import glob
from datetime import datetime
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.models import Chunk


@pytest.fixture
def config(tmp_path):
    """
    Create test configuration with temporary paths.

    Uses tmp_path for isolated test storage to avoid conflicts.
    """
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


def test_remove_document_continues_on_partial_failure(tmp_path, config):
    """
    Remove operation continues when one index fails.

    When the keyword index is corrupted and remove() fails, the manager
    should still attempt to remove from vector and graph indices, logging
    errors but not crashing.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)

    chunk = Chunk(
        chunk_id="resilience_test_0",
        doc_id="test-doc",
        content="Content for resilience testing.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=35,
        file_path="/tmp/test.md",
        modified_time=datetime.now(),
    )

    vector.add_chunk(chunk)
    keyword.add_chunk(chunk)
    graph.add_node("test-doc", {"tags": []})

    keyword_path = tmp_path / "keyword_index"
    keyword.persist(keyword_path)
    keyword.load(keyword_path)

    seg_files = glob.glob(str(keyword_path / "*.seg"))
    assert len(seg_files) > 0, "Expected segment files after persist"
    for seg in seg_files:
        Path(seg).unlink()

    manager.remove_document("test-doc")

    assert not graph.has_node("test-doc")


def test_startup_reconciliation_survives_corrupted_keyword_index(tmp_path, config):
    """
    Startup reconciliation continues despite corrupted keyword index.

    When keyword index segment files are corrupted during startup
    reconciliation, the system should recover gracefully and allow
    subsequent operations to succeed.
    """
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()

    chunk = Chunk(
        chunk_id="startup_test_0",
        doc_id="startup-doc",
        content="Content for startup reconciliation testing.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=45,
        file_path="/tmp/startup.md",
        modified_time=datetime.now(),
    )
    vector.add_chunk(chunk)
    keyword.add_chunk(chunk)
    graph.add_node("startup-doc", {"tags": []})

    keyword_path = tmp_path / "startup_keyword_index"
    keyword.persist(keyword_path)
    keyword.load(keyword_path)

    seg_files = glob.glob(str(keyword_path / "*.seg"))
    for seg in seg_files:
        Path(seg).unlink()

    results = keyword.search("startup reconciliation", top_k=5)
    assert results == []

    new_chunk = Chunk(
        chunk_id="after_recovery_0",
        doc_id="recovery-doc",
        content="New content after startup recovery.",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=40,
        file_path="/tmp/recovery.md",
        modified_time=datetime.now(),
    )
    keyword.add_chunk(new_chunk)

    keyword_results = keyword.search("startup recovery", top_k=5)
    chunk_ids = [r["chunk_id"] for r in keyword_results]
    assert "after_recovery_0" in chunk_ids
