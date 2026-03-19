import json
from pathlib import Path

import pytest

from src.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig
from src.indexing.manager import IndexManager
from src.indexing.manifest import load_manifest
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from tests.conftest import create_test_document


@pytest.fixture
def manager(tmp_path, shared_embedding_model):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        indexing=IndexingConfig(
            documents_path=str(docs_dir),
            index_path=str(tmp_path / ".index_data"),
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0, recency_bias=0.5),
        llm=LLMConfig(embedding_model="local"),
        chunking=ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=200,
            max_chunk_chars=1500,
            overlap_chars=100,
        ),
    )

    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return IndexManager(config, vector, keyword, graph)


def _wrap_persist_counts(monkeypatch: pytest.MonkeyPatch, manager: IndexManager) -> dict[str, int]:
    counts = {
        "vector": 0,
        "keyword": 0,
        "graph": 0,
        "hash_store": 0,
    }

    original_vector_persist = manager.vector.persist
    original_keyword_persist = manager.keyword.persist
    original_graph_persist = manager.graph.persist
    original_hash_store_persist = manager._hash_store.persist

    def vector_persist(path: Path) -> None:
        counts["vector"] += 1
        original_vector_persist(path)

    def keyword_persist(path: Path) -> None:
        counts["keyword"] += 1
        original_keyword_persist(path)

    def graph_persist(path: Path) -> None:
        counts["graph"] += 1
        original_graph_persist(path)

    def hash_store_persist() -> None:
        counts["hash_store"] += 1
        original_hash_store_persist()

    monkeypatch.setattr(manager.vector, "persist", vector_persist)
    monkeypatch.setattr(manager.keyword, "persist", keyword_persist)
    monkeypatch.setattr(manager.graph, "persist", graph_persist)
    monkeypatch.setattr(manager._hash_store, "persist", hash_store_persist)

    return counts


def test_index_documents_persists_once_for_the_full_batch(
    manager: IndexManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Given a batch of new files, persisting once should flush the whole batch."""
    docs_dir = tmp_path / "docs"
    first = create_test_document(docs_dir, "guide", "# Guide\n\nFirst batch document")
    second = create_test_document(docs_dir, "api", "# API\n\nSecond batch document")
    hash_store_path = tmp_path / ".index_data" / "chunk_hashes.json"

    counts = _wrap_persist_counts(monkeypatch, manager)

    manager.index_documents([first, second], persist=True)

    assert counts == {
        "vector": 1,
        "keyword": 1,
        "graph": 1,
        "hash_store": 1,
    }
    assert hash_store_path.exists()
    assert {"guide", "api"}.issubset(manager.vector.get_document_ids())

    persisted_hashes = json.loads(hash_store_path.read_text())
    assert any(chunk_id.startswith("guide") for chunk_id in persisted_hashes)
    assert any(chunk_id.startswith("api") for chunk_id in persisted_hashes)


def test_remove_documents_persists_once_and_updates_manifest(
    manager: IndexManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Given persisted docs, batched removal should coalesce persistence and manifest updates."""
    docs_dir = tmp_path / "docs"
    first = create_test_document(docs_dir, "guide", "# Guide\n\nFirst batch document")
    second = create_test_document(docs_dir, "api", "# API\n\nSecond batch document")
    index_path = tmp_path / ".index_data"
    hash_store_path = index_path / "chunk_hashes.json"

    manager.index_documents([first, second], persist=True)
    assert {"guide", "api"}.issubset(manager.vector.get_document_ids())

    counts = _wrap_persist_counts(monkeypatch, manager)

    manager.remove_documents(["guide", "api"], persist=True)

    assert counts == {
        "vector": 1,
        "keyword": 1,
        "graph": 1,
        "hash_store": 1,
    }
    assert "guide" not in manager.vector.get_document_ids()
    assert "api" not in manager.vector.get_document_ids()

    manifest = load_manifest(index_path)
    assert manifest is not None
    assert manifest.indexed_files is not None
    assert "guide" not in manifest.indexed_files
    assert "api" not in manifest.indexed_files

    persisted_hashes = json.loads(hash_store_path.read_text())
    assert not any(chunk_id.startswith("guide") for chunk_id in persisted_hashes)
    assert not any(chunk_id.startswith("api") for chunk_id in persisted_hashes)