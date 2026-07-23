from datetime import datetime, timezone

import pytest

from searchkernel.config import ChunkingConfig, Config, IndexingConfig, LLMConfig, SearchConfig
from searchkernel.domain import Record, RecordStatus
from searchkernel.indexing.manager import IndexManager
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex


@pytest.fixture
def manager(tmp_path, shared_embedding_model):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        indexing=IndexingConfig(
            documents_path=str(docs_dir), index_path=str(tmp_path / ".index_data")
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0),
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


def _make_commit_record(source_id="git:abc123", body="Fix the login bug in the API handler."):
    now = datetime.now(timezone.utc)
    return Record(
        source_kind="git_commit",
        source_id=source_id,
        title="Fix login bug",
        body=body,
        created_at=now,
        updated_at=now,
        metadata={"author": "Test User <test@example.com>", "files_changed": ["api.py"]},
        uri=None,
        status=RecordStatus.ACTIVE,
    )


def test_index_record_adds_record_to_vector_and_keyword_indices(manager):
    record = _make_commit_record()

    manager.index_record(record)

    assert "git:abc123" in manager.vector.get_document_ids()
    chunk_ids = manager.vector.get_chunk_ids_for_document("git:abc123")
    assert chunk_ids

    chunk = manager.vector.get_chunk_by_id(chunk_ids[0])
    assert chunk is not None
    assert chunk["metadata"]["source_kind"] == "git_commit"
    assert chunk["metadata"]["source_id"] == "git:abc123"


def test_index_record_skips_reindex_when_content_unchanged(manager):
    record = _make_commit_record()

    manager.index_record(record)
    chunk_ids_before = manager.vector.get_chunk_ids_for_document("git:abc123")

    manager.index_record(record)
    chunk_ids_after = manager.vector.get_chunk_ids_for_document("git:abc123")

    assert chunk_ids_before == chunk_ids_after


def test_index_record_reindexes_when_content_changes(manager):
    record = _make_commit_record(body="Original message.")
    manager.index_record(record)

    updated = _make_commit_record(body="Updated message with different content entirely.")
    manager.index_record(updated)

    chunk_ids = manager.vector.get_chunk_ids_for_document("git:abc123")
    chunk = manager.vector.get_chunk_by_id(chunk_ids[0])
    assert "Updated message" in chunk["content"]


def test_index_record_handles_multiple_distinct_records(manager):
    first = _make_commit_record(source_id="git:aaa", body="First commit changes the parser.")
    second = _make_commit_record(source_id="git:bbb", body="Second commit changes the renderer.")

    manager.index_record(first)
    manager.index_record(second)

    doc_ids = manager.vector.get_document_ids()
    assert "git:aaa" in doc_ids
    assert "git:bbb" in doc_ids
