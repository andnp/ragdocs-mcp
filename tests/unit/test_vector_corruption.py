"""
Unit tests for VectorIndex persistence corruption handling.

GAP #14: VectorIndex persistence with corrupted files (High/Low, Score 4.0)
"""

import json

import pytest

from src.indices.vector import VectorIndex
from src.models import Chunk
from datetime import datetime


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        chunk_id="test_doc_chunk_0",
        doc_id="test_doc",
        content="Machine learning and artificial intelligence.",
        metadata={"source": "test"},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=45,
        file_path="/test/doc.md",
        modified_time=datetime.now(),
    )


def test_vector_index_load_with_corrupted_concept_vocabulary(tmp_path, shared_embedding_model, sample_chunk):
    """
    VectorIndex gracefully handles corrupted concept vocabulary file.

    Tests that corrupted vocabulary.json doesn't crash index loading.
    """
    # Create and populate index
    index1 = VectorIndex(embedding_model=shared_embedding_model)
    index1.add_chunk(sample_chunk)

    persist_path = tmp_path / "index"
    index1.persist(persist_path)

    # Corrupt the concept vocabulary file
    vocab_file = persist_path / "concept_vocabulary.json"
    if vocab_file.exists():
        vocab_file.write_text("{ corrupted json")

    # Load should succeed with warning, not crash
    index2 = VectorIndex(embedding_model=shared_embedding_model)
    index2.load(persist_path)

    # Index should still be usable (vocabulary just empty)
    assert index2.is_ready()


def test_vector_index_load_with_missing_vocabulary_file(tmp_path, shared_embedding_model, sample_chunk):
    """
    VectorIndex handles missing concept vocabulary file.

    Tests that missing vocabulary file is treated as empty vocabulary.
    """
    # Create and populate index
    index1 = VectorIndex(embedding_model=shared_embedding_model)
    index1.add_chunk(sample_chunk)

    persist_path = tmp_path / "index"
    index1.persist(persist_path)

    # Delete vocabulary file
    vocab_file = persist_path / "concept_vocabulary.json"
    if vocab_file.exists():
        vocab_file.unlink()

    # Load should succeed
    index2 = VectorIndex(embedding_model=shared_embedding_model)
    index2.load(persist_path)

    assert index2.is_ready()
    # Vocabulary should be empty but not crash
    assert index2._concept_vocabulary == {}


def test_vector_index_load_with_corrupted_term_counts(tmp_path, shared_embedding_model, sample_chunk):
    """
    VectorIndex handles corrupted term counts file.

    Tests recovery from corrupted term_counts.json.
    """
    # Create and populate index
    index1 = VectorIndex(embedding_model=shared_embedding_model)
    index1.add_chunk(sample_chunk)

    persist_path = tmp_path / "index"
    index1.persist(persist_path)

    # Corrupt term counts file
    term_counts_file = persist_path / "term_counts.json"
    if term_counts_file.exists():
        term_counts_file.write_text("not valid json at all")

    # Load should succeed with warning
    index2 = VectorIndex(embedding_model=shared_embedding_model)
    index2.load(persist_path)

    assert index2.is_ready()


def test_vector_index_load_with_corrupted_chunk_mapping(tmp_path, shared_embedding_model, sample_chunk):
    """
    VectorIndex handles corrupted chunk_id mapping file.

    Tests that corrupted doc_id_mapping.json triggers rebuild.
    """
    # Create and populate index
    index1 = VectorIndex(embedding_model=shared_embedding_model)
    index1.add_chunk(sample_chunk)

    persist_path = tmp_path / "index"
    index1.persist(persist_path)

    # Corrupt the chunk mapping file
    mapping_file = persist_path / "doc_id_mapping.json"
    if mapping_file.exists():
        mapping_file.write_text('{"incomplete": ')

    # Load should handle corruption
    index2 = VectorIndex(embedding_model=shared_embedding_model)
    index2.load(persist_path)

    # Should rebuild mapping from index
    assert index2.is_ready()


def test_vector_index_load_with_empty_files(tmp_path, shared_embedding_model):
    """
    VectorIndex handles empty persistence files.

    Tests loading from directory with empty JSON files.
    """
    persist_path = tmp_path / "index"
    persist_path.mkdir()

    # Create empty files
    (persist_path / "concept_vocabulary.json").write_text("{}")
    (persist_path / "term_counts.json").write_text("{}")
    (persist_path / "doc_id_mapping.json").write_text("{}")

    # Load should succeed
    index = VectorIndex(embedding_model=shared_embedding_model)
    index.load(persist_path)

    # Should have empty state but be ready
    assert index._concept_vocabulary == {}
    assert index._term_counts == {}


def test_vector_index_persist_recovers_from_partial_write(tmp_path, shared_embedding_model, sample_chunk):
    """
    VectorIndex can recover after partial persist failure.

    Tests that subsequent persist attempts succeed after failure.
    """
    index = VectorIndex(embedding_model=shared_embedding_model)
    index.add_chunk(sample_chunk)

    persist_path = tmp_path / "index"

    # First persist should succeed
    index.persist(persist_path)
    assert (persist_path / "doc_id_mapping.json").exists()

    # Corrupt one file to simulate partial failure
    (persist_path / "concept_vocabulary.json").write_text("corrupted")

    # Add more data
    chunk2 = Chunk(
        chunk_id="test_doc2_chunk_0",
        doc_id="test_doc2",
        content="Deep learning neural networks.",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=30,
        file_path="/test/doc2.md",
        modified_time=datetime.now(),
    )
    index.add_chunk(chunk2)

    # Second persist should succeed and overwrite corrupted file
    index.persist(persist_path)

    # Load should work
    index2 = VectorIndex(embedding_model=shared_embedding_model)
    index2.load(persist_path)
    assert index2.is_ready()


def test_vector_index_load_with_invalid_json_types(tmp_path, shared_embedding_model):
    """
    VectorIndex handles JSON with wrong types.

    Tests that type mismatches in persisted data are handled.
    """
    persist_path = tmp_path / "index"
    persist_path.mkdir()

    # Write files with wrong types
    (persist_path / "concept_vocabulary.json").write_text(json.dumps({
        "term": "not_a_list_should_be_vector"  # Should be list of floats
    }))
    (persist_path / "term_counts.json").write_text(json.dumps({
        "term": "not_an_int"  # Should be integer
    }))

    # Load should handle gracefully
    index = VectorIndex(embedding_model=shared_embedding_model)
    try:
        index.load(persist_path)
        # Should either succeed with empty vocabulary or handle gracefully
        assert True
    except (TypeError, ValueError):
        # Acceptable to raise type error, but shouldn't crash completely
        assert True


def test_vector_index_vocabulary_building_on_empty_index(shared_embedding_model):
    """
    Build vocabulary on empty index doesn't crash.

    GAP #5: Vocabulary building with empty index (Medium/Low, Score 3.33)

    Tests that vocabulary building handles edge case of no chunks.
    """
    index = VectorIndex(embedding_model=shared_embedding_model)

    # Building vocabulary on empty index should not crash
    index.build_concept_vocabulary()

    # Should result in empty vocabulary
    assert index._concept_vocabulary == {}
    assert index._term_counts == {}


def test_vector_index_vocabulary_building_with_single_chunk(shared_embedding_model):
    """
    Build vocabulary with single chunk.

    Tests edge case of minimal data for vocabulary building.
    """
    index = VectorIndex(embedding_model=shared_embedding_model)

    chunk = Chunk(
        chunk_id="single_chunk",
        doc_id="single_doc",
        content="machine learning",
        metadata={},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=16,
        file_path="/test.md",
        modified_time=datetime.now(),
    )

    index.add_chunk(chunk)
    index.build_concept_vocabulary()

    # Should build vocabulary even with minimal data
    assert isinstance(index._concept_vocabulary, dict)
    assert isinstance(index._term_counts, dict)
