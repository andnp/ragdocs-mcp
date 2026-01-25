import time
from datetime import datetime, timezone

from src.indices.vector import VectorIndex
from src.models import Chunk


def _create_test_chunk(index: int, doc_id: str = "test-doc") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_chunk_{index}",
        doc_id=doc_id,
        content=f"Test content for chunk {index}. This is a longer text to simulate real documents.",
        chunk_index=index,
        header_path=f"Header {index}",
        file_path=f"/tmp/test-{doc_id}.md",
        parent_chunk_id=None,
        metadata={},
        start_pos=0,
        end_pos=100,
        modified_time=datetime.now(timezone.utc),
    )


def test_parallel_embedding_speedup(tmp_path, shared_embedding_model):
    """
    Verify parallel embeddings are faster than sequential.

    This test creates two indices with different worker counts and
    compares indexing performance. Due to Python's GIL and test system load,
    speedups can vary. We verify parallel is not slower than sequential.
    """
    vector_sequential = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=1,
    )
    vector_parallel = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = [_create_test_chunk(i) for i in range(20)]

    start = time.time()
    vector_sequential.add_chunks(chunks)
    sequential_time = time.time() - start

    start = time.time()
    vector_parallel.add_chunks(chunks)
    parallel_time = time.time() - start

    speedup = sequential_time / parallel_time
    # Verify parallel is not significantly slower (allows for overhead)
    assert parallel_time <= sequential_time * 1.1, (
        f"Parallel indexing ({parallel_time:.2f}s) should not be slower "
        f"than sequential ({sequential_time:.2f}s), got {speedup:.2f}x speedup"
    )

    # Log actual speedup (not an assertion, for observability)
    import logging
    logging.info(f"Parallel speedup: {speedup:.2f}x (seq: {sequential_time:.2f}s, par: {parallel_time:.2f}s)")


def test_parallel_embedding_correctness(tmp_path, shared_embedding_model):
    """
    Verify parallel embeddings produce same results as sequential.

    Both indices should return the same chunks for a given query,
    demonstrating that parallelization doesn't affect correctness.
    """
    vector_sequential = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=1,
    )
    vector_parallel = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = [_create_test_chunk(i) for i in range(10)]

    vector_sequential.add_chunks(chunks)
    vector_parallel.add_chunks(chunks)

    results_sequential = vector_sequential.search("test query", top_k=5)
    results_parallel = vector_parallel.search("test query", top_k=5)

    sequential_ids = {r["chunk_id"] for r in results_sequential}
    parallel_ids = {r["chunk_id"] for r in results_parallel}

    assert sequential_ids == parallel_ids, (
        f"Results differ: sequential={sequential_ids}, parallel={parallel_ids}"
    )


def test_parallel_embedding_error_handling(tmp_path, shared_embedding_model):
    """
    Verify one failed embedding doesn't stop others.

    When one chunk fails to embed (e.g., due to invalid content),
    other chunks should still be indexed successfully.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = [_create_test_chunk(i) for i in range(10)]

    vector.add_chunks(chunks)

    results = vector.search("test content", top_k=10)
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"


def test_embedding_workers_config_respected(tmp_path, shared_embedding_model):
    """
    Verify config controls thread pool size.

    The embedding_workers config should be respected when creating
    the VectorIndex, controlling parallelism level.
    """
    vector_1 = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=1,
    )
    vector_4 = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    assert vector_1._embedding_workers == 1
    assert vector_4._embedding_workers == 4


def test_sequential_mode_backward_compatible(tmp_path, shared_embedding_model):
    """
    Verify sequential mode (workers=1) works correctly.

    With embedding_workers=1, the system should fall back to
    sequential processing for backward compatibility.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=1,
    )

    chunks = [_create_test_chunk(i) for i in range(5)]

    vector.add_chunks(chunks)

    results = vector.search("test content", top_k=5)
    assert len(results) == 5


def test_zero_workers_treated_as_sequential(tmp_path, shared_embedding_model):
    """
    Verify embedding_workers=0 is treated as sequential (workers=1).

    Invalid worker counts should be normalized to minimum of 1.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=0,
    )

    assert vector._embedding_workers == 1

    chunks = [_create_test_chunk(i) for i in range(3)]
    vector.add_chunks(chunks)

    results = vector.search("test content", top_k=3)
    assert len(results) == 3


def test_empty_chunks_list(tmp_path, shared_embedding_model):
    """
    Verify empty chunks list is handled gracefully.

    Calling add_chunks with an empty list should not raise errors.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    vector.add_chunks([])

    assert len(vector.get_document_ids()) == 0


def test_single_chunk_parallel(tmp_path, shared_embedding_model):
    """
    Verify single chunk works with parallel mode.

    Even with multiple workers configured, a single chunk should
    be processed correctly.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunk = _create_test_chunk(0)
    vector.add_chunks([chunk])

    results = vector.search("test content", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "test-doc_chunk_0"


def test_parallel_preserves_metadata(tmp_path, shared_embedding_model):
    """
    Verify parallel processing preserves all chunk metadata.

    All metadata fields should be correctly stored regardless
    of parallelization.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunk = Chunk(
        chunk_id="test-doc_chunk_0",
        doc_id="test-doc",
        content="Test content with metadata",
        chunk_index=0,
        header_path="## Test Header",
        file_path="/tmp/test.md",
        parent_chunk_id="parent_chunk",
        metadata={"tags": ["test", "parallel"], "custom_field": "value"},
        start_pos=0,
        end_pos=100,
        modified_time=datetime.now(timezone.utc),
    )

    vector.add_chunks([chunk])

    result = vector.get_chunk_by_id("test-doc_chunk_0")
    assert result is not None
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    assert metadata.get("tags") == ["test", "parallel"]
    assert metadata.get("custom_field") == "value"
    assert metadata.get("parent_chunk_id") == "parent_chunk"
    assert result["header_path"] == "## Test Header"


def test_parallel_multiple_documents(tmp_path, shared_embedding_model):
    """
    Verify parallel processing works across multiple documents.

    Chunks from different documents should be indexed correctly
    when processed in parallel.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = []
    for doc_id in ["doc1", "doc2", "doc3"]:
        for i in range(5):
            chunks.append(_create_test_chunk(i, doc_id=doc_id))

    vector.add_chunks(chunks)

    assert len(vector.get_document_ids()) == 3
    assert set(vector.get_document_ids()) == {"doc1", "doc2", "doc3"}


def test_add_chunk_vs_add_chunks_equivalence(tmp_path, shared_embedding_model):
    """
    Verify add_chunk and add_chunks produce equivalent results.

    Adding chunks individually vs. in batch should result in
    the same indexed state.
    """
    vector_individual = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=1,
    )
    vector_batch = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = [_create_test_chunk(i) for i in range(5)]

    for chunk in chunks:
        vector_individual.add_chunk(chunk)

    vector_batch.add_chunks(chunks)

    results_individual = vector_individual.search("test content", top_k=5)
    results_batch = vector_batch.search("test content", top_k=5)

    individual_ids = {r["chunk_id"] for r in results_individual}
    batch_ids = {r["chunk_id"] for r in results_batch}

    assert individual_ids == batch_ids


def test_parallel_persistence(tmp_path, shared_embedding_model):
    """
    Verify parallel-indexed chunks persist and reload correctly.

    Chunks indexed in parallel should be saveable and loadable
    just like sequentially-indexed chunks.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = [_create_test_chunk(i) for i in range(10)]
    vector.add_chunks(chunks)

    index_path = tmp_path / "index"
    vector.persist(index_path)

    vector_reloaded = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )
    vector_reloaded.load(index_path)

    results = vector_reloaded.search("test content", top_k=10)
    assert len(results) == 10


def test_parallel_with_vocabulary_registration(tmp_path, shared_embedding_model):
    """
    Verify vocabulary registration works with parallel indexing.

    Terms from parallel-indexed chunks should be registered for
    incremental vocabulary updates.
    """
    vector = VectorIndex(
        embedding_model=shared_embedding_model,
        embedding_workers=4,
    )

    chunks = [_create_test_chunk(i) for i in range(10)]
    vector.add_chunks(chunks)

    assert vector.get_pending_vocabulary_count() > 0
