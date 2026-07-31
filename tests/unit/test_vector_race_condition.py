"""
Test for race condition during concurrent add_chunk and persist operations.

Regression test for issue where dictionary changed size during iteration
when persist() was called while add_chunk() was still running (typically
during shutdown).
"""

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from searchkernel.domain import Chunk
from searchkernel.indices.vector import VectorIndex


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



class TestVectorRaceCondition:
    """Test concurrent add_chunk and persist operations."""

    def test_concurrent_add_and_persist(self, tmp_path: Path, shared_embedding_model):
        """
        Test that concurrent add_chunk and persist operations don't cause
        'dictionary changed size during iteration' error.

        This simulates the shutdown scenario where:
        1. Background indexing is adding chunks
        2. Shutdown triggers persist
        3. LlamaIndex serializes internal dictionaries during persist
        """
        vector = VectorIndex(embedding_model=shared_embedding_model)

        # Create test chunks
        now = datetime.now(UTC)
        chunks = [
            _with_hash(Chunk(chunk_id=f"chunk_{i}", record_id=f"doc_{i % 3}", content=f"Test content {i} with some keywords like python async threading", metadata={"tags": ["test"], "links": [], "header_path": f"Section {i}", "start_pos": 0, "end_pos": 100, "file_path": f"test_{i % 3}.md", "modified_time": now, "parent_chunk_id": None}, chunk_index=i))
            for i in range(20)
        ]

        # Add initial chunks to have data to persist
        for chunk in chunks[:5]:
            vector.add_chunk(chunk)

        exception_holder: dict[str, Exception | None] = {"exception": None}
        persist_complete = threading.Event()
        add_complete = threading.Event()

        def add_chunks_concurrently():
            """Add chunks in background thread (simulates background indexing)."""
            try:
                for chunk in chunks[5:]:
                    vector.add_chunk(chunk)
                    # Small delay to increase chance of collision
                    threading.Event().wait(0.001)
            except Exception as e:  # noqa: BLE001 -- concurrency test captures any error for later assertion
                exception_holder["exception"] = e
            finally:
                add_complete.set()

        def persist_concurrently():
            """Persist in another thread (simulates shutdown persist)."""
            try:
                # Wait a bit for add_chunks to start
                threading.Event().wait(0.01)
                vector.persist(tmp_path / "concurrent_test")
            except Exception as e:  # noqa: BLE001 -- concurrency test captures any error for later assertion
                exception_holder["exception"] = e
            finally:
                persist_complete.set()

        # Start both operations concurrently
        add_thread = threading.Thread(target=add_chunks_concurrently)
        persist_thread = threading.Thread(target=persist_concurrently)

        add_thread.start()
        persist_thread.start()

        # Wait for both to complete
        add_thread.join(timeout=10)
        persist_thread.join(timeout=10)

        # Check no exception occurred
        if exception_holder["exception"]:
            raise exception_holder["exception"]

        assert add_complete.is_set(), "add_chunks did not complete"
        assert persist_complete.is_set(), "persist did not complete"

        # Verify index is still functional after concurrent operations
        vector2 = VectorIndex(embedding_model=shared_embedding_model)
        vector2.load(tmp_path / "concurrent_test")

        # Should have all documents indexed
        doc_ids = vector2.get_document_ids()
        assert len(doc_ids) > 0, "No documents found after concurrent operations"

    @pytest.mark.asyncio
    async def test_concurrent_add_and_persist_async(
        self, tmp_path: Path, shared_embedding_model
    ):
        """
        Test concurrent operations using asyncio (more realistic for actual server).
        """
        vector = VectorIndex(embedding_model=shared_embedding_model)

        now = datetime.now(UTC)
        chunks = [
            _with_hash(Chunk(chunk_id=f"chunk_{i}", record_id=f"doc_{i % 5}", content=f"Async test content {i} with keywords python asyncio threading", metadata={"tags": ["async"], "links": [], "header_path": f"Async Section {i}", "start_pos": 0, "end_pos": 100, "file_path": f"async_test_{i % 5}.md", "modified_time": now, "parent_chunk_id": None}, chunk_index=i))
            for i in range(30)
        ]

        # Add initial chunks
        for chunk in chunks[:10]:
            await asyncio.to_thread(vector.add_chunk, chunk)

        async def add_chunks_background():
            """Simulate background indexing."""
            for chunk in chunks[10:]:
                await asyncio.to_thread(vector.add_chunk, chunk)
                await asyncio.sleep(0.001)  # Small delay

        async def persist_during_indexing():
            """Simulate persist during active indexing."""
            await asyncio.sleep(0.02)  # Let some chunks get added
            await asyncio.to_thread(vector.persist, tmp_path / "async_test")

        # Run both concurrently
        await asyncio.gather(
            add_chunks_background(),
            persist_during_indexing(),
        )

        # Verify index is intact
        vector2 = VectorIndex(embedding_model=shared_embedding_model)
        await asyncio.to_thread(vector2.load, tmp_path / "async_test")

        doc_ids = vector2.get_document_ids()
        assert len(doc_ids) > 0, "No documents found after async concurrent operations"

    def test_multiple_persists_during_indexing(
        self, tmp_path: Path, shared_embedding_model
    ):
        """
        Test multiple persist calls during active indexing (stress test).
        """
        vector = VectorIndex(embedding_model=shared_embedding_model)

        now = datetime.now(UTC)
        chunks = [
            _with_hash(Chunk(chunk_id=f"chunk_{i}", record_id=f"doc_{i % 10}", content=f"Stress test content {i}", metadata={"tags": ["stress"], "links": [], "header_path": f"Section {i}", "start_pos": 0, "end_pos": 100, "file_path": f"stress_{i % 10}.md", "modified_time": now, "parent_chunk_id": None}, chunk_index=i))
            for i in range(50)
        ]

        # Add initial chunks so persist has something to work with
        for chunk in chunks[:10]:
            vector.add_chunk(chunk)

        exception_holder: dict[str, Exception | None] = {"exception": None}
        start_signal = threading.Event()

        def add_chunks_continuously():
            """Continuously add chunks."""
            try:
                start_signal.wait()  # Wait for signal to start
                for chunk in chunks[10:]:
                    vector.add_chunk(chunk)
                    threading.Event().wait(0.002)
            except Exception as e:  # noqa: BLE001 -- concurrency test captures any error for later assertion
                exception_holder["exception"] = e

        def persist_multiple_times():
            """Persist multiple times during indexing."""
            try:
                start_signal.wait()  # Wait for signal to start
                for i in range(5):
                    threading.Event().wait(0.02)
                    vector.persist(tmp_path / f"stress_test_{i}")
            except Exception as e:  # noqa: BLE001 -- concurrency test captures any error for later assertion
                exception_holder["exception"] = e

        add_thread = threading.Thread(target=add_chunks_continuously)
        persist_thread = threading.Thread(target=persist_multiple_times)

        add_thread.start()
        persist_thread.start()

        # Signal both threads to start simultaneously
        start_signal.set()

        add_thread.join(timeout=15)
        persist_thread.join(timeout=15)

        if exception_holder["exception"]:
            raise exception_holder["exception"]

        # The key assertion is that we didn't crash with "dictionary changed size during iteration"
        # Verify we can still query the index
        doc_ids = vector.get_document_ids()
        assert len(doc_ids) >= 10, f"Expected at least 10 docs, got {len(doc_ids)}"
