"""
Test for race condition during concurrent add_chunk and persist operations.

Regression test for issue where dictionary changed size during iteration
when persist() was called while add_chunk() was still running (typically
during shutdown).
"""

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.indices.vector import VectorIndex
from src.models import Chunk


class TestVectorRaceCondition:
    """Test concurrent add_chunk and persist operations."""

    def test_concurrent_add_and_persist(self, tmp_path: Path):
        """
        Test that concurrent add_chunk and persist operations don't cause
        'dictionary changed size during iteration' error.

        This simulates the shutdown scenario where:
        1. Background indexing is adding chunks
        2. Shutdown triggers persist
        3. LlamaIndex serializes internal dictionaries during persist
        """
        vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")

        # Create test chunks
        now = datetime.now(timezone.utc)
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i % 3}",  # Multiple chunks per doc
                chunk_index=i,
                content=f"Test content {i} with some keywords like python async threading",
                header_path=f"Section {i}",
                file_path=f"test_{i % 3}.md",
                metadata={"tags": ["test"], "links": []},
                parent_chunk_id=None,
                start_pos=0,
                end_pos=100,
                modified_time=now,
            )
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
            except Exception as e:
                exception_holder["exception"] = e
            finally:
                add_complete.set()

        def persist_concurrently():
            """Persist in another thread (simulates shutdown persist)."""
            try:
                # Wait a bit for add_chunks to start
                threading.Event().wait(0.01)
                vector.persist(tmp_path / "concurrent_test")
            except Exception as e:
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
        vector2 = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
        vector2.load(tmp_path / "concurrent_test")

        # Should have all documents indexed
        doc_ids = vector2.get_document_ids()
        assert len(doc_ids) > 0, "No documents found after concurrent operations"

    @pytest.mark.asyncio
    async def test_concurrent_add_and_persist_async(self, tmp_path: Path):
        """
        Test concurrent operations using asyncio (more realistic for actual server).
        """
        vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")

        now = datetime.now(timezone.utc)
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i % 5}",
                chunk_index=i,
                content=f"Async test content {i} with keywords python asyncio threading",
                header_path=f"Async Section {i}",
                file_path=f"async_test_{i % 5}.md",
                metadata={"tags": ["async"], "links": []},
                parent_chunk_id=None,
                start_pos=0,
                end_pos=100,
                modified_time=now,
            )
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
        vector2 = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
        await asyncio.to_thread(vector2.load, tmp_path / "async_test")

        doc_ids = vector2.get_document_ids()
        assert len(doc_ids) > 0, "No documents found after async concurrent operations"

    def test_multiple_persists_during_indexing(self, tmp_path: Path):
        """
        Test multiple persist calls during active indexing (stress test).
        """
        vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")

        now = datetime.now(timezone.utc)
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i % 10}",
                chunk_index=i,
                content=f"Stress test content {i}",
                header_path=f"Section {i}",
                file_path=f"stress_{i % 10}.md",
                metadata={"tags": ["stress"], "links": []},
                parent_chunk_id=None,
                start_pos=0,
                end_pos=100,
                modified_time=now,
            )
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
            except Exception as e:
                exception_holder["exception"] = e

        def persist_multiple_times():
            """Persist multiple times during indexing."""
            try:
                start_signal.wait()  # Wait for signal to start
                for i in range(5):
                    threading.Event().wait(0.02)
                    vector.persist(tmp_path / f"stress_test_{i}")
            except Exception as e:
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
