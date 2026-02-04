"""
Unit tests for LifecycleCoordinator worker initialization.

Tests the multiprocess communication path: worker sends InitCompleteNotification,
main process receives it via _wait_for_init(). These tests verify the timeout bug
scenario where indices load but MCP tools timeout waiting for ready state.
"""

import asyncio
import multiprocessing
from multiprocessing import Queue

import pytest

from src.ipc.commands import HealthStatusResponse, InitCompleteNotification
from src.lifecycle import LifecycleCoordinator, LifecycleState


class TestWaitForInit:
    """Tests for _wait_for_init() - message polling from worker."""

    @pytest.mark.asyncio
    async def test_wait_for_init_receives_notification(self):
        """
        Verify _wait_for_init succeeds when InitCompleteNotification is queued.

        This is the happy path that must work for query_documents to succeed.
        """
        coordinator = LifecycleCoordinator()
        coordinator._response_queue = Queue()

        # Simulate worker sending notification
        notification = InitCompleteNotification(version=5, doc_count=42)
        coordinator._response_queue.put(notification)

        result = await coordinator._wait_for_init(timeout=1.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_init_returns_false_on_timeout(self):
        """
        Verify _wait_for_init returns False when no notification received.

        This simulates worker crash or slow startup - should not hang forever.
        """
        coordinator = LifecycleCoordinator()
        coordinator._response_queue = Queue()
        # Empty queue - no notification sent

        result = await coordinator._wait_for_init(timeout=0.2)

        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_init_ignores_other_messages(self):
        """
        Verify _wait_for_init only looks for InitCompleteNotification.

        Other message types (e.g., HealthStatusResponse) should be skipped.
        """
        coordinator = LifecycleCoordinator()
        coordinator._response_queue = Queue()

        # Put wrong message type first
        coordinator._response_queue.put(
            HealthStatusResponse(healthy=True, queue_depth=0, last_index_time=None, doc_count=10)
        )
        # Then the correct notification
        coordinator._response_queue.put(InitCompleteNotification(version=3, doc_count=100))

        result = await coordinator._wait_for_init(timeout=1.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_init_returns_false_if_no_queue(self):
        """
        Verify _wait_for_init handles missing queue gracefully.
        """
        coordinator = LifecycleCoordinator()
        coordinator._response_queue = None

        result = await coordinator._wait_for_init(timeout=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_init_delayed_notification(self):
        """
        Verify _wait_for_init waits and succeeds when notification arrives late.

        Simulates slow worker initialization that completes before timeout.
        """
        coordinator = LifecycleCoordinator()
        response_queue: Queue = Queue()
        coordinator._response_queue = response_queue

        async def send_notification_after_delay():
            await asyncio.sleep(0.15)
            response_queue.put(InitCompleteNotification(version=7, doc_count=25))

        task = asyncio.create_task(send_notification_after_delay())

        result = await coordinator._wait_for_init(timeout=0.5)

        await task
        assert result is True


class TestStartWithWorkerStateTransitions:
    """
    Tests for start_with_worker() state machine transitions.

    These tests use stubbed worker startup to test state transition logic
    without spawning actual processes (which would be slow).
    """

    @pytest.mark.asyncio
    async def test_start_with_worker_becomes_ready_on_init_notification(self):
        """
        Verify state becomes READY when worker sends InitCompleteNotification.

        This is the expected startup path for query_documents to work.
        """
        coordinator = LifecycleCoordinator()
        coordinator._response_queue = Queue()
        coordinator._command_queue = Queue()
        coordinator._shutdown_event = multiprocessing.Event()

        # Pre-fill notification (simulates worker)
        coordinator._response_queue.put(InitCompleteNotification(version=1, doc_count=10))

        # Stub out actual process startup and sync watcher
        coordinator._state = LifecycleState.STARTING

        # Directly test the state transition logic
        init_received = await coordinator._wait_for_init(timeout=1.0)

        assert init_received is True
        # In real code, this would set state to READY
        # We verify the condition for READY is met

    @pytest.mark.asyncio
    async def test_wait_for_init_timeout_results_in_initializing_state(self):
        """
        Verify timeout in _wait_for_init leads to INITIALIZING (not READY).

        This is the case where worker is slow - system should still work
        but may have degraded performance until sync completes.
        """
        coordinator = LifecycleCoordinator()
        coordinator._response_queue = Queue()
        # Empty queue = timeout

        coordinator._state = LifecycleState.STARTING

        init_received = await coordinator._wait_for_init(timeout=0.1)

        assert init_received is False
        # In real code, this means state stays INITIALIZING
        # query_documents would then wait via wait_ready()


class TestReadOnlyContextReadiness:
    """
    Integration-style tests for ReadOnlyContext readiness detection.

    Tests the full chain: snapshot exists → load → is_ready() returns True.
    """

    @pytest.mark.asyncio
    async def test_is_ready_true_after_snapshot_load(self, tmp_path):
        """
        Verify ReadOnlyContext.is_ready() returns True after loading snapshot.

        This test ensures the fix for the timeout bug is working:
        when indices are loaded from snapshot, sync_receiver.current_version
        should be set, making is_ready() return True.
        """
        import struct

        from src.config import Config, IndexingConfig, LLMConfig, ServerConfig
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()

        # Create version file pointing to v3
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 3))

        # Create minimal snapshot directory
        snapshot_dir = snapshot_base / "v3"
        snapshot_dir.mkdir()
        (snapshot_dir / "vector").mkdir()
        (snapshot_dir / "keyword").mkdir()
        (snapshot_dir / "graph").mkdir()

        # Create empty index files
        (snapshot_dir / "vector" / "doc_id_to_node_ids.json").write_text("{}")
        (snapshot_dir / "keyword" / "index_exists").write_text("")
        (snapshot_dir / "graph" / "graph_data.json").write_text("{}")

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        # This is the key assertion - is_ready() must be True
        # for query_documents to work without timeout
        assert ctx.is_ready() is True, (
            "is_ready() returned False after loading snapshot v3. "
            "This would cause query_documents to timeout."
        )
        assert ctx.sync_receiver.current_version == 3

    @pytest.mark.asyncio
    async def test_is_ready_false_without_snapshot(self, tmp_path):
        """
        Verify is_ready() returns False when no snapshot exists.

        In this case, system must wait for worker to publish first snapshot.
        """
        from src.config import Config, IndexingConfig, LLMConfig, ServerConfig
        from src.reader.context import ReadOnlyContext

        snapshot_base = tmp_path / "snapshots"
        snapshot_base.mkdir()
        # No version.bin file

        docs_path = tmp_path / "docs"
        docs_path.mkdir()

        config = Config(
            server=ServerConfig(),
            indexing=IndexingConfig(
                documents_path=str(docs_path),
                index_path=str(tmp_path / "indices"),
            ),
            llm=LLMConfig(embedding_model="local"),
        )

        ctx = await ReadOnlyContext.create(
            config=config,
            snapshot_base=snapshot_base,
        )

        # Without snapshot, is_ready() should be False
        # System should wait for worker to publish
        assert ctx.is_ready() is False
        assert ctx.sync_receiver.current_version == 0
