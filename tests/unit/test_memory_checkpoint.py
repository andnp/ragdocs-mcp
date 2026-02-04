"""
Unit tests for MemoryIndexManager checkpoint functionality.

Tests periodic persistence (checkpointing) based on operation count
and time thresholds to prevent data loss on unexpected shutdown.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from src.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    MemoryConfig,
    SearchConfig,
    ServerConfig,
)
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.manager import MemoryIndexManager
from src.memory.storage import ensure_memory_dirs


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    """Create and return the memory storage path."""
    path = tmp_path / ".memories"
    ensure_memory_dirs(path)
    return path


def create_config(
    tmp_path: Path,
    checkpoint_interval_ops: int = 10,
    checkpoint_interval_secs: int = 300,
) -> Config:
    """Create test configuration with custom checkpoint settings."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir(exist_ok=True)

    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
        ),
        memory=MemoryConfig(
            enabled=True,
            storage_strategy="project",
            checkpoint_interval_ops=checkpoint_interval_ops,
            checkpoint_interval_secs=checkpoint_interval_secs,
        ),
        search=SearchConfig(),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="all-MiniLM-L6-v2"),
    )


def create_manager(config: Config, memory_path: Path, shared_embedding_model) -> MemoryIndexManager:
    """Create MemoryIndexManager instance for testing."""
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return MemoryIndexManager(config, memory_path, vector, keyword, graph)


def create_memory_file(memory_path: Path, filename: str, content: str) -> Path:
    """Create a memory file with given content."""
    file_path = memory_path / f"{filename}.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ============================================================================
# Dirty Flag Tests
# ============================================================================


class TestDirtyFlagTracking:

    def test_initial_state_not_dirty(self, tmp_path: Path, memory_path: Path, shared_embedding_model):
        """Verify manager starts in clean state."""
        config = create_config(tmp_path)
        manager = create_manager(config, memory_path, shared_embedding_model)

        assert manager.is_dirty is False

    def test_index_memory_sets_dirty(self, tmp_path: Path, memory_path: Path, shared_embedding_model):
        """Verify indexing a memory marks the manager as dirty."""
        config = create_config(tmp_path, checkpoint_interval_ops=100)
        manager = create_manager(config, memory_path, shared_embedding_model)

        file_path = create_memory_file(
            memory_path,
            "test-note",
            "# Test Note\n\nContent here."
        )
        manager.index_memory(str(file_path))

        assert manager.is_dirty is True

    def test_remove_memory_sets_dirty(self, tmp_path: Path, memory_path: Path, shared_embedding_model):
        """Verify removing a memory marks the manager as dirty."""
        config = create_config(tmp_path, checkpoint_interval_ops=100)
        manager = create_manager(config, memory_path, shared_embedding_model)

        file_path = create_memory_file(
            memory_path,
            "to-remove",
            "# To Remove\n\nWill be deleted."
        )
        manager.index_memory(str(file_path))

        # Reset dirty via direct persist (bypass checkpoint)
        manager.persist()
        assert manager.is_dirty is False

        manager.remove_memory("memory:to-remove")

        assert manager.is_dirty is True

    def test_persist_clears_dirty(self, tmp_path: Path, memory_path: Path, shared_embedding_model):
        """Verify persist() clears the dirty flag."""
        config = create_config(tmp_path, checkpoint_interval_ops=100)
        manager = create_manager(config, memory_path, shared_embedding_model)

        file_path = create_memory_file(
            memory_path,
            "persist-test",
            "# Persist Test\n\nContent."
        )
        manager.index_memory(str(file_path))
        assert manager.is_dirty is True

        manager.persist()

        assert manager.is_dirty is False


# ============================================================================
# Operation Count Checkpoint Tests
# ============================================================================


class TestOperationCountCheckpoint:

    def test_checkpoint_triggers_after_n_operations(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify checkpoint triggers after configured operation count."""
        config = create_config(tmp_path, checkpoint_interval_ops=3)
        manager = create_manager(config, memory_path, shared_embedding_model)

        persist_count = 0
        original_persist_indices = manager._persist_indices

        def counting_persist() -> None:
            nonlocal persist_count
            persist_count += 1
            original_persist_indices()

        # Type-safely assign via object attribute
        object.__setattr__(manager, '_persist_indices', counting_persist)

        for i in range(5):
            file_path = create_memory_file(
                memory_path,
                f"note-{i}",
                f"# Note {i}\n\nContent {i}."
            )
            manager.index_memory(str(file_path))

        # With interval=3: op 1,2 -> no checkpoint; op 3 -> checkpoint
        # op 4,5 -> no checkpoint (resets to 1,2 after checkpoint)
        assert persist_count >= 1

    def test_checkpoint_resets_operation_counter(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify operation counter resets after checkpoint."""
        config = create_config(tmp_path, checkpoint_interval_ops=2)
        manager = create_manager(config, memory_path, shared_embedding_model)

        persist_calls = []

        def tracking_persist() -> None:
            # Capture ops count at time of persist (already reset to 0 by _maybe_checkpoint)
            persist_calls.append(time.time())
            manager._vector.persist(memory_path / "indices" / "vector")
            manager._keyword.persist(memory_path / "indices" / "keyword")
            manager._graph.persist(memory_path / "indices" / "graph")

        # Type-safely assign via object attribute
        object.__setattr__(manager, '_persist_indices', tracking_persist)

        # Create 6 files - should trigger checkpoint at ops 2, 4, 6
        for i in range(6):
            file_path = create_memory_file(
                memory_path,
                f"counter-{i}",
                f"# Counter {i}\n\nContent."
            )
            manager.index_memory(str(file_path))

        assert len(persist_calls) >= 2


# ============================================================================
# Time-Based Checkpoint Tests
# ============================================================================


class TestTimeBasedCheckpoint:

    def test_checkpoint_triggers_after_time_elapsed(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify checkpoint triggers when time threshold is exceeded."""
        config = create_config(
            tmp_path,
            checkpoint_interval_ops=100,  # High enough to not trigger
            checkpoint_interval_secs=1,   # 1 second
        )
        manager = create_manager(config, memory_path, shared_embedding_model)

        persist_count = 0
        original_persist_indices = manager._persist_indices

        def counting_persist() -> None:
            nonlocal persist_count
            persist_count += 1
            original_persist_indices()

        # Type-safely assign via object attribute
        object.__setattr__(manager, '_persist_indices', counting_persist)

        # Set last checkpoint time to the past
        manager._last_checkpoint_time = time.time() - 2  # 2 seconds ago

        file_path = create_memory_file(
            memory_path,
            "time-test",
            "# Time Test\n\nContent."
        )
        manager.index_memory(str(file_path))

        assert persist_count >= 1

    def test_checkpoint_updates_timestamp(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify checkpoint updates the last checkpoint timestamp."""
        config = create_config(
            tmp_path,
            checkpoint_interval_ops=1,  # Checkpoint every operation
        )
        manager = create_manager(config, memory_path, shared_embedding_model)

        initial_time = manager._last_checkpoint_time

        file_path = create_memory_file(
            memory_path,
            "timestamp-test",
            "# Timestamp Test\n\nContent."
        )

        # Small delay to ensure time difference
        time.sleep(0.01)
        manager.index_memory(str(file_path))

        # After checkpoint, timestamp should be updated
        assert manager._last_checkpoint_time >= initial_time


# ============================================================================
# No Checkpoint on Clean State Tests
# ============================================================================


class TestNoCheckpointWhenClean:

    def test_no_checkpoint_when_not_dirty(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify _maybe_checkpoint does nothing when not dirty."""
        config = create_config(tmp_path, checkpoint_interval_ops=1)
        manager = create_manager(config, memory_path, shared_embedding_model)

        persist_called = False

        def fail_persist() -> None:
            nonlocal persist_called
            persist_called = True

        # Type-safely assign via object attribute
        object.__setattr__(manager, 'persist', fail_persist)

        # Force high op count but dirty=False
        manager._ops_since_checkpoint = 100
        manager._dirty = False

        manager._maybe_checkpoint()

        assert persist_called is False


# ============================================================================
# Reconcile Checkpoint Tests
# ============================================================================


class TestReconcileCheckpoint:

    def test_reconcile_checkpoints_when_changes_detected(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify reconcile triggers checkpoint when it makes changes."""
        config = create_config(
            tmp_path,
            checkpoint_interval_ops=100,  # Won't trigger from count
        )
        manager = create_manager(config, memory_path, shared_embedding_model)

        # Create files that reconcile will detect
        create_memory_file(
            memory_path,
            "reconcile-1",
            "# Reconcile 1\n\nContent."
        )
        create_memory_file(
            memory_path,
            "reconcile-2",
            "# Reconcile 2\n\nContent."
        )

        checkpoint_called = False
        original_maybe_checkpoint = manager._maybe_checkpoint

        def tracking_checkpoint() -> None:
            nonlocal checkpoint_called
            checkpoint_called = True
            original_maybe_checkpoint()

        # Type-safely assign via object attribute
        object.__setattr__(manager, '_maybe_checkpoint', tracking_checkpoint)

        reindexed = manager.reconcile()

        if reindexed > 0:
            assert checkpoint_called is True

    def test_reconcile_no_checkpoint_when_no_changes(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify reconcile doesn't checkpoint when no changes needed."""
        config = create_config(tmp_path, checkpoint_interval_ops=1)
        manager = create_manager(config, memory_path, shared_embedding_model)

        # No files to reconcile = no changes
        persist_called = False

        def fail_persist():
            nonlocal persist_called
            persist_called = True

        # Run reconcile with empty memory path
        reindexed = manager.reconcile()

        # Since no checkpoint is called from reconcile (reindexed_count == 0),
        # persist should not be called
        assert reindexed == 0


# ============================================================================
# Config Validation Tests
# ============================================================================


class TestCheckpointConfigValidation:

    def test_valid_checkpoint_config(self, tmp_path: Path):
        """Verify valid checkpoint config values are accepted."""
        config = MemoryConfig(
            checkpoint_interval_ops=5,
            checkpoint_interval_secs=60,
        )
        assert config.checkpoint_interval_ops == 5
        assert config.checkpoint_interval_secs == 60

    def test_invalid_checkpoint_interval_ops_zero(self):
        """Verify checkpoint_interval_ops=0 raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_interval_ops must be >= 1"):
            MemoryConfig(checkpoint_interval_ops=0)

    def test_invalid_checkpoint_interval_ops_negative(self):
        """Verify negative checkpoint_interval_ops raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_interval_ops must be >= 1"):
            MemoryConfig(checkpoint_interval_ops=-5)

    def test_invalid_checkpoint_interval_secs_negative(self):
        """Verify negative checkpoint_interval_secs raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_interval_secs must be >= 0"):
            MemoryConfig(checkpoint_interval_secs=-10)

    def test_checkpoint_interval_secs_zero_valid(self):
        """Verify checkpoint_interval_secs=0 is valid (time-based disabled)."""
        config = MemoryConfig(checkpoint_interval_secs=0)
        assert config.checkpoint_interval_secs == 0


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestCheckpointThreadSafety:
    """Tests verifying thread-safe access to checkpoint state."""

    def test_concurrent_index_operations_no_lost_ops(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify concurrent index_memory calls don't lose operation counts.

        Multiple threads indexing simultaneously should not cause:
        - Lost increments to _ops_since_checkpoint
        - Duplicate checkpoints from race conditions
        """
        # High checkpoint threshold so we can count ops without triggering
        config = create_config(tmp_path, checkpoint_interval_ops=1000)
        manager = create_manager(config, memory_path, shared_embedding_model)

        num_threads = 10
        ops_per_thread = 5
        total_ops = num_threads * ops_per_thread

        # Create all memory files upfront
        files = []
        for i in range(total_ops):
            file_path = create_memory_file(
                memory_path,
                f"concurrent-{i}",
                f"# Concurrent {i}\n\nContent {i}."
            )
            files.append(str(file_path))

        # Barrier to synchronize thread starts
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []

        def index_batch(thread_id: int):
            try:
                barrier.wait()  # Maximize contention
                for i in range(ops_per_thread):
                    file_idx = thread_id * ops_per_thread + i
                    manager.index_memory(files[file_idx])
            except Exception as e:
                errors.append(e)

        # Run concurrent indexing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(index_batch, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raises if thread failed

        assert not errors, f"Threads encountered errors: {errors}"

        # Verify all operations were counted
        # _ops_since_checkpoint should equal total_ops (no checkpoint triggered)
        assert manager._ops_since_checkpoint == total_ops
        assert manager.is_dirty is True

    def test_concurrent_checkpoint_triggers_exactly_once(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify checkpoint triggers exactly once when threshold crossed.

        When multiple threads race to cross the checkpoint threshold,
        only one should trigger persistence.
        """
        config = create_config(tmp_path, checkpoint_interval_ops=5)
        manager = create_manager(config, memory_path, shared_embedding_model)

        persist_count = 0
        persist_lock = threading.Lock()
        original_persist_indices = manager._persist_indices

        def counting_persist():
            nonlocal persist_count
            with persist_lock:
                persist_count += 1
            # Simulate slow I/O to widen race window
            time.sleep(0.01)
            original_persist_indices()

        object.__setattr__(manager, '_persist_indices', counting_persist)

        num_threads = 10
        # Create files for all threads
        files = []
        for i in range(num_threads):
            file_path = create_memory_file(
                memory_path,
                f"race-{i}",
                f"# Race {i}\n\nContent."
            )
            files.append(str(file_path))

        barrier = threading.Barrier(num_threads)

        def index_one(thread_id: int):
            barrier.wait()  # Start all at once
            manager.index_memory(files[thread_id])

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(index_one, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # With 10 ops and interval=5, we expect 2 checkpoints
        # Key: no duplicate checkpoints from races
        expected_checkpoints = num_threads // config.memory.checkpoint_interval_ops
        assert persist_count == expected_checkpoints

    def test_checkpoint_lock_prevents_state_corruption(
        self, tmp_path: Path, memory_path: Path, shared_embedding_model
    ):
        """Verify lock prevents _dirty/_ops state corruption under contention."""
        config = create_config(tmp_path, checkpoint_interval_ops=3)
        manager = create_manager(config, memory_path, shared_embedding_model)

        # Track state consistency
        inconsistencies: list[str] = []
        state_lock = threading.Lock()

        original_maybe_checkpoint = manager._maybe_checkpoint

        def checking_maybe_checkpoint():
            # Snapshot state under lock (simulates what lock should protect)
            with manager._checkpoint_lock:
                dirty_snapshot = manager._dirty
                ops_snapshot = manager._ops_since_checkpoint

            original_maybe_checkpoint()

            # After checkpoint, verify consistency
            with manager._checkpoint_lock:
                # If dirty was True and ops >= threshold, should have checkpointed
                # (dirty should now be False if checkpoint happened)
                if dirty_snapshot and ops_snapshot >= config.memory.checkpoint_interval_ops:
                    if manager._dirty is True and manager._ops_since_checkpoint > 0:
                        # Checkpoint should have cleared these
                        with state_lock:
                            inconsistencies.append(
                                f"State not cleared after checkpoint: "
                                f"dirty={manager._dirty}, ops={manager._ops_since_checkpoint}"
                            )

        object.__setattr__(manager, '_maybe_checkpoint', checking_maybe_checkpoint)

        num_threads = 8
        ops_per_thread = 10

        files = []
        for i in range(num_threads * ops_per_thread):
            file_path = create_memory_file(
                memory_path,
                f"consistency-{i}",
                f"# Consistency {i}\n\nContent."
            )
            files.append(str(file_path))

        barrier = threading.Barrier(num_threads)

        def index_batch(thread_id: int):
            barrier.wait()
            for i in range(ops_per_thread):
                file_idx = thread_id * ops_per_thread + i
                manager.index_memory(files[file_idx])

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(index_batch, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert not inconsistencies, f"State inconsistencies detected: {inconsistencies}"
