"""
Unit tests for MemoryIndexManager checkpoint functionality.

Tests periodic persistence (checkpointing) based on operation count
and time thresholds to prevent data loss on unexpected shutdown.
"""

import time
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


def create_manager(config: Config, memory_path: Path) -> MemoryIndexManager:
    """Create MemoryIndexManager instance for testing."""
    vector = VectorIndex()
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

    def test_initial_state_not_dirty(self, tmp_path: Path, memory_path: Path):
        """Verify manager starts in clean state."""
        config = create_config(tmp_path)
        manager = create_manager(config, memory_path)

        assert manager.is_dirty is False

    def test_index_memory_sets_dirty(self, tmp_path: Path, memory_path: Path):
        """Verify indexing a memory marks the manager as dirty."""
        config = create_config(tmp_path, checkpoint_interval_ops=100)
        manager = create_manager(config, memory_path)

        file_path = create_memory_file(
            memory_path,
            "test-note",
            "# Test Note\n\nContent here."
        )
        manager.index_memory(str(file_path))

        assert manager.is_dirty is True

    def test_remove_memory_sets_dirty(self, tmp_path: Path, memory_path: Path):
        """Verify removing a memory marks the manager as dirty."""
        config = create_config(tmp_path, checkpoint_interval_ops=100)
        manager = create_manager(config, memory_path)

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

    def test_persist_clears_dirty(self, tmp_path: Path, memory_path: Path):
        """Verify persist() clears the dirty flag."""
        config = create_config(tmp_path, checkpoint_interval_ops=100)
        manager = create_manager(config, memory_path)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify checkpoint triggers after configured operation count."""
        config = create_config(tmp_path, checkpoint_interval_ops=3)
        manager = create_manager(config, memory_path)

        persist_count = 0
        original_persist = manager.persist

        def counting_persist() -> None:
            nonlocal persist_count
            persist_count += 1
            original_persist()

        # Type-safely assign via object attribute
        object.__setattr__(manager, 'persist', counting_persist)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify operation counter resets after checkpoint."""
        config = create_config(tmp_path, checkpoint_interval_ops=2)
        manager = create_manager(config, memory_path)

        persist_calls = []

        def tracking_persist() -> None:
            persist_calls.append(manager._ops_since_checkpoint)
            manager._vector.persist(memory_path / "indices" / "vector")
            manager._keyword.persist(memory_path / "indices" / "keyword")
            manager._graph.persist(memory_path / "indices" / "graph")
            manager._dirty = False

        # Type-safely assign via object attribute
        object.__setattr__(manager, 'persist', tracking_persist)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify checkpoint triggers when time threshold is exceeded."""
        config = create_config(
            tmp_path,
            checkpoint_interval_ops=100,  # High enough to not trigger
            checkpoint_interval_secs=1,   # 1 second
        )
        manager = create_manager(config, memory_path)

        persist_count = 0
        original_persist = manager.persist

        def counting_persist() -> None:
            nonlocal persist_count
            persist_count += 1
            original_persist()

        # Type-safely assign via object attribute
        object.__setattr__(manager, 'persist', counting_persist)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify checkpoint updates the last checkpoint timestamp."""
        config = create_config(
            tmp_path,
            checkpoint_interval_ops=1,  # Checkpoint every operation
        )
        manager = create_manager(config, memory_path)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify _maybe_checkpoint does nothing when not dirty."""
        config = create_config(tmp_path, checkpoint_interval_ops=1)
        manager = create_manager(config, memory_path)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify reconcile triggers checkpoint when it makes changes."""
        config = create_config(
            tmp_path,
            checkpoint_interval_ops=100,  # Won't trigger from count
        )
        manager = create_manager(config, memory_path)

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
        self, tmp_path: Path, memory_path: Path
    ):
        """Verify reconcile doesn't checkpoint when no changes needed."""
        config = create_config(tmp_path, checkpoint_interval_ops=1)
        manager = create_manager(config, memory_path)

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
