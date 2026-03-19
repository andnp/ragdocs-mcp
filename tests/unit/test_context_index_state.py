"""
Unit tests for ApplicationContext IndexState tracking and retry behavior.

Tests cover:
- IndexState dataclass
- Partial failure handling in _background_index()
- Retry behavior with exponential backoff
- State transitions (uninitialized → indexing → ready/partial/failed)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.context import ApplicationContext, IndexState
from src.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    compute_bootstrap_generation,
    load_bootstrap_checkpoint,
    save_bootstrap_checkpoint,
)
from src.indexing.manifest import (
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    load_manifest,
    save_manifest,
)
from src.indexing.tasks import TaskBatchSubmissionResult


def _setattr(obj: Any, name: str, value: Any):
    """Helper to set attributes bypassing type checking."""
    object.__setattr__(obj, name, value)


class TestIndexState:
    """Tests for IndexState dataclass."""

    def test_index_state_defaults(self):
        """Verify IndexState defaults to zero counts and no error."""
        state = IndexState(status="uninitialized")

        assert state.status == "uninitialized"
        assert state.indexed_count == 0
        assert state.total_count == 0
        assert state.last_error is None

    def test_index_state_with_values(self):
        """Verify IndexState stores provided values."""
        state = IndexState(
            status="partial",
            indexed_count=5,
            total_count=10,
            last_error="Connection failed",
        )

        assert state.status == "partial"
        assert state.indexed_count == 5
        assert state.total_count == 10
        assert state.last_error == "Connection failed"

    def test_index_state_status_values(self):
        """Verify all valid status values are accepted."""
        for status in ("uninitialized", "indexing", "partial", "ready", "failed"):
            state = IndexState(status=status)  # pyright: ignore[reportArgumentType] -- Testing literal values
            assert state.status == status


@dataclass
class MockConfig:
    """Minimal mock config for testing."""

    indexing: MagicMock = field(default_factory=MagicMock)
    llm: MagicMock = field(default_factory=MagicMock)
    chunking: MagicMock = field(default_factory=MagicMock)

    def __post_init__(self):
        self.indexing.reconciliation_interval_seconds = 0


@dataclass
class MockIndexManager:
    """Mock index manager tracking calls."""

    index_calls: list[str] = field(default_factory=list)
    persist_called: bool = False
    fail_on_file: str | None = None
    _ready: bool = True

    def index_document(self, path: str):
        if self.fail_on_file and path == self.fail_on_file:
            raise RuntimeError(f"Failed to index {path}")
        self.index_calls.append(path)

    def persist(self):
        self.persist_called = True

    def is_ready(self) -> bool:
        return self._ready


class SlowLoadIndexManager:
    def __init__(self, delay_seconds: float = 0.2):
        self.delay_seconds = delay_seconds
        self.loaded = False
        self.vector = type(
            "VectorStub",
            (),
            {
                "_concept_vocabulary": {"warm": 1},
                "model_ready": lambda self: False,
                "warm_up": lambda self: None,
            },
        )()

    def load(self):
        time.sleep(self.delay_seconds)
        self.loaded = True

    def is_ready(self) -> bool:
        return self.loaded


class ExistingIndexManager:
    def __init__(self):
        self.loaded = False
        self.load_calls = 0
        self.vector = type(
            "VectorStub",
            (),
            {
                "_concept_vocabulary": {"warm": 1},
                "model_ready": lambda self: False,
                "warm_up": lambda self: None,
            },
        )()

    def load(self):
        self.load_calls += 1
        self.loaded = True

    def get_document_count(self) -> int:
        return 2

    def is_ready(self) -> bool:
        return self.loaded


class WarmupVectorStub:
    def __init__(self, *, ready: bool = False):
        self._ready = ready
        self.warm_up_calls = 0
        self._concept_vocabulary = {"warm": 1}

    def model_ready(self) -> bool:
        return self._ready

    def warm_up(self) -> None:
        self.warm_up_calls += 1
        self._ready = True


class WarmupIndexManager:
    def __init__(self, *, ready: bool = True, model_ready: bool = False):
        self._ready = ready
        self.vector = WarmupVectorStub(ready=model_ready)

    def is_ready(self) -> bool:
        return self._ready

    def get_document_count(self) -> int:
        return 0


class TestBackgroundIndexRetry:
    """Tests for _background_index() retry behavior."""

    @pytest.fixture
    def mock_context(self, tmp_path: Path) -> Any:
        """Create a minimal context for testing _background_index."""
        ctx = object.__new__(ApplicationContext)
        mock_config = MockConfig()
        mock_config.indexing.documents_path = str(tmp_path)
        _setattr(ctx, "config", mock_config)
        _setattr(ctx, "index_manager", MockIndexManager())
        _setattr(ctx, "index_path", tmp_path / ".index")
        ctx.index_path.mkdir()
        _setattr(ctx, "current_manifest", None)
        _setattr(ctx, "_ready_event", asyncio.Event())
        _setattr(ctx, "_init_error", None)
        _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
        _setattr(ctx, "_is_virgin_startup", False)
        return ctx

    @pytest.mark.asyncio
    async def test_successful_indexing_sets_ready_state(
        self, mock_context: Any, tmp_path: Path
    ):
        """Verify successful indexing sets status to 'ready'."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")

        # Mock discover_files to return our test files
        files = [str(tmp_path / "doc1.md"), str(tmp_path / "doc2.md")]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))

        await mock_context._background_index()

        assert mock_context._index_state.status == "ready"
        assert mock_context._index_state.indexed_count == 2
        assert mock_context._index_state.total_count == 2
        assert mock_context._index_state.last_error is None
        assert mock_context._ready_event.is_set()
        assert mock_context._init_error is None

    @pytest.mark.asyncio
    async def test_partial_failure_sets_partial_state(
        self, mock_context: Any, tmp_path: Path
    ):
        """Verify partial failure sets status to 'partial' with correct counts."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")
        (tmp_path / "doc3.md").write_text("# Doc 3")

        files = [
            str(tmp_path / "doc1.md"),
            str(tmp_path / "doc2.md"),
            str(tmp_path / "doc3.md"),
        ]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))

        # Fail on second file
        mock_context.index_manager.fail_on_file = str(tmp_path / "doc2.md")

        await mock_context._background_index()

        # After exhausting retries, should be partial (1 file indexed before failure)
        assert mock_context._index_state.status == "partial"
        assert mock_context._index_state.indexed_count == 1
        assert mock_context._index_state.total_count == 3
        assert mock_context._index_state.last_error is not None
        assert "doc2.md" in mock_context._index_state.last_error
        assert mock_context._ready_event.is_set()
        assert mock_context._init_error is not None

    @pytest.mark.asyncio
    async def test_complete_failure_sets_failed_state(
        self, mock_context: Any, tmp_path: Path
    ):
        """Verify failure on first file sets status to 'failed'."""
        (tmp_path / "doc1.md").write_text("# Doc 1")

        files = [str(tmp_path / "doc1.md")]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))

        # Fail on first file
        mock_context.index_manager.fail_on_file = str(tmp_path / "doc1.md")

        await mock_context._background_index()

        assert mock_context._index_state.status == "failed"
        assert mock_context._index_state.indexed_count == 0
        assert mock_context._index_state.total_count == 1
        assert mock_context._index_state.last_error is not None
        assert mock_context._init_error is not None

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(
        self, mock_context: Any, tmp_path: Path
    ):
        """Verify retries use exponential backoff timing."""
        (tmp_path / "doc1.md").write_text("# Doc 1")

        files = [str(tmp_path / "doc1.md")]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))
        mock_context.index_manager.fail_on_file = str(tmp_path / "doc1.md")

        sleep_delays: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float):
            sleep_delays.append(delay)
            await original_sleep(0)  # Don't actually wait

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await mock_context._background_index()

        # Should have 2 retry delays (3 attempts total, 2 sleeps between)
        assert len(sleep_delays) == 2
        # Exponential backoff: 1.0, 2.0
        assert sleep_delays[0] == pytest.approx(1.0)
        assert sleep_delays[1] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(
        self, mock_context: Any, tmp_path: Path
    ):
        """Verify successful retry results in 'ready' state."""
        (tmp_path / "doc1.md").write_text("# Doc 1")

        files = [str(tmp_path / "doc1.md")]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))

        # Fail once, then succeed
        attempt_count = 0

        def conditional_fail(path: str):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise RuntimeError("Transient failure")
            mock_context.index_manager.index_calls.append(path)

        mock_context.index_manager.index_document = conditional_fail

        with patch("asyncio.sleep", return_value=None):
            await mock_context._background_index()

        assert mock_context._index_state.status == "ready"
        assert mock_context._index_state.indexed_count == 1
        assert mock_context._init_error is None


@pytest.mark.asyncio
async def test_ensure_fresh_indices_reloads_when_store_version_advances(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path)
    _setattr(ctx, "_ready_event", asyncio.Event())
    ctx._ready_event.set()
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_loaded_index_state_version", 1.0)
    _setattr(ctx, "_index_state", IndexState(status="ready"))
    _setattr(ctx, "_is_virgin_startup", False)

    class _Manager:
        def __init__(self):
            self.load_calls = 0

        def load(self):
            self.load_calls += 1

        def get_document_count(self) -> int:
            return 0

    manager = _Manager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 2.0)

    await ctx.ensure_fresh_indices()

    assert manager.load_calls == 1
    assert ctx._loaded_index_state_version == 2.0


@pytest.mark.asyncio
async def test_ensure_fresh_indices_keeps_serving_loaded_state_on_lock_timeout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path)
    _setattr(ctx, "_ready_event", asyncio.Event())
    ctx._ready_event.set()
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_loaded_index_state_version", 1.0)
    _setattr(
        ctx,
        "_index_state",
        IndexState(status="partial", indexed_count=3, total_count=5),
    )
    _setattr(ctx, "_is_virgin_startup", False)

    class _Manager:
        def __init__(self):
            self.load_calls = 0

        def load(self):
            self.load_calls += 1
            raise TimeoutError("Failed to acquire shared lock after 5.0s")

        def get_document_count(self) -> int:
            return 99

    manager = _Manager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 2.0)

    with caplog.at_level(logging.WARNING):
        await ctx.ensure_fresh_indices()

    assert manager.load_calls == 1
    assert ctx._loaded_index_state_version == 1.0
    assert ctx._index_state == IndexState(
        status="partial",
        indexed_count=3,
        total_count=5,
    )
    assert (
        "Freshness reload timed out acquiring shared index lock" in caplog.text
    )


@pytest.mark.asyncio
async def test_schedule_freshness_refresh_runs_reload_in_background(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path)
    _setattr(ctx, "_ready_event", asyncio.Event())
    ctx._ready_event.set()
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_freshness_task", None)
    _setattr(ctx, "_loaded_index_state_version", 1.0)
    _setattr(ctx, "_index_state", IndexState(status="ready"))
    _setattr(ctx, "_is_virgin_startup", False)

    class _Manager:
        def __init__(self):
            self.load_calls = 0

        def load(self):
            self.load_calls += 1

        def get_document_count(self) -> int:
            return 0

    manager = _Manager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 2.0)

    scheduled = ctx.schedule_freshness_refresh()

    assert scheduled is True
    assert ctx._freshness_task is not None

    task = ctx._freshness_task
    await asyncio.wait_for(task, timeout=1.0)

    assert manager.load_calls == 1
    assert ctx._loaded_index_state_version == 2.0
    assert ctx._freshness_task is None


@pytest.mark.asyncio
async def test_schedule_freshness_refresh_deduplicates_in_flight_task(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path)
    _setattr(ctx, "_ready_event", asyncio.Event())
    ctx._ready_event.set()
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_freshness_task", None)
    _setattr(ctx, "_loaded_index_state_version", 1.0)
    _setattr(ctx, "_index_state", IndexState(status="ready"))
    _setattr(ctx, "_is_virgin_startup", False)

    class _Manager:
        def get_document_count(self) -> int:
            return 0

    manager = _Manager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 2.0)

    refresh_started = asyncio.Event()
    allow_refresh_to_finish = asyncio.Event()

    async def fake_run_freshness_refresh():
        refresh_started.set()
        await allow_refresh_to_finish.wait()

    _setattr(ctx, "_run_freshness_refresh", fake_run_freshness_refresh)

    first_scheduled = ctx.schedule_freshness_refresh()
    await asyncio.wait_for(refresh_started.wait(), timeout=1.0)

    second_scheduled = ctx.schedule_freshness_refresh()

    assert first_scheduled is True
    assert second_scheduled is False
    assert ctx._freshness_task is not None

    task = ctx._freshness_task
    allow_refresh_to_finish.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert ctx._freshness_task is None


@pytest.mark.asyncio
async def test_schedule_embedding_model_warmup_runs_in_background(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path)
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_embedding_warmup_task", None)
    _setattr(ctx, "index_manager", WarmupIndexManager())

    scheduled = ctx.schedule_embedding_model_warmup()

    assert scheduled is True
    assert ctx._embedding_warmup_task is not None

    task = ctx._embedding_warmup_task
    await asyncio.wait_for(task, timeout=1.0)

    assert ctx.index_manager.vector.warm_up_calls == 1
    assert ctx.index_manager.vector.model_ready() is True
    assert ctx._embedding_warmup_task is None


@pytest.mark.asyncio
async def test_schedule_embedding_model_warmup_deduplicates_in_flight_task(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path)
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_embedding_warmup_task", None)
    _setattr(ctx, "index_manager", WarmupIndexManager())

    warmup_started = asyncio.Event()
    allow_warmup_to_finish = asyncio.Event()

    async def fake_run_embedding_model_warmup() -> None:
        warmup_started.set()
        await allow_warmup_to_finish.wait()

    _setattr(ctx, "_run_embedding_model_warmup", fake_run_embedding_model_warmup)

    first_scheduled = ctx.schedule_embedding_model_warmup()
    await asyncio.wait_for(warmup_started.wait(), timeout=1.0)

    second_scheduled = ctx.schedule_embedding_model_warmup()

    assert first_scheduled is True
    assert second_scheduled is False
    assert ctx._embedding_warmup_task is not None

    task = ctx._embedding_warmup_task
    allow_warmup_to_finish.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert ctx._embedding_warmup_task is None


def test_schedule_embedding_model_warmup_skips_ready_or_unloaded_states(
    tmp_path: Path,
):
    ready_ctx = object.__new__(ApplicationContext)
    _setattr(ready_ctx, "index_path", tmp_path)
    _setattr(ready_ctx, "_init_error", None)
    _setattr(ready_ctx, "_embedding_warmup_task", None)
    _setattr(
        ready_ctx,
        "index_manager",
        WarmupIndexManager(ready=True, model_ready=True),
    )

    not_ready_ctx = object.__new__(ApplicationContext)
    _setattr(not_ready_ctx, "index_path", tmp_path)
    _setattr(not_ready_ctx, "_init_error", None)
    _setattr(not_ready_ctx, "_embedding_warmup_task", None)
    _setattr(
        not_ready_ctx,
        "index_manager",
        WarmupIndexManager(ready=False, model_ready=False),
    )

    assert ready_ctx.schedule_embedding_model_warmup() is False
    assert not_ready_ctx.schedule_embedding_model_warmup() is False


@pytest.mark.asyncio
async def test_start_uses_task_bootstrap_for_virgin_background_start(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "index_manager", MockIndexManager())
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=True))

    bootstrap_calls: list[str] = []

    async def fake_bootstrap_via_tasks() -> None:
        bootstrap_calls.append("called")

    async def fail_if_called() -> None:
        raise AssertionError("main-process _background_index should not run")

    _setattr(ctx, "_bootstrap_via_tasks", fake_bootstrap_via_tasks)
    _setattr(ctx, "_background_index", fail_if_called)

    await ctx.start(background_index=True)
    await asyncio.wait_for(ctx._background_index_task, timeout=1.0)

    assert bootstrap_calls == ["called"]


@pytest.mark.asyncio
async def test_start_uses_task_bootstrap_when_checkpoint_resume_is_pending(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "index_manager", MockIndexManager())
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=False))

    save_bootstrap_checkpoint(
        ctx.index_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation="resume-me",
            complete=False,
            targets={
                "doc.md": BootstrapFileStamp("doc.md", mtime_ns=1, size=1),
            },
            completed={},
        ),
    )

    bootstrap_calls: list[str] = []

    async def fake_bootstrap_via_tasks() -> None:
        bootstrap_calls.append("called")

    async def fail_if_called() -> None:
        raise AssertionError("main-process _background_index should not run")

    _setattr(ctx, "_bootstrap_via_tasks", fake_bootstrap_via_tasks)
    _setattr(ctx, "_background_index", fail_if_called)

    await ctx.start(background_index=True)
    await asyncio.wait_for(ctx._background_index_task, timeout=1.0)

    assert bootstrap_calls == ["called"]


@pytest.mark.asyncio
async def test_start_resume_bootstrap_loads_partial_indices_before_background_task(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "documents_roots", [tmp_path])
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "_background_index_task", None)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_loaded_index_state_version", 0.0)
    _setattr(ctx, "_is_virgin_startup", False)
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=False))

    file_path = tmp_path / "doc.md"
    file_path.write_text("# Doc")
    later_file_path = tmp_path / "later.md"
    later_file_path.write_text("# Later")
    file_stat = file_path.stat()
    later_file_stat = later_file_path.stat()
    _setattr(
        ctx,
        "discover_files",
        MagicMock(return_value=[str(file_path), str(later_file_path)]),
    )

    save_manifest(
        ctx.index_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={},
            indexed_files={"doc": "doc.md"},
        ),
    )
    save_bootstrap_checkpoint(
        ctx.index_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation="resume-me",
            complete=False,
            targets={
                "doc.md": BootstrapFileStamp(
                    "doc.md",
                    mtime_ns=file_stat.st_mtime_ns,
                    size=file_stat.st_size,
                ),
                    "later.md": BootstrapFileStamp(
                        "later.md",
                        mtime_ns=later_file_stat.st_mtime_ns,
                        size=later_file_stat.st_size,
                    ),
            },
            completed={
                "doc.md": BootstrapFileStamp(
                    "doc.md",
                    mtime_ns=file_stat.st_mtime_ns,
                    size=file_stat.st_size,
                )
            },
        ),
    )

    class ResumeIndexManager:
        def __init__(self) -> None:
            self.load_calls = 0
            self.vector = type(
                "VectorStub",
                (),
                {
                    "_concept_vocabulary": {"warm": 1},
                    "model_ready": lambda self: False,
                    "warm_up": lambda self: None,
                },
            )()

        def load(self) -> None:
            self.load_calls += 1

        def get_document_count(self) -> int:
            return 1

        def is_ready(self) -> bool:
            return self.load_calls > 0

    manager = ResumeIndexManager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 1.0)

    bootstrap_started = asyncio.Event()
    allow_bootstrap_to_finish = asyncio.Event()
    scheduled_warmups: list[str] = []

    async def fake_bootstrap_via_tasks() -> None:
        bootstrap_started.set()
        await allow_bootstrap_to_finish.wait()

    def fake_schedule_embedding_model_warmup() -> bool:
        scheduled_warmups.append("called")
        return True

    _setattr(ctx, "_bootstrap_via_tasks", fake_bootstrap_via_tasks)
    _setattr(
        ctx,
        "schedule_embedding_model_warmup",
        fake_schedule_embedding_model_warmup,
    )

    await asyncio.wait_for(ctx.start(background_index=True), timeout=1.0)
    await asyncio.wait_for(bootstrap_started.wait(), timeout=1.0)

    assert manager.load_calls == 1
    assert ctx._ready_event.is_set()
    assert ctx._index_state.status == "partial"
    assert ctx._index_state.indexed_count == 1
    assert ctx._index_state.total_count == 2
    assert scheduled_warmups == ["called"]
    assert ctx._background_index_task is not None
    assert ctx._background_index_task.done() is False

    allow_bootstrap_to_finish.set()
    await asyncio.wait_for(ctx._background_index_task, timeout=1.0)


@pytest.mark.asyncio
async def test_start_rebuild_preloads_partial_indices_before_background_task(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "documents_roots", [tmp_path])
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "_background_index_task", None)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_loaded_index_state_version", 0.0)
    _setattr(ctx, "_is_virgin_startup", False)
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=True))

    file_one = tmp_path / "doc1.md"
    file_two = tmp_path / "doc2.md"
    file_one.write_text("# Doc 1")
    file_two.write_text("# Doc 2")
    files = [str(file_one), str(file_two)]
    _setattr(ctx, "discover_files", MagicMock(return_value=files))

    save_manifest(
        ctx.index_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={},
            indexed_files={"doc1": "doc1.md"},
        ),
    )

    class RebuildIndexManager:
        def __init__(self) -> None:
            self.load_calls = 0
            self.vector = type(
                "VectorStub",
                (),
                {
                    "_concept_vocabulary": {"warm": 1},
                    "model_ready": lambda self: False,
                    "warm_up": lambda self: None,
                },
            )()

        def load(self) -> None:
            self.load_calls += 1

        def get_document_count(self) -> int:
            return 1

        def is_ready(self) -> bool:
            return self.load_calls > 0

    manager = RebuildIndexManager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 1.0)

    bootstrap_started = asyncio.Event()
    allow_bootstrap_to_finish = asyncio.Event()
    scheduled_warmups: list[str] = []

    async def fake_bootstrap_via_tasks() -> None:
        bootstrap_started.set()
        await allow_bootstrap_to_finish.wait()

    def fake_schedule_embedding_model_warmup() -> bool:
        scheduled_warmups.append("called")
        return True

    _setattr(ctx, "_bootstrap_via_tasks", fake_bootstrap_via_tasks)
    _setattr(
        ctx,
        "schedule_embedding_model_warmup",
        fake_schedule_embedding_model_warmup,
    )

    await asyncio.wait_for(ctx.start(background_index=True), timeout=1.0)
    await asyncio.wait_for(bootstrap_started.wait(), timeout=1.0)

    assert manager.load_calls == 1
    assert ctx._ready_event.is_set()
    assert ctx._index_state.status == "partial"
    assert ctx._index_state.indexed_count == 1
    assert ctx._index_state.total_count == 2
    assert scheduled_warmups == ["called"]
    assert ctx._background_index_task is not None
    assert ctx._background_index_task.done() is False

    allow_bootstrap_to_finish.set()
    await asyncio.wait_for(ctx._background_index_task, timeout=1.0)


@pytest.mark.asyncio
async def test_preload_existing_indices_for_background_bootstrap_marks_complete_rebuild_as_indexing(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "documents_roots", [tmp_path])
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_loaded_index_state_version", 0.0)
    _setattr(ctx, "_is_virgin_startup", False)

    file_path = tmp_path / "doc.md"
    file_path.write_text("# Doc")
    _setattr(ctx, "discover_files", MagicMock(return_value=[str(file_path)]))

    class PreloadedIndexManager:
        def __init__(self) -> None:
            self.load_calls = 0

        def load(self) -> None:
            self.load_calls += 1

        def get_document_count(self) -> int:
            return 1

        def is_ready(self) -> bool:
            return self.load_calls > 0

    _setattr(ctx, "index_manager", PreloadedIndexManager())
    _setattr(ctx, "_compute_index_state_version", lambda: 1.0)

    save_manifest(
        ctx.index_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={},
            indexed_files={"doc": "doc.md"},
        ),
    )

    scheduled_warmups: list[str] = []

    def fake_schedule_embedding_model_warmup() -> bool:
        scheduled_warmups.append("called")
        return True

    _setattr(
        ctx,
        "schedule_embedding_model_warmup",
        fake_schedule_embedding_model_warmup,
    )

    preloaded = await ctx._preload_existing_indices_for_background_bootstrap(
        rebuild_pending=True,
    )

    assert preloaded is True
    assert ctx._ready_event.is_set()
    assert ctx._index_state == IndexState(
        status="indexing",
        indexed_count=1,
        total_count=1,
    )
    assert scheduled_warmups == ["called"]


@pytest.mark.asyncio
async def test_task_bootstrap_marks_context_ready_from_partial_persisted_state(
    tmp_path: Path,
    monkeypatch,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "documents_roots", [tmp_path])
    _setattr(
        ctx,
        "current_manifest",
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={},
            indexed_files={},
        ),
    )
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_loaded_index_state_version", 0.0)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_is_virgin_startup", True)

    (tmp_path / "doc1.md").write_text("# Doc 1")
    (tmp_path / "doc2.md").write_text("# Doc 2")
    files = [str(tmp_path / "doc1.md"), str(tmp_path / "doc2.md")]
    _setattr(ctx, "discover_files", MagicMock(return_value=files))

    checkpoint_targets = {
        "doc1.md": BootstrapFileStamp(
            "doc1.md",
            mtime_ns=(tmp_path / "doc1.md").stat().st_mtime_ns,
            size=(tmp_path / "doc1.md").stat().st_size,
        ),
        "doc2.md": BootstrapFileStamp(
            "doc2.md",
            mtime_ns=(tmp_path / "doc2.md").stat().st_mtime_ns,
            size=(tmp_path / "doc2.md").stat().st_size,
        ),
    }

    save_manifest(
        ctx.index_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={},
            indexed_files={"doc1": "doc1.md"},
        ),
    )
    save_bootstrap_checkpoint(
        ctx.index_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation=compute_bootstrap_generation(ctx.current_manifest, checkpoint_targets),
            complete=False,
            targets=checkpoint_targets,
            completed={
                "doc1.md": checkpoint_targets["doc1.md"],
            },
        ),
    )

    class BootstrapIndexManager:
        def __init__(self) -> None:
            self.load_calls = 0
            self.allow_full_completion = False
            self.vector = type(
                "VectorStub",
                (),
                {
                    "_concept_vocabulary": {"warm": 1},
                    "model_ready": lambda self: False,
                    "warm_up": lambda self: None,
                },
            )()

        def persist(self) -> None:
            return None

        def load(self) -> None:
            self.load_calls += 1

        def get_document_count(self) -> int:
            return 1 if self.load_calls == 1 else 2

        def is_ready(self) -> bool:
            return self.load_calls >= 1

    manager = BootstrapIndexManager()
    _setattr(ctx, "index_manager", manager)

    def compute_version() -> float:
        if manager.load_calls == 0:
            return 1.0
        if manager.allow_full_completion:
            return 2.0
        return 1.0

    _setattr(ctx, "_compute_index_state_version", compute_version)

    enqueued: list[str] = []

    def fake_submit_index_batch(
        file_paths: list[str], force: bool = False
    ) -> TaskBatchSubmissionResult:
        assert force is False
        enqueued.extend(file_paths)
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(set(file_paths)),
            enqueued_count=len(set(file_paths)),
        )

    monkeypatch.setattr(
        "src.indexing.tasks.submit_index_batch",
        fake_submit_index_batch,
    )

    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    bootstrap_task = asyncio.create_task(ctx._bootstrap_via_tasks())

    await asyncio.wait_for(ctx._ready_event.wait(), timeout=1.0)

    assert enqueued == [str(tmp_path / "doc2.md")]
    assert ctx._index_state.status == "partial"
    assert ctx._index_state.indexed_count == 1
    assert ctx._index_state.total_count == 2
    assert ctx.is_ready() is True

    bootstrap_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bootstrap_task


@pytest.mark.asyncio
async def test_task_bootstrap_keeps_monitoring_when_remaining_work_is_already_pending(
    tmp_path: Path,
    monkeypatch,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "documents_roots", [tmp_path])
    _setattr(
        ctx,
        "current_manifest",
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={},
            indexed_files={},
        ),
    )
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_loaded_index_state_version", 0.0)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_is_virgin_startup", True)

    (tmp_path / "doc1.md").write_text("# Doc 1")
    (tmp_path / "doc2.md").write_text("# Doc 2")
    files = [str(tmp_path / "doc1.md"), str(tmp_path / "doc2.md")]
    _setattr(ctx, "discover_files", MagicMock(return_value=files))

    class PendingOnlyIndexManager:
        def __init__(self) -> None:
            self.vector = type(
                "VectorStub",
                (),
                {
                    "_concept_vocabulary": {"warm": 1},
                    "model_ready": lambda self: False,
                    "warm_up": lambda self: None,
                },
            )()

        def persist(self) -> None:
            return None

        def load(self) -> None:
            return None

        def get_document_count(self) -> int:
            return 0

        def is_ready(self) -> bool:
            return False

    _setattr(ctx, "index_manager", PendingOnlyIndexManager())
    _setattr(ctx, "_compute_index_state_version", lambda: 0.0)

    enqueue_checked = asyncio.Event()

    monkeypatch.setattr(
        "src.indexing.tasks.submit_index_batch",
        lambda file_paths, force=False: enqueue_checked.set()
        or TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(set(file_paths)),
            enqueued_count=0,
            already_pending_count=len(set(file_paths)),
        ),
    )

    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    bootstrap_task = asyncio.create_task(ctx._bootstrap_via_tasks())
    await asyncio.wait_for(enqueue_checked.wait(), timeout=1.0)

    assert ctx._index_state.status == "indexing"
    assert ctx._index_state.last_error is None
    assert ctx._init_error is None
    assert ctx._ready_event.is_set() is False

    bootstrap_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bootstrap_task


@pytest.mark.asyncio
async def test_task_bootstrap_skips_durably_completed_files(
    tmp_path: Path,
    monkeypatch,
):
    file_one = tmp_path / "doc1.md"
    file_two = tmp_path / "doc2.md"
    file_one.write_text("# Doc 1")
    file_two.write_text("# Doc 2")

    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "documents_roots", [tmp_path])
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    manifest = IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model="local",
        chunking_config={},
        indexed_files={"doc1": "doc1.md"},
    )
    _setattr(ctx, "current_manifest", manifest)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_freshness_lock", asyncio.Lock())
    _setattr(ctx, "_loaded_index_state_version", 0.0)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_is_virgin_startup", False)
    _setattr(
        ctx,
        "discover_files",
        MagicMock(return_value=[str(file_one), str(file_two)]),
    )

    save_manifest(ctx.index_path, manifest)
    doc_one_stat = file_one.stat()
    doc_two_stat = file_two.stat()
    save_bootstrap_checkpoint(
        ctx.index_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation=compute_bootstrap_generation(
                manifest,
                {
                    "doc1.md": BootstrapFileStamp(
                        "doc1.md",
                        mtime_ns=doc_one_stat.st_mtime_ns,
                        size=doc_one_stat.st_size,
                    ),
                    "doc2.md": BootstrapFileStamp(
                        "doc2.md",
                        mtime_ns=doc_two_stat.st_mtime_ns,
                        size=doc_two_stat.st_size,
                    ),
                },
            ),
            complete=False,
            targets={
                "doc1.md": BootstrapFileStamp(
                    "doc1.md",
                    mtime_ns=doc_one_stat.st_mtime_ns,
                    size=doc_one_stat.st_size,
                ),
                "doc2.md": BootstrapFileStamp(
                    "doc2.md",
                    mtime_ns=doc_two_stat.st_mtime_ns,
                    size=doc_two_stat.st_size,
                ),
            },
            completed={
                "doc1.md": BootstrapFileStamp(
                    "doc1.md",
                    mtime_ns=doc_one_stat.st_mtime_ns,
                    size=doc_one_stat.st_size,
                ),
            },
        ),
    )

    class BootstrapIndexManager:
        def __init__(self) -> None:
            self.load_calls = 0
            self.vector = type(
                "VectorStub",
                (),
                {
                    "_concept_vocabulary": {"warm": 1},
                    "model_ready": lambda self: False,
                    "warm_up": lambda self: None,
                },
            )()

        def load(self) -> None:
            self.load_calls += 1

        def persist(self) -> None:
            return None

        def get_document_count(self) -> int:
            return 2

        def is_ready(self) -> bool:
            return self.load_calls >= 1

    manager = BootstrapIndexManager()
    _setattr(ctx, "index_manager", manager)
    _setattr(ctx, "_compute_index_state_version", lambda: 1.0)

    enqueued: list[str] = []

    def fake_submit_index_batch(
        file_paths: list[str], force: bool = False
    ) -> TaskBatchSubmissionResult:
        enqueued.extend(file_paths)
        updated_manifest = load_manifest(ctx.index_path)
        assert updated_manifest is not None
        updated_manifest.indexed_files = {
            "doc1": "doc1.md",
            "doc2": "doc2.md",
        }
        save_manifest(ctx.index_path, updated_manifest)
        doc_two_current = file_two.stat()
        save_bootstrap_checkpoint(
            ctx.index_path,
            BootstrapCheckpoint(
                schema_version="1.0.0",
                generation=load_bootstrap_checkpoint(ctx.index_path).generation,
                complete=True,
                targets={
                    "doc1.md": BootstrapFileStamp(
                        "doc1.md",
                        mtime_ns=doc_one_stat.st_mtime_ns,
                        size=doc_one_stat.st_size,
                    ),
                    "doc2.md": BootstrapFileStamp(
                        "doc2.md",
                        mtime_ns=doc_two_current.st_mtime_ns,
                        size=doc_two_current.st_size,
                    ),
                },
                completed={
                    "doc1.md": BootstrapFileStamp(
                        "doc1.md",
                        mtime_ns=doc_one_stat.st_mtime_ns,
                        size=doc_one_stat.st_size,
                    ),
                    "doc2.md": BootstrapFileStamp(
                        "doc2.md",
                        mtime_ns=doc_two_current.st_mtime_ns,
                        size=doc_two_current.st_size,
                    ),
                },
            ),
        )
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(set(file_paths)),
            enqueued_count=len(set(file_paths)),
        )

    monkeypatch.setattr(
        "src.indexing.tasks.submit_index_batch",
        fake_submit_index_batch,
    )

    await asyncio.wait_for(ctx._bootstrap_via_tasks(), timeout=1.0)

    assert enqueued == [str(file_two)]
    assert ctx._index_state.status == "ready"
    assert ctx._index_state.indexed_count == 2
    assert ctx._index_state.total_count == 2

    @pytest.mark.asyncio
    async def test_state_transitions_during_indexing(
        self, mock_context: Any, tmp_path: Path
    ):
        """Verify state transitions: uninitialized → indexing → ready."""
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")

        files = [str(tmp_path / "doc1.md"), str(tmp_path / "doc2.md")]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))

        observed_states: list[tuple[str, int]] = []

        def tracking_index(path: str):
            observed_states.append(
                (
                    mock_context._index_state.status,
                    mock_context._index_state.indexed_count,
                )
            )
            mock_context.index_manager.index_calls.append(path)

        mock_context.index_manager.index_document = tracking_index

        assert mock_context._index_state.status == "uninitialized"

        await mock_context._background_index()

        # Should have observed indexing state with incrementing counts
        assert ("indexing", 0) in observed_states
        assert ("indexing", 1) in observed_states
        assert mock_context._index_state.status == "ready"


@pytest.mark.asyncio
async def test_background_start_with_existing_index_does_not_block_event_loop(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "index_manager", SlowLoadIndexManager(delay_seconds=0.2))
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "_background_index_task", None)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_is_virgin_startup", False)
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=False))

    reconciliation_calls: list[str] = []

    async def fake_startup_reconciliation() -> None:
        reconciliation_calls.append("called")

    _setattr(ctx, "_startup_reconciliation", fake_startup_reconciliation)

    async def fake_build_initial_vocabulary() -> None:
        return None

    _setattr(ctx, "_build_initial_vocabulary", fake_build_initial_vocabulary)

    await asyncio.wait_for(ctx.start(background_index=True), timeout=0.05)

    assert ctx._index_state.status == "indexing"
    assert not ctx._ready_event.is_set()

    await ctx.ensure_ready(timeout=1.0)

    assert ctx._index_state.status == "ready"
    assert ctx.index_manager.loaded is True
    assert reconciliation_calls == ["called"]


@pytest.mark.asyncio
async def test_task_mode_existing_index_becomes_ready_before_reconciliation(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "index_manager", ExistingIndexManager())
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "_background_index_task", None)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_is_virgin_startup", False)
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=False))

    reconciliation_started = asyncio.Event()
    allow_reconciliation_to_finish = asyncio.Event()

    async def fake_startup_reconciliation() -> None:
        reconciliation_started.set()
        await allow_reconciliation_to_finish.wait()

    async def fake_build_initial_vocabulary() -> None:
        return None

    _setattr(ctx, "_startup_reconciliation", fake_startup_reconciliation)
    _setattr(ctx, "_build_initial_vocabulary", fake_build_initial_vocabulary)

    await asyncio.wait_for(ctx.start(background_index=True), timeout=1.0)
    await asyncio.wait_for(reconciliation_started.wait(), timeout=1.0)

    assert ctx.index_manager.load_calls == 1
    assert ctx.index_manager.loaded is True
    assert ctx._ready_event.is_set()
    assert ctx._index_state.status == "ready"
    assert ctx._background_index_task is not None
    assert ctx._background_index_task.done() is False

    allow_reconciliation_to_finish.set()
    await asyncio.wait_for(ctx._background_index_task, timeout=1.0)


@pytest.mark.asyncio
async def test_task_mode_existing_index_schedules_embedding_warmup(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.indexing.documents_path = str(tmp_path)
    _setattr(ctx, "config", mock_config)
    _setattr(ctx, "use_tasks", True)
    _setattr(ctx, "index_manager", ExistingIndexManager())
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "_background_index_task", None)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
    _setattr(ctx, "_is_virgin_startup", False)
    _setattr(ctx, "_check_and_rebuild_if_needed", MagicMock(return_value=False))

    scheduled: list[str] = []

    def fake_schedule_embedding_model_warmup() -> bool:
        scheduled.append("called")
        return True

    async def fake_startup_reconciliation() -> None:
        return None

    _setattr(
        ctx,
        "schedule_embedding_model_warmup",
        fake_schedule_embedding_model_warmup,
    )
    _setattr(ctx, "_startup_reconciliation", fake_startup_reconciliation)

    await asyncio.wait_for(ctx.start(background_index=True), timeout=1.0)

    assert ctx._ready_event.is_set()
    assert scheduled == ["called"]


@pytest.mark.asyncio
async def test_stop_cancels_inflight_embedding_warmup_task(tmp_path: Path):
    ctx = object.__new__(ApplicationContext)
    _setattr(ctx, "index_path", tmp_path / ".index")
    ctx.index_path.mkdir()
    _setattr(ctx, "watcher", None)
    _setattr(ctx, "commit_indexer", None)
    _setattr(ctx, "reconciliation_task", None)
    _setattr(ctx, "_background_index_task", None)
    _setattr(ctx, "_freshness_task", None)
    _setattr(ctx, "_ready_event", asyncio.Event())
    _setattr(ctx, "_init_error", None)
    _setattr(ctx, "_index_state", IndexState(status="ready"))
    _setattr(ctx, "_is_virgin_startup", False)

    class _Manager:
        def __init__(self) -> None:
            self.persist_calls = 0
            self.vector = WarmupVectorStub()

        def persist(self) -> None:
            self.persist_calls += 1

        def is_ready(self) -> bool:
            return True

    _setattr(ctx, "index_manager", _Manager())
    _setattr(ctx, "_mark_index_state_loaded", lambda: None)

    warmup_started = asyncio.Event()
    allow_warmup_to_finish = asyncio.Event()

    async def fake_run_embedding_model_warmup() -> None:
        warmup_started.set()
        await allow_warmup_to_finish.wait()

    _setattr(ctx, "_run_embedding_model_warmup", fake_run_embedding_model_warmup)

    assert ctx.schedule_embedding_model_warmup() is True
    await asyncio.wait_for(warmup_started.wait(), timeout=1.0)

    await ctx.stop()

    assert ctx._embedding_warmup_task is None


def test_check_and_rebuild_uses_fallback_persisted_index_for_runtime_root(
    tmp_path: Path,
):
    ctx = object.__new__(ApplicationContext)
    mock_config = MockConfig()
    mock_config.llm.embedding_model = "local"
    mock_config.chunking.strategy = "header_based"
    mock_config.chunking.min_chunk_chars = 200
    mock_config.chunking.max_chunk_chars = 2000
    mock_config.chunking.overlap_chars = 100
    _setattr(ctx, "config", mock_config)

    runtime_index_path = tmp_path / "daemon"
    fallback_index_path = tmp_path / "persisted"
    runtime_index_path.mkdir()
    fallback_index_path.mkdir()
    (fallback_index_path / "vector").mkdir()
    (fallback_index_path / "vector" / "docstore.json").write_text("{}")

    save_manifest(
        fallback_index_path,
        IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model="local",
            chunking_config={
                "strategy": "header_based",
                "min_chunk_chars": 200,
                "max_chunk_chars": 2000,
                "overlap_chars": 100,
            },
            indexed_files={"doc": "doc.md"},
        ),
    )

    _setattr(ctx, "index_path", runtime_index_path)
    _setattr(ctx, "fallback_index_path", fallback_index_path)
    _setattr(ctx, "current_manifest", None)
    _setattr(ctx, "_is_virgin_startup", True)

    needs_rebuild = ctx._check_and_rebuild_if_needed()

    assert needs_rebuild is False
    assert ctx._is_virgin_startup is False
    assert (runtime_index_path / "index.manifest.json").exists()
    assert (runtime_index_path / "vector" / "docstore.json").exists()


class TestIsReadyMethods:
    """Tests for is_ready() and is_fully_ready() methods."""

    @pytest.fixture
    def mock_context(self, tmp_path: Path) -> Any:
        """Create a minimal context for testing ready methods."""
        ctx = object.__new__(ApplicationContext)
        _setattr(ctx, "index_manager", MockIndexManager())
        _setattr(ctx, "_ready_event", asyncio.Event())
        _setattr(ctx, "_init_error", None)
        _setattr(ctx, "_index_state", IndexState(status="uninitialized"))
        _setattr(ctx, "_is_virgin_startup", False)
        return ctx

    def test_is_ready_returns_true_when_index_is_queryable_before_ready_event(
        self,
        mock_context: Any,
    ):
        """Verify non-virgin startups can serve queries before ready_event flips."""
        mock_context._index_state = IndexState(status="ready")

        assert mock_context.is_ready() is True

    def test_is_ready_returns_false_when_init_error(self, mock_context: Any):
        """Verify is_ready() returns False when there's an init error."""
        mock_context._ready_event.set()
        mock_context._init_error = RuntimeError("Failed")
        mock_context._index_state = IndexState(status="ready")

        assert mock_context.is_ready() is False

    def test_is_ready_returns_true_for_ready_state(self, mock_context: Any):
        """Verify is_ready() returns True for 'ready' status."""
        mock_context._ready_event.set()
        mock_context._index_state = IndexState(status="ready")

        assert mock_context.is_ready() is True

    def test_is_ready_returns_true_for_partial_state(self, mock_context: Any):
        """Verify is_ready() returns True for 'partial' status."""
        mock_context._ready_event.set()
        mock_context._index_state = IndexState(
            status="partial", indexed_count=5, total_count=10
        )

        assert mock_context.is_ready() is True

    def test_is_ready_returns_true_while_rebuilding_existing_index(
        self,
        mock_context: Any,
    ):
        """Verify background rebuilds can serve queries after prior startup."""
        mock_context._index_state = IndexState(status="indexing", indexed_count=1)
        mock_context._is_virgin_startup = False

        assert mock_context.is_ready() is True

    def test_is_ready_returns_false_during_virgin_startup_indexing(
        self,
        mock_context: Any,
    ):
        """Verify first launch stays strict while indexing is still in progress."""
        mock_context._index_state = IndexState(status="indexing", indexed_count=1)
        mock_context._is_virgin_startup = True

        assert mock_context.is_ready() is False

    def test_is_fully_ready_returns_true_only_for_ready(self, mock_context: Any):
        """Verify is_fully_ready() returns True only for 'ready' status."""
        mock_context._ready_event.set()

        mock_context._index_state = IndexState(status="ready")
        assert mock_context.is_fully_ready() is True

        mock_context._index_state = IndexState(status="partial")
        assert mock_context.is_fully_ready() is False

    def test_is_fully_ready_returns_false_when_init_error(self, mock_context: Any):
        """Verify is_fully_ready() returns False when there's an init error."""
        mock_context._ready_event.set()
        mock_context._index_state = IndexState(status="ready")
        mock_context._init_error = RuntimeError("Failed")

        assert mock_context.is_fully_ready() is False


class TestGetIndexState:
    """Tests for get_index_state() method."""

    def test_get_index_state_returns_current_state(self, tmp_path: Path):
        """Verify get_index_state() returns the current IndexState."""
        ctx = object.__new__(ApplicationContext)
        expected_state = IndexState(
            status="partial",
            indexed_count=3,
            total_count=5,
            last_error="Some error",
        )
        ctx._index_state = expected_state

        result = ctx.get_index_state()

        assert result is expected_state
        assert result.status == "partial"
        assert result.indexed_count == 3
        assert result.total_count == 5
        assert result.last_error == "Some error"
