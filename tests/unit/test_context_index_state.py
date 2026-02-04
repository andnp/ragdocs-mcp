"""
Unit tests for ApplicationContext IndexState tracking and retry behavior.

Tests cover:
- IndexState dataclass
- Partial failure handling in _background_index()
- Retry behavior with exponential backoff
- State transitions (uninitialized → indexing → ready/partial/failed)
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.context import ApplicationContext, IndexState


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
    parsers: dict = field(default_factory=dict)
    document_chunking: MagicMock = field(default_factory=MagicMock)


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
        return ctx

    @pytest.mark.asyncio
    async def test_successful_indexing_sets_ready_state(self, mock_context: Any, tmp_path: Path):
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
    async def test_partial_failure_sets_partial_state(self, mock_context: Any, tmp_path: Path):
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
    async def test_complete_failure_sets_failed_state(self, mock_context: Any, tmp_path: Path):
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
    async def test_retry_with_exponential_backoff(self, mock_context: Any, tmp_path: Path):
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
    async def test_retry_succeeds_on_second_attempt(self, mock_context: Any, tmp_path: Path):
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
    async def test_state_transitions_during_indexing(self, mock_context: Any, tmp_path: Path):
        """Verify state transitions: uninitialized → indexing → ready."""
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")

        files = [str(tmp_path / "doc1.md"), str(tmp_path / "doc2.md")]
        _setattr(mock_context, "discover_files", MagicMock(return_value=files))

        observed_states: list[tuple[str, int]] = []

        def tracking_index(path: str):
            observed_states.append(
                (mock_context._index_state.status, mock_context._index_state.indexed_count)
            )
            mock_context.index_manager.index_calls.append(path)

        mock_context.index_manager.index_document = tracking_index

        assert mock_context._index_state.status == "uninitialized"

        await mock_context._background_index()

        # Should have observed indexing state with incrementing counts
        assert ("indexing", 0) in observed_states
        assert ("indexing", 1) in observed_states
        assert mock_context._index_state.status == "ready"


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
        return ctx

    def test_is_ready_returns_false_when_event_not_set(self, mock_context: Any):
        """Verify is_ready() returns False when _ready_event not set."""
        mock_context._index_state = IndexState(status="ready")

        assert mock_context.is_ready() is False

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
        mock_context._index_state = IndexState(status="partial", indexed_count=5, total_count=10)

        assert mock_context.is_ready() is True

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
