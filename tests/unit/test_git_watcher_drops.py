"""Tests for GitWatcher dropped event tracking and catch-up mechanism."""

import queue
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    Config,
    GitIndexingConfig,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
    ServerConfig,
)
from src.git.commit_indexer import CommitIndexer
from src.git.watcher import MAX_QUEUE_SIZE, GitWatcher, _GitEventHandler


@pytest.fixture
def test_config(tmp_path):
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index_data"),
        ),
        git_indexing=GitIndexingConfig(
            enabled=True,
            delta_max_lines=200,
            batch_size=100,
            watch_enabled=True,
            watch_cooldown=5.0,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(),
    )


@pytest.fixture
def commit_indexer(tmp_path, shared_embedding_model):
    db_path = tmp_path / "commits.db"
    return CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)


class TestDroppedEventCounter:
    """Tests for dropped event counter functionality."""

    def test_dropped_event_count_initial_zero(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        assert watcher.dropped_event_count == 0

    def test_dropped_event_count_property(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        # Simulate drops
        watcher._dropped_events = 42

        assert watcher.dropped_event_count == 42

    def test_reset_dropped_counter(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        watcher._dropped_events = 100
        watcher.reset_dropped_counter()

        assert watcher.dropped_event_count == 0

    def test_handler_increments_dropped_count_on_full_queue(
        self, test_config, commit_indexer, tmp_path
    ):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        # Create a tiny queue that's already full
        tiny_queue: queue.Queue[Path] = queue.Queue(maxsize=1)
        tiny_queue.put(Path("/dummy"))

        handler = _GitEventHandler(
            watcher=watcher,
            event_queue=tiny_queue,
            git_dir=tmp_path / ".git",
        )

        # Queue is full, this should increment dropped counter
        handler._queue_event()

        assert watcher.dropped_event_count == 1

    def test_multiple_drops_accumulate(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        # Create full queue
        tiny_queue: queue.Queue[Path] = queue.Queue(maxsize=1)
        tiny_queue.put(Path("/dummy"))

        handler = _GitEventHandler(
            watcher=watcher,
            event_queue=tiny_queue,
            git_dir=tmp_path / ".git",
        )

        # Multiple drops
        for _ in range(25):
            handler._queue_event()

        assert watcher.dropped_event_count == 25


class TestLogThrottling:
    """Tests for log throttling on dropped events."""

    def test_log_on_threshold_multiple(self, test_config, commit_indexer, tmp_path, caplog):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )
        # Default threshold is 10

        tiny_queue: queue.Queue[Path] = queue.Queue(maxsize=1)
        tiny_queue.put(Path("/dummy"))

        handler = _GitEventHandler(
            watcher=watcher,
            event_queue=tiny_queue,
            git_dir=tmp_path / ".git",
        )

        import logging

        with caplog.at_level(logging.WARNING):
            # Drop 9 events - no log
            for _ in range(9):
                handler._queue_event()

            assert "queue full" not in caplog.text.lower()

            # 10th drop should log
            handler._queue_event()

            assert "queue full" in caplog.text.lower()
            assert "10 events dropped" in caplog.text

    def test_log_every_n_drops(self, test_config, commit_indexer, tmp_path, caplog):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )
        watcher._dropped_log_threshold = 5  # Log every 5 drops

        tiny_queue: queue.Queue[Path] = queue.Queue(maxsize=1)
        tiny_queue.put(Path("/dummy"))

        handler = _GitEventHandler(
            watcher=watcher,
            event_queue=tiny_queue,
            git_dir=tmp_path / ".git",
        )

        import logging

        with caplog.at_level(logging.WARNING):
            for i in range(1, 16):
                handler._queue_event()

                if i in (5, 10, 15):
                    assert f"{i} events dropped" in caplog.text
                    caplog.clear()


class TestShouldCatchup:
    """Tests for should_catchup flag."""

    def test_should_catchup_false_initially(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        assert watcher.should_catchup() is False

    def test_should_catchup_true_after_drops(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        watcher._dropped_events = 1

        assert watcher.should_catchup() is True

    def test_should_catchup_false_after_reset(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )

        watcher._dropped_events = 50
        assert watcher.should_catchup() is True

        watcher.reset_dropped_counter()
        assert watcher.should_catchup() is False


class TestCatchupMechanism:
    """Tests for run_catchup method."""

    @pytest.mark.asyncio
    async def test_run_catchup_resets_counter(self, test_config, commit_indexer, tmp_path):
        git_repos = [tmp_path / ".git"]
        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )
        watcher._dropped_events = 25

        with patch("src.git.repository.get_commits_after_timestamp", return_value=[]):
            result = await watcher.run_catchup()

        assert watcher.dropped_event_count == 0
        assert result == 0  # No commits to index

    @pytest.mark.asyncio
    async def test_run_catchup_indexes_new_commits(
        self, test_config, commit_indexer, tmp_path
    ):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        git_repos = [git_dir]

        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )
        watcher._dropped_events = 5

        # Mock the git functions to simulate new commits
        with (
            patch(
                "src.git.repository.get_commits_after_timestamp",
                return_value=["abc123", "def456"],
            ),
            patch(
                "src.git.parallel_indexer.index_commits_parallel",
                return_value=2,
            ) as mock_index,
        ):
            result = await watcher.run_catchup()

        assert result == 2
        assert watcher.dropped_event_count == 0
        mock_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_catchup_handles_multiple_repos(
        self, test_config, commit_indexer, tmp_path
    ):
        git_dir1 = tmp_path / "repo1" / ".git"
        git_dir2 = tmp_path / "repo2" / ".git"
        git_dir1.mkdir(parents=True)
        git_dir2.mkdir(parents=True)

        git_repos = [git_dir1, git_dir2]

        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )
        watcher._dropped_events = 10

        with (
            patch(
                "src.git.repository.get_commits_after_timestamp",
                side_effect=[["abc123"], ["def456", "ghi789"]],
            ),
            patch(
                "src.git.parallel_indexer.index_commits_parallel",
                side_effect=[1, 2],
            ),
        ):
            result = await watcher.run_catchup()

        assert result == 3  # 1 + 2 commits from both repos
        assert watcher.dropped_event_count == 0

    @pytest.mark.asyncio
    async def test_run_catchup_continues_on_single_repo_failure(
        self, test_config, commit_indexer, tmp_path
    ):
        git_dir1 = tmp_path / "repo1" / ".git"
        git_dir2 = tmp_path / "repo2" / ".git"
        git_dir1.mkdir(parents=True)
        git_dir2.mkdir(parents=True)

        git_repos = [git_dir1, git_dir2]

        watcher = GitWatcher(
            git_repos=git_repos,
            commit_indexer=commit_indexer,
            config=test_config,
        )
        watcher._dropped_events = 5

        def mock_get_commits(git_dir, timestamp):
            if git_dir == git_dir1:
                raise RuntimeError("Repo 1 error")
            return ["abc123"]

        with (
            patch(
                "src.git.repository.get_commits_after_timestamp",
                side_effect=mock_get_commits,
            ),
            patch(
                "src.git.parallel_indexer.index_commits_parallel",
                return_value=1,
            ),
        ):
            result = await watcher.run_catchup()

        # Should still index repo2 despite repo1 failure
        assert result == 1
        assert watcher.dropped_event_count == 0

    @pytest.mark.asyncio
    async def test_run_catchup_skips_empty_repos(
        self, test_config, commit_indexer, tmp_path
    ):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        watcher = GitWatcher(
            git_repos=[git_dir],
            commit_indexer=commit_indexer,
            config=test_config,
        )
        watcher._dropped_events = 3

        with (
            patch(
                "src.git.repository.get_commits_after_timestamp",
                return_value=[],  # No new commits
            ),
            patch(
                "src.git.parallel_indexer.index_commits_parallel",
            ) as mock_index,
        ):
            result = await watcher.run_catchup()

        assert result == 0
        mock_index.assert_not_called()
        assert watcher.dropped_event_count == 0


class TestMaxQueueSize:
    """Tests for MAX_QUEUE_SIZE constant."""

    def test_max_queue_size_constant(self):
        assert MAX_QUEUE_SIZE == 1000

    def test_watcher_uses_bounded_queue(self, test_config, commit_indexer, tmp_path):
        watcher = GitWatcher(
            git_repos=[tmp_path / ".git"],
            commit_indexer=commit_indexer,
            config=test_config,
        )

        assert watcher._event_queue.maxsize == MAX_QUEUE_SIZE
