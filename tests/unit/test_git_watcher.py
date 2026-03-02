from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, GitIndexingConfig, LLMConfig, SearchConfig
from src.git.commit_indexer import CommitIndexer
from src.git.watcher import GitWatcher


@pytest.fixture
def test_config(tmp_path):
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index_data")
        ),
        git_indexing=GitIndexingConfig(
            enabled=True,
            watch_enabled=True,
            poll_interval_seconds=5.0
        ),
        search=SearchConfig(),
        llm=LLMConfig()
    )


@pytest.fixture
def commit_indexer(tmp_path, shared_embedding_model):
    db_path = tmp_path / "commits.db"
    return CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)


def test_git_watcher_instantiation(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=0.5
    )

    assert watcher is not None
    assert watcher._git_repos == git_repos
    assert watcher._commit_indexer is commit_indexer
    assert watcher._config is test_config
    assert watcher._poll_interval == 0.5
    assert watcher._running is False


def test_git_watcher_constructor_types(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=1.0
    )

    from src.git.commit_indexer import CommitIndexer
    from src.config import Config

    assert isinstance(watcher._commit_indexer, CommitIndexer)
    assert isinstance(watcher._config, Config)
    assert isinstance(watcher._git_repos, list)
    assert all(isinstance(p, Path) for p in watcher._git_repos)


def test_git_watcher_empty_repos_list(test_config, commit_indexer):
    watcher = GitWatcher(
        git_repos=[],
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=0.5
    )

    assert watcher._git_repos == []
    assert watcher._running is False


def test_git_watcher_default_poll_interval(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config
    )

    assert watcher._poll_interval == 30.0


def test_git_watcher_config_access(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=0.5
    )

    assert watcher._config.git_indexing.poll_interval_seconds == 5.0


@pytest.mark.asyncio
async def test_git_watcher_lifecycle(test_config, commit_indexer, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=100.0,  # long interval so it never fires during test
    )

    assert not watcher._running

    watcher.start()
    assert watcher._running
    assert watcher._task is not None

    await watcher.stop()
    assert not watcher._running
    assert watcher._task is None


@pytest.mark.asyncio
async def test_git_watcher_idempotent_start(test_config, commit_indexer, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=100.0
    )

    watcher.start()
    first_task = watcher._task

    watcher.start()
    assert watcher._task is first_task  # same task, not replaced

    await watcher.stop()


@pytest.mark.asyncio
async def test_git_watcher_idempotent_stop(test_config, commit_indexer, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        commit_indexer=commit_indexer,
        config=test_config,
        poll_interval=100.0
    )

    watcher.start()
    await watcher.stop()
    await watcher.stop()  # second stop must not raise
    assert not watcher._running
