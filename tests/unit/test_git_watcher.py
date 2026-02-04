from pathlib import Path
from queue import Queue

import pytest

from src.config import Config, IndexingConfig, GitIndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.git.commit_indexer import CommitIndexer
from src.git.watcher import GitWatcher, _GitEventHandler


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


def test_git_watcher_instantiation(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.5,
    )

    assert watcher is not None
    assert watcher._git_repos == git_repos
    assert watcher._commit_indexer is commit_indexer
    assert watcher._config is test_config
    assert watcher._cooldown == 0.5
    assert watcher._running is False


def test_git_watcher_constructor_types(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=1.0,
    )

    from src.git.commit_indexer import CommitIndexer
    from src.config import Config

    assert isinstance(watcher._commit_indexer, CommitIndexer)
    assert isinstance(watcher._config, Config)
    assert isinstance(watcher._git_repos, list)
    assert all(isinstance(p, Path) for p in watcher._git_repos)


def test_git_event_handler_instantiation(tmp_path, test_config, commit_indexer):
    git_dir = tmp_path / ".git"
    event_queue = Queue()
    git_repos = [git_dir]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
    )

    handler = _GitEventHandler(
        watcher=watcher,
        event_queue=event_queue,
        git_dir=git_dir,
    )

    assert handler is not None
    assert handler._queue is event_queue
    assert handler._git_dir == git_dir
    assert handler._watcher is watcher


def test_git_watcher_empty_repos_list(test_config, commit_indexer):
    watcher = GitWatcher(
        git_repos=[],
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.5,
    )

    assert watcher._git_repos == []
    assert watcher._running is False


def test_git_watcher_default_cooldown(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
    )

    assert watcher._cooldown == 5.0


def test_git_watcher_config_access(test_config, commit_indexer, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.5,
    )

    assert watcher._config.git_indexing.delta_max_lines == 200
    assert watcher._config.git_indexing.batch_size == 100


@pytest.mark.asyncio
async def test_git_watcher_lifecycle(test_config, commit_indexer, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    git_repos = [git_dir]

    watcher = GitWatcher(
        git_repos=git_repos,
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.1,
    )

    assert not watcher._running

    watcher.start()
    assert watcher._running

    await watcher.stop()
    assert not watcher._running
