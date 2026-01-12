import subprocess
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, GitIndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.git.commit_indexer import CommitIndexer
from src.git.watcher import GitWatcher
from src.git.repository import is_git_available


def _init_git_repo(path: Path):
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _create_commit(repo_path: Path, file_name: str, content: str, message: str):
    file_path = repo_path / file_name
    file_path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


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
            watch_cooldown=0.5,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(),
    )


@pytest.fixture
def commit_indexer(tmp_path, shared_embedding_model):
    db_path = tmp_path / "commits.db"
    return CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)


@pytest.fixture
def git_repo(tmp_path):
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    _create_commit(repo_path, "README.md", "# Test Repo", "Initial commit")

    return repo_path


@pytest.mark.skipif(not is_git_available(), reason="git not available")
@pytest.mark.asyncio
async def test_git_watcher_start_stop(test_config, commit_indexer, git_repo):
    git_dir = git_repo / ".git"

    watcher = GitWatcher(
        git_repos=[git_dir],
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.1,
    )

    watcher.start()
    assert watcher._running
    assert len(watcher._observers) > 0
    assert watcher._task is not None

    await watcher.stop()
    assert not watcher._running
    assert len(watcher._observers) == 0
    assert watcher._task is None


@pytest.mark.skipif(not is_git_available(), reason="git not available")
@pytest.mark.asyncio
async def test_git_watcher_idempotent_start(test_config, commit_indexer, git_repo):
    git_dir = git_repo / ".git"

    watcher = GitWatcher(
        git_repos=[git_dir],
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.1,
    )

    watcher.start()
    initial_observers = len(watcher._observers)

    watcher.start()
    assert len(watcher._observers) == initial_observers

    await watcher.stop()


@pytest.mark.skipif(not is_git_available(), reason="git not available")
@pytest.mark.asyncio
async def test_git_watcher_idempotent_stop(test_config, commit_indexer, git_repo):
    git_dir = git_repo / ".git"

    watcher = GitWatcher(
        git_repos=[git_dir],
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.1,
    )

    watcher.start()
    await watcher.stop()

    await watcher.stop()
    assert not watcher._running


@pytest.mark.skipif(not is_git_available(), reason="git not available")
@pytest.mark.asyncio
async def test_git_watcher_multiple_repos(test_config, commit_indexer, tmp_path):
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"

    repo1.mkdir()
    repo2.mkdir()

    _init_git_repo(repo1)
    _init_git_repo(repo2)

    _create_commit(repo1, "README.md", "# Repo 1", "Initial commit")
    _create_commit(repo2, "README.md", "# Repo 2", "Initial commit")

    git_dir1 = repo1 / ".git"
    git_dir2 = repo2 / ".git"

    watcher = GitWatcher(
        git_repos=[git_dir1, git_dir2],
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.1,
    )

    watcher.start()
    assert len(watcher._git_repos) == 2

    await watcher.stop()


@pytest.mark.skipif(not is_git_available(), reason="git not available")
@pytest.mark.asyncio
async def test_git_watcher_nonexistent_paths_skipped(test_config, commit_indexer, tmp_path):
    git_dir_exists = tmp_path / "exists" / ".git"
    git_dir_exists.parent.mkdir()
    _init_git_repo(git_dir_exists.parent)

    git_dir_missing = tmp_path / "missing" / ".git"

    watcher = GitWatcher(
        git_repos=[git_dir_exists, git_dir_missing],
        commit_indexer=commit_indexer,
        config=test_config,
        cooldown=0.1,
    )

    watcher.start()

    await watcher.stop()
