import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_markdown_ragdocs.config import (
    Config,
    GitIndexingConfig,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.git.watcher import GitWatcher
from mcp_markdown_ragdocs.indexing.manager import IndexManager
from mcp_markdown_ragdocs.indexing.tasks import (
    GIT_REFRESH_BATCH_SIZE,
    TaskSubmissionResult,
)


@pytest.fixture
def test_config(tmp_path):
    return Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / ".index_data"),
        ),
        git_indexing=GitIndexingConfig(
            enabled=True, watch_enabled=True, poll_interval_seconds=5.0
        ),
        search=SearchConfig(),
        llm=LLMConfig(),
    )


@pytest.fixture
def index_manager() -> MagicMock:
    return MagicMock(spec=IndexManager)


def test_git_watcher_instantiation(test_config, index_manager, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        index_manager=index_manager,
        config=test_config,
        poll_interval=0.5,
    )

    assert watcher is not None
    assert watcher._git_repos == git_repos
    assert watcher._index_manager is index_manager
    assert watcher._config is test_config
    assert watcher._poll_interval == 0.5
    assert watcher._running is False


def test_git_watcher_constructor_types(test_config, index_manager, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        index_manager=index_manager,
        config=test_config,
        poll_interval=1.0,
    )

    assert isinstance(watcher._config, Config)
    assert isinstance(watcher._git_repos, list)
    assert all(isinstance(p, Path) for p in watcher._git_repos)


def test_git_watcher_empty_repos_list(test_config, index_manager):
    watcher = GitWatcher(
        git_repos=[],
        index_manager=index_manager,
        config=test_config,
        poll_interval=0.5,
    )

    assert watcher._git_repos == []
    assert watcher._running is False


def test_git_watcher_default_poll_interval(test_config, index_manager, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos, index_manager=index_manager, config=test_config
    )

    assert watcher._poll_interval == 30.0
    assert watcher._use_tasks is False


def test_git_watcher_config_access(test_config, index_manager, tmp_path):
    git_repos = [tmp_path / ".git"]

    watcher = GitWatcher(
        git_repos=git_repos,
        index_manager=index_manager,
        config=test_config,
        poll_interval=0.5,
    )

    assert watcher._config.git_indexing.poll_interval_seconds == 5.0


@pytest.mark.asyncio
async def test_git_watcher_lifecycle(test_config, index_manager, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
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
async def test_git_watcher_idempotent_start(test_config, index_manager, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
        config=test_config,
        poll_interval=100.0,
    )

    watcher.start()
    first_task = watcher._task

    watcher.start()
    assert watcher._task is first_task  # same task, not replaced

    await watcher.stop()


@pytest.mark.asyncio
async def test_git_watcher_idempotent_stop(test_config, index_manager, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
        config=test_config,
        poll_interval=100.0,
    )

    watcher.start()
    await watcher.stop()
    await watcher.stop()  # second stop must not raise
    assert not watcher._running


@pytest.mark.asyncio
async def test_git_watcher_enqueues_refresh_tasks_when_enabled(
    test_config, index_manager, tmp_path, monkeypatch
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    observed: list[str] = []

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
        config=test_config,
        use_tasks=True,
    )

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.submit_refresh_git_request",
        lambda git_dir_str: observed.append(git_dir_str)
        or TaskSubmissionResult(status="enqueued"),
    )

    await watcher._batch_process({git_dir})

    assert observed == [str(git_dir)]


@pytest.mark.asyncio
async def test_git_watcher_skips_unchanged_repository_after_refresh(
    test_config, index_manager, tmp_path, monkeypatch
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    observed: list[str] = []

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
        config=test_config,
        use_tasks=True,
    )

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.get_git_ref_signature",
        lambda _git_dir: "same-head",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.get_head",
        lambda _root, _git_dir: "same-head",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.submit_refresh_git_request",
        lambda git_dir_str: observed.append(git_dir_str)
        or TaskSubmissionResult(status="enqueued"),
    )

    await watcher._batch_process({git_dir})

    assert observed == []


@pytest.mark.asyncio
async def test_git_watcher_direct_refresh_accumulates_batches(
    test_config, index_manager, tmp_path, monkeypatch, caplog
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    observed: dict[str, object] = {}

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
        config=test_config,
        use_tasks=True,
    )
    watcher._last_indexed[git_dir] = 123

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.submit_refresh_git_request",
        lambda git_dir_str: TaskSubmissionResult(status="unavailable"),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.get_git_ref_signature",
        lambda _git_dir: "new-head",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.get_head",
        lambda _root, _git_dir: "old-head",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.resolve_project_id_for_path",
        lambda _path, _config: "repo-project",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.time.time",
        lambda: 456,
    )

    async def _receipts(_manager, source, *, since, batch_size):
        observed["repo_path"] = source.repo_path
        observed["workspace_id"] = source.workspace_id
        observed["since"] = since
        observed["batch_size"] = batch_size
        yield SimpleNamespace(records=(object(), object()))
        yield SimpleNamespace(records=(object(),))

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.git_ingestion.iter_git_ingestion_receipts",
        _receipts,
    )

    caplog.set_level(logging.INFO)
    await watcher._batch_process({git_dir})

    assert observed["repo_path"] == git_dir.parent
    assert observed["workspace_id"] == "repo-project"
    assert observed["since"] == "123"
    assert observed["batch_size"] == GIT_REFRESH_BATCH_SIZE
    assert watcher._last_indexed[git_dir] == 456
    assert f"Updated commit index for {tmp_path.name}: 3 commits" in caplog.text


@pytest.mark.asyncio
async def test_git_watcher_direct_refresh_failure_preserves_cursor(
    test_config, index_manager, tmp_path, monkeypatch
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    watcher = GitWatcher(
        git_repos=[git_dir],
        index_manager=index_manager,
        config=test_config,
        use_tasks=True,
    )
    watcher._last_indexed[git_dir] = 123

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.submit_refresh_git_request",
        lambda git_dir_str: TaskSubmissionResult(status="unavailable"),
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.get_git_ref_signature",
        lambda _git_dir: "new-head",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.git.watcher.get_head",
        lambda _root, _git_dir: "old-head",
    )

    async def _receipts(_manager, _source, *, since, batch_size):
        assert since == "123"
        assert batch_size == GIT_REFRESH_BATCH_SIZE
        yield SimpleNamespace(records=(object(),))
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.git_ingestion.iter_git_ingestion_receipts",
        _receipts,
    )

    await watcher._batch_process({git_dir})

    assert watcher._last_indexed[git_dir] == 123
