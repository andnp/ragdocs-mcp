"""Unit tests for the attribution-repair skip on unchanged git repositories.

Covers the ordering fix in `_refresh_git_repository`: an unchanged repository
must return early without paying for a full `reconcile_git_project_attribution`
call (a full history walk plus a full record-store scan), while a changed
repository must still trigger reconciliation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from huey import SqliteHuey

from searchkernel.domain import Record

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.indexing.git_refresh_state import save_cursor, save_head
from mcp_markdown_ragdocs.indexing.tasks import enqueue_refresh_git, register_tasks


class FakeIndexManager:
    """Minimal stub recording whether attribution reconciliation ran."""

    ingestor = object()

    def __init__(self) -> None:
        self.persist_calls = 0
        self.reconcile_calls: list[tuple[Path, str | None, list[str] | None]] = []

    def index_document(self, file_path: str, force: bool = False) -> bool:
        return True

    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ) -> None:
        return None

    def remove_document(self, doc_id: str) -> None:
        return None

    def remove_documents(self, doc_ids: list[str], persist: bool = False) -> None:
        return None

    def persist(self) -> None:
        self.persist_calls += 1

    def index_record(self, record: Record) -> None:
        return None

    def reconcile_git_project_attribution(
        self,
        git_dir: Path,
        workspace_id: str | None,
        commit_hashes=None,
    ) -> int:
        self.reconcile_calls.append(
            (
                git_dir,
                workspace_id,
                None if commit_hashes is None else list(commit_hashes),
            )
        )
        return 1


@pytest.fixture()
def huey_instance(tmp_path: Path) -> SqliteHuey:
    return SqliteHuey(
        name="test-git-refresh-task",
        filename=str(tmp_path / "tasks.db"),
        immediate=False,
    )


@pytest.fixture()
def fake_manager(tmp_path: Path) -> FakeIndexManager:
    manager = FakeIndexManager()
    cast(Any, manager)._config = SimpleNamespace(
        projects=[],
        detected_project="repo-project",
        indexing=SimpleNamespace(documents_path=str(tmp_path)),
        git_indexing=SimpleNamespace(git_diff_embedding_days=0),
    )
    return manager


def _register_tasks(
    huey: SqliteHuey, manager: FakeIndexManager, **kwargs: Any
) -> Any:
    queue_path = Path(cast(Any, huey.storage).filename)
    return register_tasks(
        huey,
        manager,
        TaskLeaseStore(queue_path),
        WorkIntentStore(queue_path),
        **kwargs,
    )


def test_refresh_git_skips_reconciliation_when_repository_unchanged(
    huey_instance: SqliteHuey,
    fake_manager: FakeIndexManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "index"
    git_dir = tmp_path / "repo" / ".git"
    git_dir.parent.mkdir(parents=True)
    git_dir.mkdir()
    save_head(state_root, git_dir, "same-head")
    save_cursor(state_root, git_dir, 123)

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.get_git_ref_signature",
        lambda _git_dir: "same-head",
    )

    runtime = _register_tasks(
        huey_instance,
        fake_manager,
        bootstrap_index_path=state_root,
    )
    enqueue_refresh_git(str(git_dir), runtime=runtime)
    task = huey_instance.dequeue()
    assert task is not None

    assert huey_instance.execute(task) is True
    assert fake_manager.reconcile_calls == []
    assert fake_manager.persist_calls == 0


def test_refresh_git_runs_reconciliation_when_repository_changed(
    huey_instance: SqliteHuey,
    fake_manager: FakeIndexManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "index"
    git_dir = tmp_path / "repo" / ".git"
    git_dir.parent.mkdir(parents=True)
    git_dir.mkdir()
    save_head(state_root, git_dir, "old-head")
    save_cursor(state_root, git_dir, 123)

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.get_git_ref_signature",
        lambda _git_dir: "new-head",
    )
    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.tasks.iter_commit_hashes_after_timestamp",
        lambda _git_dir, after_timestamp: iter(
            [f"changed-after-{after_timestamp}"]
        ),
    )

    async def _receipts(_manager, _source, *, since, batch_size):
        yield SimpleNamespace(records=(), failed=0, checkpoint=None)

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.indexing.git_ingestion.iter_git_ingestion_receipts",
        _receipts,
    )

    runtime = _register_tasks(
        huey_instance,
        fake_manager,
        bootstrap_index_path=state_root,
    )
    enqueue_refresh_git(str(git_dir), runtime=runtime)
    task = huey_instance.dequeue()
    assert task is not None

    assert huey_instance.execute(task) is True
    assert len(fake_manager.reconcile_calls) == 1
    assert fake_manager.reconcile_calls[0][0] == git_dir.resolve()
    assert fake_manager.reconcile_calls[0][2] == ["changed-after-122"]
    assert fake_manager.persist_calls >= 1
