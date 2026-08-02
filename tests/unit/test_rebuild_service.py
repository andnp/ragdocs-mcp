from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import Record

import mcp_markdown_ragdocs.indexing.rebuild_service as rebuild_service


class _FakeSource:
    def __init__(self, _repo_path: Path) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        self._records = [
            Record(
                source_kind="git_commit",
                source_id=f"git:{index}",
                title=f"Commit {index}",
                body="Body",
                created_at=timestamp,
                updated_at=timestamp,
            )
            for index in range(27)
        ]

    def iter_records(self):
        yield from self._records


class _FakeIndexManager:
    def __init__(self) -> None:
        self.persist_checkpoint_calls = 0
        self.indexed_records: list[Record] = []

    def index_record(self, record: Record) -> bool:
        self.indexed_records.append(record)
        return True

    def persist_checkpoint(self) -> None:
        self.persist_checkpoint_calls += 1


def test_ingest_git_repository_checkpoints_bounded_batches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manager = _FakeIndexManager()
    progress: list[dict[str, object]] = []

    monkeypatch.setattr(rebuild_service, "GitContentSource", _FakeSource)
    monkeypatch.setattr(
        rebuild_service,
        "_update_rebuild_progress",
        lambda _runtime_root, **changes: progress.append(changes),
    )

    total_indexed = rebuild_service._ingest_git_repository(
        runtime_root=tmp_path,
        index_manager=manager,
        repo_path=tmp_path / ".git",
        git_commits_indexed=5,
    )

    assert total_indexed == 32
    assert len(manager.indexed_records) == 27
    assert manager.persist_checkpoint_calls == 2
    assert [entry["git_commits_indexed"] for entry in progress] == [30, 32]
    assert all(entry["phase"] == "indexing_git" for entry in progress)
