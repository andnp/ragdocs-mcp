from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from mcp_markdown_ragdocs.config import Config
from searchkernel.domain import Record

import mcp_markdown_ragdocs.indexing.rebuild_service as rebuild_service


class _FakeSource:
    def __init__(self, _repo_path: Path, workspace_id: str | None = None) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        self._records = [
            Record(
                source_kind="git_commit",
                source_id=f"git:{index}",
                workspace_id=workspace_id,
                title=f"Commit {index}",
                body="Body",
                created_at=timestamp,
                updated_at=timestamp,
            )
            for index in range(27)
        ]

    def iter_records(self):
        yield from self._records


class _FakeIngestor:
    def __init__(self, manager) -> None:
        self._manager = manager

    async def index_records(self, records, **_kwargs):
        for record in records:
            self._manager.index_record(record)
        return SimpleNamespace(successful=len(records), failed=0)


class _FakeIndexManager:
    def __init__(self) -> None:
        self.persist_checkpoint_calls = 0
        self.indexed_records: list[Record] = []
        self.ingestor = _FakeIngestor(self)

    def index_record(self, record: Record) -> bool:
        self.indexed_records.append(record)
        return True

    def persist_checkpoint(self) -> None:
        self.persist_checkpoint_calls += 1


class _RebuildManager:
    def __init__(self, *, fail_document_batch: int | None = None) -> None:
        self.clear_calls = 0
        self.indexed_batches: list[list[str]] = []
        self.persist_checkpoint_calls = 0
        self.fail_document_batch = fail_document_batch
        self._encoder_fingerprint = None
        self.embedding_provider = SimpleNamespace(model_name="test-embedding")
        self._failed_files: list[dict[str, str]] = []
        self.indexed_records: list[Record] = []
        self.ingestor = _FakeIngestor(self)

    def clear_documents(self) -> None:
        self.clear_calls += 1

    def index_documents(
        self,
        file_paths: list[str],
        *,
        force: bool,
        persist: bool,
    ) -> None:
        _ = force, persist
        self.indexed_batches.append(list(file_paths))
        if self.fail_document_batch == len(self.indexed_batches):
            raise RuntimeError("interrupted document batch")

    def get_failed_files(self) -> list[dict[str, str]]:
        return list(self._failed_files)

    def persist_checkpoint(self) -> None:
        self.persist_checkpoint_calls += 1

    def finalize_derived_graph_state(self) -> None:
        return

    def index_record(self, record: Record) -> bool:
        self.indexed_records.append(record)
        return True

def _rebuild_config(tmp_path: Path, *, git_enabled: bool = False) -> Config:
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    config = Config()
    config.indexing.documents_path = str(documents_root)
    config.indexing.index_path = str(tmp_path / "index")
    config.indexing.include = ["**/*.md"]
    config.indexing.exclude = []
    config.indexing.rebuild_checkpoint_interval = 2
    config.git_indexing.enabled = git_enabled
    return config


def _run_rebuild(
    *,
    tmp_path: Path,
    config: Config,
    manager: _RebuildManager,
    request_id: str,
) -> dict[str, object]:
    return rebuild_service.run_rebuild(
        runtime_root=tmp_path / "index",
        config=config,
        index_manager=manager,
        global_documents_roots=[Path(config.indexing.documents_path)],
        request_id=request_id,
        project_override=None,
    )


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
        workspace_id="project-a",
    )

    assert total_indexed == 32
    assert len(manager.indexed_records) == 27
    assert {record.workspace_id for record in manager.indexed_records} == {"project-a"}
    assert manager.persist_checkpoint_calls == 2
    assert [entry["git_commits_indexed"] for entry in progress] == [30, 32]
    assert all(entry["phase"] == "indexing_git" for entry in progress)


def test_rebuild_status_round_trip_preserves_telemetry(tmp_path: Path) -> None:
    payload = rebuild_service.write_rebuild_status(
        tmp_path,
        {
            "status": "running",
            "phase": "indexing_documents",
            "request_id": "request-1",
            "documents_total": 4,
            "documents_completed": 2,
            "current_document_path": "/docs/two.md",
            "last_checkpoint_at": 100.0,
            "elapsed_seconds": 3.0,
        },
    )

    assert rebuild_service.read_rebuild_status(tmp_path) == payload
    assert payload["schema_version"] == 1
    assert payload["documents_completed"] == 2
    assert payload["current_document_path"] == "/docs/two.md"


def test_rebuild_status_progress_is_monotonic(tmp_path: Path) -> None:
    rebuild_service.write_rebuild_status(
        tmp_path,
        {
            "request_id": "request-1",
            "indexed_files": 5,
            "documents_completed": 5,
            "git_commits_indexed": 8,
            "git_records_completed": 8,
        },
    )
    status = rebuild_service.write_rebuild_status(
        tmp_path,
        {
            "request_id": "request-1",
            "indexed_files": 2,
            "documents_completed": 2,
            "git_commits_indexed": 3,
            "git_records_completed": 3,
        },
    )

    assert status["indexed_files"] == 5
    assert status["documents_completed"] == 5
    assert status["git_commits_indexed"] == 8
    assert status["git_records_completed"] == 8


def test_rebuild_status_timing_and_eta_use_deterministic_clock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = iter([100.0, 110.0])
    monkeypatch.setattr(rebuild_service.time, "time", lambda: next(now))

    rebuild_service.write_rebuild_status(
        tmp_path,
        {
            "status": "running",
            "phase": "indexing_documents",
            "request_id": "request-1",
            "submitted_at": 95.0,
            "started_at": 100.0,
            "documents_total": 4,
            "documents_completed": 2,
        },
    )
    status = rebuild_service.write_rebuild_status(
        tmp_path,
        {
            "request_id": "request-1",
            "documents_total": 4,
            "documents_completed": 2,
        },
    )

    assert status["elapsed_seconds"] == 10.0
    assert status["processing_rate"] == 0.2
    assert status["eta_seconds"] == 10.0
    assert status["queue_wait_seconds"] == 5.0
    assert status["writer_wait_seconds"] == 5.0


def test_corrupt_rebuild_status_is_explicitly_recoverable(tmp_path: Path) -> None:
    rebuild_service.rebuild_status_path(tmp_path).write_text(
        "{not-json",
        encoding="utf-8",
    )

    status = rebuild_service.read_rebuild_status(tmp_path)

    assert status["status"] == "recoverable"
    assert status["phase"] == "recoverable"
    assert status["error"] == "rebuild_status_corrupt"


def test_run_rebuild_resumes_interrupted_document_batch(tmp_path: Path) -> None:
    config = _rebuild_config(tmp_path)
    documents_root = Path(config.indexing.documents_path)
    for name in ("one.md", "two.md", "three.md", "four.md"):
        (documents_root / name).write_text(name, encoding="utf-8")

    first = _RebuildManager(fail_document_batch=2)
    failed = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=first,
        request_id="request-1",
    )
    assert failed["status"] == "failed"
    assert first.clear_calls == 1

    resumed = _RebuildManager()
    succeeded = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=resumed,
        request_id="request-1",
    )
    assert succeeded["status"] == "succeeded"
    assert resumed.clear_calls == 0
    assert [len(batch) for batch in resumed.indexed_batches] == [2]
    assert succeeded["indexed_files"] == 4


def test_run_rebuild_reports_current_document_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _rebuild_config(tmp_path)
    documents_root = Path(config.indexing.documents_path)
    (documents_root / "one.md").write_text("one", encoding="utf-8")
    (documents_root / "two.md").write_text("two", encoding="utf-8")
    updates: list[dict[str, object]] = []
    original_update = rebuild_service._update_rebuild_progress

    def capture_update(runtime_root: Path, **changes: object) -> dict[str, object]:
        updates.append(changes)
        return original_update(runtime_root, **changes)

    monkeypatch.setattr(rebuild_service, "_update_rebuild_progress", capture_update)

    _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=_RebuildManager(),
        request_id="request-1",
    )

    assert any(
        update.get("current_document_path") == str(documents_root / "one.md")
        for update in updates
    )
    assert any(update.get("current_document_path") is None for update in updates)


def test_run_rebuild_invalidates_stale_document_checkpoint(tmp_path: Path) -> None:
    config = _rebuild_config(tmp_path)
    documents_root = Path(config.indexing.documents_path)
    (documents_root / "one.md").write_text("one", encoding="utf-8")
    (documents_root / "two.md").write_text("two", encoding="utf-8")

    first = _RebuildManager(fail_document_batch=1)
    _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=first,
        request_id="request-1",
    )
    (documents_root / "one.md").write_text("changed", encoding="utf-8")

    resumed = _RebuildManager()
    result = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=resumed,
        request_id="request-1",
    )
    assert result["status"] == "succeeded"
    assert resumed.clear_calls == 1
    assert [len(batch) for batch in resumed.indexed_batches] == [2]


def test_run_rebuild_skips_completed_stable_batches(tmp_path: Path) -> None:
    config = _rebuild_config(tmp_path)
    documents_root = Path(config.indexing.documents_path)
    (documents_root / "one.md").write_text("one", encoding="utf-8")
    (documents_root / "two.md").write_text("two", encoding="utf-8")

    first = _RebuildManager()
    assert _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=first,
        request_id="request-1",
    )["status"] == "succeeded"

    resumed = _RebuildManager()
    result = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=resumed,
        request_id="request-1",
    )
    assert result["status"] == "succeeded"
    assert resumed.clear_calls == 0
    assert resumed.indexed_batches == []
    assert result["indexed_files"] == 2


def test_run_rebuild_reuses_checkpoint_across_configured_path_changes(
    tmp_path: Path,
) -> None:
    """
    Given a daemon-owned runtime with a changed raw config path.
    When the same rebuild is resumed.
    Then completed work remains reusable because runtime identity is stable.
    """
    config = _rebuild_config(tmp_path)
    documents_root = Path(config.indexing.documents_path)
    (documents_root / "one.md").write_text("one", encoding="utf-8")
    runtime_root = tmp_path / "runtime-index"
    config.indexing.index_path = str(tmp_path / "legacy-index")

    first = _RebuildManager()
    assert _run_rebuild(
        tmp_path=runtime_root,
        config=config,
        manager=first,
        request_id="request-1",
    )["status"] == "succeeded"

    config.indexing.index_path = str(tmp_path / "new-configured-index")
    resumed = _RebuildManager()
    result = _run_rebuild(
        tmp_path=runtime_root,
        config=config,
        manager=resumed,
        request_id="request-1",
    )

    assert result["status"] == "succeeded"
    assert resumed.clear_calls == 0
    assert resumed.indexed_batches == []


def test_run_rebuild_fresh_request_resets_completed_corpus(tmp_path: Path) -> None:
    config = _rebuild_config(tmp_path)
    documents_root = Path(config.indexing.documents_path)
    (documents_root / "one.md").write_text("one", encoding="utf-8")

    first = _RebuildManager()
    assert _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=first,
        request_id="request-1",
    )["status"] == "succeeded"

    fresh = _RebuildManager()
    result = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=fresh,
        request_id="request-2",
    )
    assert result["status"] == "succeeded"
    assert fresh.clear_calls == 1
    assert fresh.indexed_batches == [[str(documents_root / "one.md")]]


def test_run_rebuild_resumes_interrupted_git_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _rebuild_config(tmp_path, git_enabled=True)
    repo_path = Path(config.indexing.documents_path) / "repo" / ".git"
    repo_path.mkdir(parents=True)
    source_records = [
        Record(
            source_kind="git_commit",
            source_id=f"git:{index}",
            title=f"Commit {index}",
            body="Body",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for index in range(27)
    ]

    class _Source:
        def __init__(self, _repo_path: Path) -> None:
            self._repo_path = _repo_path

        def iter_records(self):
            yield from source_records

    monkeypatch.setattr(rebuild_service, "GitContentSource", _Source)
    monkeypatch.setattr(rebuild_service, "is_git_available", lambda: True)
    monkeypatch.setattr(
        rebuild_service,
        "_discover_scope_git_repositories",
        lambda _config, _roots: [repo_path],
    )
    monkeypatch.setattr(
        rebuild_service,
        "get_git_ref_signature",
        lambda _repo: "stable-ref",
    )

    first = _RebuildManager()
    original_index_record = first.index_record
    calls = 0

    def fail_second_batch(record: Record) -> bool:
        nonlocal calls
        calls += 1
        if calls == 26:
            raise RuntimeError("interrupted git batch")
        return original_index_record(record)

    first.index_record = fail_second_batch
    failed = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=first,
        request_id="request-1",
    )
    assert failed["status"] == "failed"
    assert len(first.indexed_records) == 25

    resumed = _RebuildManager()
    succeeded = _run_rebuild(
        tmp_path=tmp_path,
        config=config,
        manager=resumed,
        request_id="request-1",
    )
    assert succeeded["status"] == "succeeded"
    assert len(resumed.indexed_records) == 2
    assert succeeded["git_commits_indexed"] == 27
