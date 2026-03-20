import asyncio
from pathlib import Path

import pytest

from src.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    compute_bootstrap_generation,
    save_bootstrap_checkpoint,
)
from src.indexing.bootstrap_session import BootstrapSession
from src.indexing.manifest import (
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    save_manifest,
)
from src.indexing.tasks import TaskBatchSubmissionResult


def _manifest() -> IndexManifest:
    return IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model="local",
        chunking_config={},
        indexed_files={},
    )


def _stamp(path: Path) -> BootstrapFileStamp:
    stat_result = path.stat()
    return BootstrapFileStamp(
        path.name,
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
    )


@pytest.mark.asyncio
async def test_preload_persisted_state_marks_ready_from_partial_snapshot(
    tmp_path: Path,
) -> None:
    """
    Given a persisted partial bootstrap snapshot.
    When the bootstrap session preloads existing state.
    Then it should publish partial readiness immediately and mark queries ready.
    """

    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    manifest = _manifest()
    manifest.indexed_files = {"doc1": "doc1.md"}
    save_manifest(tmp_path, manifest)

    checkpoint_targets = {
        "doc1.md": _stamp(doc_one),
        "doc2.md": _stamp(doc_two),
    }
    save_bootstrap_checkpoint(
        tmp_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation=compute_bootstrap_generation(manifest, checkpoint_targets),
            complete=False,
            targets=checkpoint_targets,
            completed={"doc1.md": checkpoint_targets["doc1.md"]},
        ),
    )

    published_states = []
    ready_calls: list[str] = []
    warmup_calls: list[str] = []
    load_calls: list[str] = []

    async def load_persisted_indices() -> None:
        load_calls.append("called")

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=False,
        discover_files=lambda: [str(doc_one), str(doc_two)],
        discover_git_repositories=lambda: [],
        get_bootstrap_manifest=_manifest,
        load_persisted_indices=load_persisted_indices,
        persist_indices=lambda: asyncio.sleep(0),
        compute_index_state_version=lambda: asyncio.sleep(0, result=1.0),
        get_loaded_index_state_version=lambda: 0.0,
        get_loaded_document_count=lambda: 1,
        is_queryable=lambda: True,
        publish_public_state=published_states.append,
        mark_ready=lambda: ready_calls.append("called"),
        schedule_embedding_warmup=lambda: warmup_calls.append("called") or True,
        schedule_vocabulary_catch_up=lambda: False,
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
    )

    preloaded = await session.preload_persisted_state(rebuild_pending=False)

    assert preloaded is True
    assert load_calls == ["called"]
    assert ready_calls == ["called"]
    assert warmup_calls == ["called"]
    assert published_states[-1].status == "partial"
    assert published_states[-1].indexed_count == 1
    assert published_states[-1].total_count == 2


@pytest.mark.asyncio
async def test_preload_persisted_state_does_not_mark_ready_when_not_queryable(
    tmp_path: Path,
) -> None:
    """
    Given a persisted partial bootstrap snapshot that is not yet queryable.
    When the bootstrap session preloads existing state.
    Then it should publish partial state without marking the runtime ready.
    """

    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    manifest = _manifest()
    manifest.indexed_files = {"doc1": "doc1.md"}
    save_manifest(tmp_path, manifest)

    checkpoint_targets = {
        "doc1.md": _stamp(doc_one),
        "doc2.md": _stamp(doc_two),
    }
    save_bootstrap_checkpoint(
        tmp_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation=compute_bootstrap_generation(manifest, checkpoint_targets),
            complete=False,
            targets=checkpoint_targets,
            completed={"doc1.md": checkpoint_targets["doc1.md"]},
        ),
    )

    published_states = []
    ready_calls: list[str] = []
    warmup_calls: list[str] = []

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=False,
        discover_files=lambda: [str(doc_one), str(doc_two)],
        discover_git_repositories=lambda: [],
        get_bootstrap_manifest=_manifest,
        load_persisted_indices=lambda: asyncio.sleep(0),
        persist_indices=lambda: asyncio.sleep(0),
        compute_index_state_version=lambda: asyncio.sleep(0, result=1.0),
        get_loaded_index_state_version=lambda: 0.0,
        get_loaded_document_count=lambda: 1,
        is_queryable=lambda: False,
        publish_public_state=published_states.append,
        mark_ready=lambda: ready_calls.append("called"),
        schedule_embedding_warmup=lambda: warmup_calls.append("called") or True,
        schedule_vocabulary_catch_up=lambda: False,
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
    )

    preloaded = await session.preload_persisted_state(rebuild_pending=False)

    assert preloaded is True
    assert ready_calls == []
    assert warmup_calls == ["called"]
    assert published_states[-1].status == "partial"
    assert published_states[-1].indexed_count == 1
    assert published_states[-1].total_count == 2


@pytest.mark.asyncio
async def test_run_keeps_monitoring_when_remaining_work_is_already_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given remaining startup work already represented in the queue.
    When the bootstrap session runs.
    Then it should keep monitoring persisted progress without failing startup.
    """

    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    published_states = []
    failures: list[tuple[str, int, int]] = []
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

    async def compute_index_state_version() -> float:
        return 0.0

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=False,
        discover_files=lambda: [str(doc_one), str(doc_two)],
        discover_git_repositories=lambda: [],
        get_bootstrap_manifest=_manifest,
        load_persisted_indices=lambda: asyncio.sleep(0),
        persist_indices=lambda: asyncio.sleep(0),
        compute_index_state_version=compute_index_state_version,
        get_loaded_index_state_version=lambda: 0.0,
        get_loaded_document_count=lambda: 0,
        is_queryable=lambda: False,
        publish_public_state=published_states.append,
        mark_ready=lambda: pytest.fail("session should not mark ready yet"),
        schedule_embedding_warmup=lambda: pytest.fail("warmup should not run yet"),
        schedule_vocabulary_catch_up=lambda: pytest.fail(
            "vocabulary catch-up should not run yet"
        ),
        report_failure=lambda error, indexed_count, total_count: failures.append(
            (str(error), indexed_count, total_count)
        ),
    )

    bootstrap_task = asyncio.create_task(session.run())
    await asyncio.wait_for(enqueue_checked.wait(), timeout=1.0)

    assert published_states[-1].status == "indexing"
    assert published_states[-1].indexed_count == 0
    assert published_states[-1].total_count == 2
    assert failures == []

    bootstrap_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bootstrap_task


@pytest.mark.asyncio
async def test_run_skips_completed_files_and_finishes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given one file already durably completed in the checkpoint.
    When the bootstrap session runs.
    Then it should enqueue only the remaining file and finish ready after persisted progress advances.
    """

    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    manifest = _manifest()
    manifest.indexed_files = {"doc1": "doc1.md"}
    save_manifest(tmp_path, manifest)

    checkpoint_targets = {
        "doc1.md": _stamp(doc_one),
        "doc2.md": _stamp(doc_two),
    }
    save_bootstrap_checkpoint(
        tmp_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation=compute_bootstrap_generation(manifest, checkpoint_targets),
            complete=False,
            targets=checkpoint_targets,
            completed={"doc1.md": checkpoint_targets["doc1.md"]},
        ),
    )

    published_states = []
    ready_calls: list[str] = []
    warmup_calls: list[str] = []
    vocabulary_calls: list[str] = []
    enqueued_batches: list[list[str]] = []
    load_calls = 0
    completion_written = False
    loaded_version = 0.0
    loaded_document_count = 1

    async def load_persisted_indices() -> None:
        nonlocal load_calls, loaded_version
        load_calls += 1
        loaded_version = 2.0 if completion_written else 1.0

    def current_version() -> float:
        return 2.0 if completion_written else 1.0

    def fake_submit_index_batch(
        file_paths: list[str],
        force: bool = False,
    ) -> TaskBatchSubmissionResult:
        nonlocal completion_written, loaded_document_count
        assert force is False
        enqueued_batches.append(file_paths)
        updated_manifest = _manifest()
        updated_manifest.indexed_files = {
            "doc1": "doc1.md",
            "doc2": "doc2.md",
        }
        save_manifest(tmp_path, updated_manifest)
        completion_written = True
        loaded_document_count = 2
        save_bootstrap_checkpoint(
            tmp_path,
            BootstrapCheckpoint(
                schema_version="1.0.0",
                generation=compute_bootstrap_generation(
                    updated_manifest,
                    checkpoint_targets,
                ),
                complete=True,
                targets=checkpoint_targets,
                completed=dict(checkpoint_targets),
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

    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    async def compute_index_state_version() -> float:
        return current_version()

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=False,
        discover_files=lambda: [str(doc_one), str(doc_two)],
        discover_git_repositories=lambda: [],
        get_bootstrap_manifest=_manifest,
        load_persisted_indices=load_persisted_indices,
        persist_indices=lambda: asyncio.sleep(0),
        compute_index_state_version=compute_index_state_version,
        get_loaded_index_state_version=lambda: loaded_version,
        get_loaded_document_count=lambda: loaded_document_count,
        is_queryable=lambda: load_calls > 0,
        publish_public_state=published_states.append,
        mark_ready=lambda: ready_calls.append("called"),
        schedule_embedding_warmup=lambda: warmup_calls.append("called") or True,
        schedule_vocabulary_catch_up=lambda: vocabulary_calls.append("called") or True,
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
    )

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert enqueued_batches == [[str(doc_two)]]
    assert ready_calls != []
    assert warmup_calls != []
    assert vocabulary_calls == ["called"]
    assert published_states[-1].status == "ready"
    assert published_states[-1].indexed_count == 2
    assert published_states[-1].total_count == 2


@pytest.mark.asyncio
async def test_run_enqueues_startup_git_refresh_batch_in_task_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given task-backed startup bootstrap with git refresh enabled.
    When the session runs with no document work.
    Then it should still enqueue the startup git refresh batch.
    """

    git_repo = tmp_path / ".git"
    git_repo.mkdir()

    git_submissions: list[list[str]] = []
    persisted: list[str] = []
    ready_calls: list[str] = []
    warmup_calls: list[str] = []

    monkeypatch.setattr(
        "src.indexing.tasks.submit_refresh_git_batch",
        lambda git_dirs: git_submissions.append(git_dirs)
        or TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(set(git_dirs)),
            enqueued_count=len(set(git_dirs)),
        ),
    )

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=True,
        discover_files=lambda: [],
        discover_git_repositories=lambda: [git_repo],
        get_bootstrap_manifest=_manifest,
        load_persisted_indices=lambda: asyncio.sleep(0),
        persist_indices=lambda: persisted.append("called") or asyncio.sleep(0),
        compute_index_state_version=lambda: asyncio.sleep(0, result=1.0),
        get_loaded_index_state_version=lambda: 0.0,
        get_loaded_document_count=lambda: 0,
        is_queryable=lambda: True,
        publish_public_state=lambda snapshot: None,
        mark_ready=lambda: ready_calls.append("called"),
        schedule_embedding_warmup=lambda: warmup_calls.append("called") or True,
        schedule_vocabulary_catch_up=lambda: False,
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
    )

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert git_submissions == [[str(git_repo)]]
    assert persisted == ["called"]
    assert ready_calls == ["called"]
    assert warmup_calls == ["called"]