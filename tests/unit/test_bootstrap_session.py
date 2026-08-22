import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from searchkernel.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    compute_bootstrap_generation,
    save_bootstrap_checkpoint,
)
from searchkernel.indexing.manifest import (
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    save_manifest,
)
from searchkernel.indexing.bootstrap_snapshot import compute_bootstrap_completed_paths

from mcp_markdown_ragdocs.indexing.bootstrap_session import BootstrapSession
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort
from mcp_markdown_ragdocs.indexing.tasks import TaskBatchSubmissionResult


def _manifest() -> IndexManifest:
    return IndexManifest(
        spec_version=CURRENT_MANIFEST_SPEC_VERSION,
        embedding_model="local",
        chunking_config={},
        indexed_files={},
    )


def _stamp(path: Path, relative_path: str | None = None) -> BootstrapFileStamp:
    stat_result = path.stat()
    return BootstrapFileStamp(
        relative_path or path.name,
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
    )


@pytest.mark.asyncio
async def test_run_reuses_completion_for_multiple_document_roots(
    tmp_path: Path,
) -> None:
    """Checkpoint paths may be common-root-relative while manifests are root-relative."""
    root_one = tmp_path / "docs-one"
    root_two = tmp_path / "docs-two"
    root_one.mkdir()
    root_two.mkdir()
    doc_one = root_one / "one.md"
    doc_two = root_two / "two.md"
    doc_one.write_text("# One")
    doc_two.write_text("# Two")

    manifest = _manifest()
    manifest.indexed_files = {"one": "one.md", "two": "two.md"}
    save_manifest(tmp_path, manifest)
    target_stamps = {
        "docs-one/one.md": _stamp(doc_one, "docs-one/one.md"),
        "docs-two/two.md": _stamp(doc_two, "docs-two/two.md"),
    }
    save_bootstrap_checkpoint(
        tmp_path,
        BootstrapCheckpoint(
            schema_version="1.0.0",
            generation=compute_bootstrap_generation(manifest, target_stamps),
            complete=True,
            targets=target_stamps,
            completed=dict(target_stamps),
        ),
    )

    submission = MagicMock(spec=TaskSubmissionPort)
    submission.submit_index_batch.side_effect = lambda file_paths, **kwargs: pytest.fail(
        "completed multi-root files should not be enqueued"
    )
    ready_calls: list[str] = []
    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[root_one, root_two],
        git_refresh_enabled=False,
        discover_files=lambda: [str(doc_one), str(doc_two)],
        discover_git_repositories=list,
        get_bootstrap_manifest=lambda: manifest,
        load_persisted_indices=lambda: asyncio.sleep(0),
        persist_indices=lambda: asyncio.sleep(0),
        compute_index_state_version=lambda: asyncio.sleep(0, result=1.0),
        get_loaded_index_state_version=lambda: 0.0,
        get_loaded_document_count=lambda: 2,
        is_queryable=lambda: True,
        publish_public_state=lambda snapshot: None,
        mark_ready=lambda: ready_calls.append("called"),
        schedule_embedding_warmup=lambda: True,
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
        task_submission=submission,
    )
    await session.run()

    submission.submit_index_batch.assert_not_called()
    assert ready_calls == ["called"]


def test_multi_root_same_relative_name_only_matches_indexed_root(tmp_path: Path) -> None:
    root_one = tmp_path / "docs-one"
    root_two = tmp_path / "docs-two"
    root_one.mkdir()
    root_two.mkdir()
    doc_one = root_one / "same.md"
    doc_two = root_two / "same.md"
    doc_one.write_text("# One")
    doc_two.write_text("# Two")

    manifest = _manifest()
    manifest.indexed_files = {"docs-one/same": "same.md"}
    targets = {
        "docs-one/same.md": _stamp(doc_one, "docs-one/same.md"),
        "docs-two/same.md": _stamp(doc_two, "docs-two/same.md"),
    }
    checkpoint = BootstrapCheckpoint(
        schema_version="1.0.0",
        generation=compute_bootstrap_generation(manifest, targets),
        complete=True,
        targets=targets,
        completed=dict(targets),
    )
    session = BootstrapSession.__new__(BootstrapSession)
    session.documents_roots = [root_one, root_two]

    normalized = session._manifest_for_bootstrap(manifest, targets)
    completed = compute_bootstrap_completed_paths(checkpoint, normalized, targets)

    assert completed == {"docs-one/same.md"}


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
        discover_git_repositories=list,
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
        discover_git_repositories=list,
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

    submission = MagicMock(spec=TaskSubmissionPort)
    def submit_index_batch(
        file_paths: list[str], **kwargs: object
    ) -> TaskBatchSubmissionResult:
        enqueue_checked.set()
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(set(file_paths)),
            enqueued_count=0,
            already_pending_count=len(set(file_paths)),
        )
    submission.submit_index_batch.side_effect = submit_index_batch

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
        discover_git_repositories=list,
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
        report_failure=lambda error, indexed_count, total_count: failures.append(
            (str(error), indexed_count, total_count)
        ),
        task_submission=submission,
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

    submission = MagicMock(spec=TaskSubmissionPort)
    submission.submit_index_batch.side_effect = fake_submit_index_batch

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
        discover_git_repositories=list,
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
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
        task_submission=submission,
    )

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert enqueued_batches == [[str(doc_two)]]
    assert ready_calls != []
    assert warmup_calls != []
    assert published_states[-1].status == "ready"
    assert published_states[-1].indexed_count == 2
    assert published_states[-1].total_count == 2


@pytest.mark.asyncio
async def test_run_stops_monitoring_once_queryable_even_if_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given remaining startup work that never fully completes.
    When the bootstrap session becomes queryable partway through.
    Then it should stop monitoring instead of polling forever.
    """

    doc_one = tmp_path / "doc1.md"
    doc_two = tmp_path / "doc2.md"
    doc_one.write_text("# Doc 1")
    doc_two.write_text("# Doc 2")

    submission = MagicMock(spec=TaskSubmissionPort)
    submission.submit_index_batch.side_effect = lambda file_paths, **kwargs: TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=len(set(file_paths)),
        enqueued_count=0,
        already_pending_count=len(set(file_paths)),
    )

    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    load_calls = 0

    async def load_persisted_indices() -> None:
        nonlocal load_calls
        load_calls += 1

    async def compute_index_state_version() -> float:
        return 1.0

    ready_calls: list[str] = []
    warmup_calls: list[str] = []

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=False,
        discover_files=lambda: [str(doc_one), str(doc_two)],
        discover_git_repositories=list,
        get_bootstrap_manifest=_manifest,
        load_persisted_indices=load_persisted_indices,
        persist_indices=lambda: asyncio.sleep(0),
        compute_index_state_version=compute_index_state_version,
        get_loaded_index_state_version=lambda: 0.0,
        get_loaded_document_count=lambda: 0,
        is_queryable=lambda: load_calls > 0,
        publish_public_state=lambda snapshot: None,
        mark_ready=lambda: ready_calls.append("called"),
        schedule_embedding_warmup=lambda: warmup_calls.append("called") or True,
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
        task_submission=submission,
    )

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert ready_calls == ["called"]
    assert warmup_calls == ["called"]


@pytest.mark.asyncio
async def test_run_enqueues_startup_git_refresh_batch_in_task_mode(
    tmp_path: Path,
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

    submission = MagicMock(spec=TaskSubmissionPort)
    def submit_refresh_git_batch(git_dirs: list[str]) -> TaskBatchSubmissionResult:
        git_submissions.append(git_dirs)
        return TaskBatchSubmissionResult(
            queue_available=True,
            requested_unique_count=len(set(git_dirs)),
            enqueued_count=len(set(git_dirs)),
        )
    submission.submit_refresh_git_batch.side_effect = submit_refresh_git_batch

    session = BootstrapSession(
        index_path=tmp_path,
        documents_roots=[tmp_path],
        git_refresh_enabled=True,
        discover_files=list,
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
        report_failure=lambda error, indexed_count, total_count: pytest.fail(
            f"unexpected failure: {error}"
        ),
        task_submission=submission,
    )

    await asyncio.wait_for(session.run(), timeout=1.0)

    assert git_submissions == [[str(git_repo)]]
    assert persisted == ["called"]
    assert ready_calls == ["called"]
    assert warmup_calls == ["called"]
