import types
from datetime import UTC, datetime

from searchkernel.domain import Record, RecordIdentity, RecordStatus

from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from mcp_markdown_ragdocs.indexing.record_ports import LocalRecordStorage
from tests.conftest import make_test_config


def _make_commit_record(source_id: str, body: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="git_commit",
        source_id=source_id,
        title="Fix login bug",
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"author": "Test User", "files_changed": ["api.py"]},
        status=RecordStatus.ACTIVE,
    )


def test_iter_records_streams_lazily(record_manager) -> None:
    result = record_manager.storage.iter_records()

    assert isinstance(result, types.GeneratorType)


def test_iter_records_yields_every_stored_record(record_manager) -> None:
    records = [_make_commit_record(f"commit-{i}", f"message {i}") for i in range(5)]
    for record in records:
        assert record_manager.index_record(record) is True

    yielded_keys = {record.storage_key for record in record_manager.storage.iter_records()}

    assert yielded_keys == {record.storage_key for record in records}


def test_iter_records_hydrates_in_bounded_batches(record_manager, monkeypatch) -> None:
    """Record hydration consumes identities in fixed-size batches.

    This keeps large local indexes from being materialized during iteration.
    """
    records = [_make_commit_record(f"commit-{i}", f"message {i}") for i in range(5)]
    for record in records:
        assert record_manager.index_record(record) is True

    storage = record_manager.storage
    monkeypatch.setattr(storage.__class__, "_ITER_RECORDS_BATCH_SIZE", 2)
    original_hydrate_records = storage.hydrate_records
    batch_sizes: list[int] = []

    def _tracking_hydrate_records(identities):
        batch_sizes.append(len(identities))
        return original_hydrate_records(identities)

    monkeypatch.setattr(storage, "hydrate_records", _tracking_hydrate_records)

    consumed = list(storage.iter_records())

    assert len(consumed) == len(records)
    assert batch_sizes == [2, 2, 1]


def test_iter_identities_filters_by_source_kind(record_manager) -> None:
    commit = _make_commit_record("commit-1", "message")
    note = Record(
        source_kind="note",
        source_id="note-1",
        title="Note",
        body="body",
        created_at=commit.created_at,
        updated_at=commit.updated_at,
        metadata={},
        status=RecordStatus.ACTIVE,
    )
    assert record_manager.index_record(commit) is True
    assert record_manager.index_record(note) is True

    identities = list(record_manager.storage.iter_identities(source_kind="git_commit"))

    assert [identity.source_id for identity in identities] == ["commit-1"]


def test_iter_identities_filters_by_status(record_manager) -> None:
    active = _make_commit_record("commit-active", "message")
    stale = Record(
        source_kind="git_commit",
        source_id="commit-stale",
        title="Fix login bug",
        body="message",
        created_at=active.created_at,
        updated_at=active.updated_at,
        metadata={},
        status=RecordStatus.STALE,
    )
    assert record_manager.index_record(active) is True
    assert record_manager.index_record(stale) is True

    identities = list(record_manager.storage.iter_identities(status="active"))

    assert [identity.source_id for identity in identities] == ["commit-active"]


def test_reconcile_git_project_attribution_only_hydrates_matches(
    local_record_kernel, deterministic_embedding_provider, tmp_path, monkeypatch
) -> None:
    """Reconciliation must not hydrate records outside the matched commit set.

    Regression guard: this used to hydrate every record in the index to
    find the handful belonging to one repository's commits.
    """
    config = make_test_config(tmp_path)
    manager = RecordIndexManager(
        config,
        local_record_kernel,
        deterministic_embedding_provider,
        commit_history=lambda git_dir, after_timestamp=None: iter(["abc"]),
    )

    target_commit = _make_commit_record("git:abc:summary:0", "target commit")
    other_repo_commit = _make_commit_record("git:def:summary:0", "other repo commit")
    note = Record(
        source_kind="note",
        source_id="note-1",
        title="Note",
        body="body",
        created_at=target_commit.created_at,
        updated_at=target_commit.updated_at,
        metadata={},
        status=RecordStatus.ACTIVE,
    )
    for record in (target_commit, other_repo_commit, note):
        assert manager.index_record(record) is True

    original_hydrate_records = manager.storage.hydrate_records
    hydrated_batch_sizes: list[int] = []

    def _tracking_hydrate_records(identities):
        hydrated_batch_sizes.append(len(identities))
        return original_hydrate_records(identities)

    monkeypatch.setattr(manager.storage, "hydrate_records", _tracking_hydrate_records)

    repaired = manager.reconcile_git_project_attribution(tmp_path / ".git", "target-project")

    assert repaired == 1
    assert hydrated_batch_sizes == [1]


def test_storage_delegates_identity_enumeration_to_catalog(local_record_kernel) -> None:
    """Storage delegates identity enumeration to the injected application port.

    The storage adapter must not need to know how the catalog persists keys.
    """
    expected = [
        RecordIdentity(None, "git_commit", "one"),
        RecordIdentity(None, "git_commit", "two"),
    ]

    class Catalog:
        def iter_identities(self, *, source_kind=None, status=None):
            return iter(expected)

    storage = LocalRecordStorage(local_record_kernel, identity_catalog=Catalog())

    assert list(storage.iter_identities()) == expected
