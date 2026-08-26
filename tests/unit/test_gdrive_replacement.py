"""Tests for idempotent Google Drive record replacement."""

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.api import (
    GraphEdge,
    Record,
    RecordStatus,
    build_local_record_kernel,
)

from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementEntry,
    GDriveReplacementJournal,
    canonical_gdrive_source_key,
)
from mcp_markdown_ragdocs.gdrive.replacement_policy import GDriveReplacementPolicy
from mcp_markdown_ragdocs.gdrive.adapter import GDriveStateRepository
from mcp_markdown_ragdocs.gdrive.domain import GDriveScopeIdentity
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from mcp_markdown_ragdocs.indexing.record_ports import RecordStorage, SqliteSourceMapStore


class _CountingRecordStorage:
    """Spy that delegates to a real RecordStorage while counting hydration calls."""

    def __init__(self, storage: RecordStorage) -> None:
        self._storage = storage
        self.hydrate_record_calls = 0
        self.hydrate_records_calls = 0

    def register_content_source(self, source):
        self._storage.register_content_source(source)

    def hydrate_record(self, identity):
        self.hydrate_record_calls += 1
        return self._storage.hydrate_record(identity)

    def hydrate_records(self, identities):
        self.hydrate_records_calls += 1
        return self._storage.hydrate_records(identities)

    def iter_records(self, *, source_kind=None, status=None):
        return self._storage.iter_records(source_kind=source_kind, status=status)

    def iter_identities(self, *, source_kind=None, status=None):
        return self._storage.iter_identities(source_kind=source_kind, status=status)

    def count_distinct_git_commits(self, *, status=None):
        return self._storage.count_distinct_git_commits(status=status)

    def run_incremental_vacuum(self, page_limit):
        return self._storage.run_incremental_vacuum(page_limit)

    def delete(self, storage_keys):
        self._storage.delete(storage_keys)


class _SingleConnectionProvider:
    """Minimal real-connection SQLiteConnectionProvider for tests."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(str(path))

    def get_connection(self) -> sqlite3.Connection:
        return self._connection


class _DeleteSpyStorage:
    """Fake RecordStorage that only records delete() calls.

    _delete_stale_keys only ever calls storage.delete(); every other method
    raises so a test fails loudly if the method under test starts depending
    on more than that.
    """

    def __init__(self) -> None:
        self.deleted: list[tuple[str, ...]] = []

    def register_content_source(self, source) -> None:
        raise NotImplementedError

    def hydrate_record(self, identity):
        raise NotImplementedError

    def hydrate_records(self, identities):
        raise NotImplementedError

    def iter_records(self, *, source_kind=None, status=None):
        raise NotImplementedError

    def iter_identities(self, *, source_kind=None, status=None):
        raise NotImplementedError

    def count_distinct_git_commits(self, *, status=None) -> int:
        raise NotImplementedError

    def run_incremental_vacuum(self, page_limit: int) -> int:
        raise NotImplementedError

    def delete(self, storage_keys) -> None:
        self.deleted.append(tuple(storage_keys))


class _SourceMapWriteSpy(SqliteSourceMapStore):
    """Wrap a real SqliteSourceMapStore, recording which write path is used."""

    def __init__(self, connection_provider) -> None:
        super().__init__(connection_provider)
        self.save_calls: list[dict[str, list[str]]] = []
        self.apply_delta_calls: list[tuple[dict[str, list[str]], list[str]]] = []

    def save(self, records) -> None:
        self.save_calls.append({doc_id: list(keys) for doc_id, keys in records.items()})
        super().save(records)

    def apply_delta(self, upserts, removals) -> None:
        self.apply_delta_calls.append(
            ({doc_id: list(keys) for doc_id, keys in upserts.items()}, list(removals))
        )
        super().apply_delta(upserts, removals)


def _policy_for_delete_stale_keys(
    tmp_path: Path,
    record_manager,
    source_records: dict[str, list[str]],
    storage=None,
) -> GDriveReplacementPolicy:
    """Build a policy whose only exercised collaborator is storage.delete()."""
    source_map = SqliteSourceMapStore(_SingleConnectionProvider(tmp_path / "index.db"))
    return GDriveReplacementPolicy(
        record_manager.ingestor,
        storage if storage is not None else _DeleteSpyStorage(),
        source_records,
        source_map,
        GDriveReplacementJournal(tmp_path / "gdrive-replacements.json"),
    )


def _record(
    source_id: str,
    body: str,
    *,
    deleted: bool = False,
    scopes: tuple[str, ...] = ("shared-with-me",),
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="gdrive",
        source_id=source_id,
        workspace_id="workspace",
        title="Drive note",
        body=body if not deleted else "",
        indexed_text=body if not deleted else "",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "gdrive_source_id": "file-1",
            "scope_memberships": list(scopes),
            "deleted": deleted,
            "extraction_status": "tombstone" if deleted else "indexed",
        },
        status=RecordStatus.ARCHIVED if deleted else RecordStatus.ACTIVE,
    )


def test_drive_replacement_deletes_stale_chunks_and_deduplicates_source_map(
    record_manager,
) -> None:
    """
    Replace a Drive source with fewer chunks without retaining stale records.
    """
    first = (_record("file-1:chunk-a", "old A"), _record("file-1:chunk-b", "old B"))
    replacement = (_record("file-1:chunk-a", "new A"),)

    assert record_manager.index_records(first) is True
    assert record_manager.index_records(replacement) is True

    assert record_manager.storage.hydrate_record(first[1].storage_key) is None
    assert record_manager.storage.hydrate_record(replacement[0].storage_key) is not None
    source_key = canonical_gdrive_source_key(replacement[0])
    source_map = record_manager._source_map_store.load()
    assert source_map[source_key] == [replacement[0].storage_key]


def test_drive_replacement_removes_stale_records_from_every_search_surface(
    record_manager,
    monkeypatch,
) -> None:
    """
    Remove stale Drive records from canonical, keyword, vector, and graph data.
    """
    stale = _record("file-1:chunk-b", "stale body")
    retained = _record("file-1:chunk-a", "retained body")
    replacement = _record("file-1:chunk-a", "replacement body")

    assert record_manager.index_records((stale, retained)) is True
    record_manager.graph.upsert_edges(
        [GraphEdge(stale.identity, retained.identity, "related_to", 1.0)]
    )
    keyword_before = record_manager.keyword.search("stale body", 10)
    vector_before = record_manager.embedding_provider.embed([stale.body])[0]
    vector_hits_before = record_manager.vector.search(
        vector_before,
        10,
        model_name=record_manager.embedding_provider.model_name,
        dim=record_manager.embedding_provider.dim,
    )
    assert any(hit.identity == stale.identity for hit in keyword_before)
    assert any(hit.identity == stale.identity for hit in vector_hits_before)
    assert any(
        neighbor.identity == retained.identity
        for neighbor in record_manager.graph.neighbors(stale.identity)
    )

    monkeypatch.setattr(record_manager.vector, "delete", lambda _keys: None)
    assert record_manager.index_records((replacement,)) is True

    assert record_manager.storage.hydrate_record(stale.storage_key) is None
    assert all(hit.identity != stale.identity for hit in record_manager.keyword.search("stale body", 10))
    vector_hits_after = record_manager.vector.search(
        vector_before,
        10,
        model_name=record_manager.embedding_provider.model_name,
        dim=record_manager.embedding_provider.dim,
    )
    assert all(hit.identity != stale.identity for hit in vector_hits_after)
    assert record_manager.graph.neighbors(stale.identity) == []
    assert all(
        neighbor.identity != stale.identity
        for neighbor in record_manager.graph.incoming_neighbors(retained.identity)
    )


def test_drive_replay_is_idempotent_for_records_and_memberships(record_manager) -> None:
    """
    Replay the same Drive batch without duplicating index or source-map entries.
    """
    record = _record("file-1:chunk-a", "stable body")

    assert record_manager.index_records((record, record)) is True
    assert record_manager.index_records((record, record)) is True

    source_key = canonical_gdrive_source_key(record)
    source_map = record_manager._source_map_store.load()
    assert source_map[source_key] == [record.storage_key]
    assert record_manager.count_records("gdrive") == 1


def test_drive_batch_indexes_multiple_sources_in_one_ingestor_call(
    record_manager,
    monkeypatch,
) -> None:
    """
    Combine independent Drive source replacements into one write pipeline pass.
    """
    first = _record("file-1:chunk-a", "first body")
    second = _record("file-2:chunk-a", "second body")
    original_index_records = record_manager.ingestor.index_records
    calls: list[tuple[Record, ...]] = []

    async def tracked_index_records(records):
        calls.append(tuple(records))
        return await original_index_records(records)

    monkeypatch.setattr(record_manager.ingestor, "index_records", tracked_index_records)

    assert record_manager.index_records((first, second)) is True

    assert [[record.storage_key for record in batch] for batch in calls] == [
        [first.storage_key, second.storage_key]
    ]
    assert record_manager.count_records("gdrive") == 2


def test_drive_tombstone_removes_the_source_from_search(record_manager) -> None:
    """
    Treat a confirmed Drive tombstone as deletion rather than searchable content.
    """
    active = _record("file-1:chunk-a", "searchable body")
    tombstone = _record("file-1", "ignored", deleted=True)

    assert record_manager.index_record(active) is True
    assert record_manager.index_record(tombstone) is True

    assert record_manager.storage.hydrate_record(active.storage_key) is None
    assert record_manager.count_records("gdrive") == 0


def test_drive_tombstone_keeps_a_source_visible_in_remaining_scope(record_manager) -> None:
    """
    Remove one Drive scope membership without deleting a still-visible source.
    """
    active = _record("file-1:chunk-a", "shared body", scopes=("drive-a", "drive-b"))
    tombstone = _record("file-1", "ignored", deleted=True, scopes=("drive-a",))

    assert record_manager.index_record(active) is True
    assert record_manager.index_record(tombstone) is True

    hydrated = record_manager.storage.hydrate_record(active.storage_key)
    assert hydrated is not None
    assert hydrated.metadata["scope_memberships"] == ["drive-b"]


def test_indexed_drive_replacement_recovers_after_manager_restart(
    record_manager,
    deterministic_embedding_provider,
) -> None:
    """
    Finish stale cleanup from an indexed journal after a simulated crash.
    """
    old = _record("file-1:chunk-a", "old body")
    new = _record("file-1:chunk-b", "new body")
    assert record_manager.index_record(old) is True

    asyncio.run(record_manager.ingestor.index_records((new,)))
    journal = GDriveReplacementJournal(
        Path(record_manager.index_path) / "gdrive-replacements.json"
    )
    entry = journal.prepare(
        canonical_gdrive_source_key(new),
        (old.storage_key,),
        (new.storage_key,),
    )
    journal.mark_indexed(entry)

    restarted_kernel = build_local_record_kernel(
        Path(record_manager.index_path).parent / "records.db",
        embedding_provider=deterministic_embedding_provider,
        embedding_model_name=deterministic_embedding_provider.model_name,
        embedding_dim=deterministic_embedding_provider.dim,
        vector_engine="exact",
    )
    restarted = RecordIndexManager(
        record_manager._config,
        restarted_kernel,
        deterministic_embedding_provider,
    )

    assert restarted.storage.hydrate_record(old.storage_key) is None
    assert restarted.storage.hydrate_record(new.storage_key) is not None
    assert GDriveReplacementJournal(
        Path(record_manager.index_path) / "gdrive-replacements.json"
    ).load() == ()


def test_prepared_drive_replacement_discards_partial_index_write_after_restart(
    record_manager,
    deterministic_embedding_provider,
) -> None:
    """
    Remove an uncommitted replacement write while retaining the prior source.
    """
    old = _record("file-1:chunk-a", "old body")
    new = _record("file-1:chunk-b", "partial body")
    assert record_manager.index_record(old) is True

    asyncio.run(record_manager.ingestor.index_records((new,)))
    journal = GDriveReplacementJournal(
        Path(record_manager.index_path) / "gdrive-replacements.json"
    )
    journal.prepare(
        canonical_gdrive_source_key(new),
        (old.storage_key,),
        (new.storage_key,),
    )

    restarted_kernel = build_local_record_kernel(
        Path(record_manager.index_path).parent / "records.db",
        embedding_provider=deterministic_embedding_provider,
        embedding_model_name=deterministic_embedding_provider.model_name,
        embedding_dim=deterministic_embedding_provider.dim,
        vector_engine="exact",
    )
    restarted = RecordIndexManager(
        record_manager._config,
        restarted_kernel,
        deterministic_embedding_provider,
    )

    assert restarted.storage.hydrate_record(old.storage_key) is not None
    assert restarted.storage.hydrate_record(new.storage_key) is None
    assert journal.load() == ()


def test_replacement_journal_persists_drive_state_status(tmp_path: Path) -> None:
    """
    Mirror replacement phases in the committed Drive state repository.
    """
    identity = GDriveScopeIdentity("gdrive", "workspace", "shared-with-me")
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    journal = GDriveReplacementJournal(tmp_path / "replacements.json", repository)

    entry = journal.prepare("source", ("old",), ("new",), (identity,))
    status = repository.load_sync_status(identity)
    assert status is not None
    assert status.status == "replacement-pending"
    journal.mark_indexed(entry, (identity,))
    status = repository.load_sync_status(identity)
    assert status is not None
    assert status.status == "replacement-indexed"
    journal.complete("source", (identity,))
    status = repository.load_sync_status(identity)
    assert status is not None
    assert status.status == "healthy"


def test_drive_policy_groups_replacements_before_saving_source_membership(
    record_manager,
    tmp_path: Path,
) -> None:
    """
    Apply a grouped Drive batch through the isolated policy boundary.
    """
    first = _record("file-1:chunk-a", "first body")
    second = _record("file-2:chunk-a", "second body")
    second = replace(
        second,
        metadata={**second.metadata, "gdrive_source_id": "file-2"},
    )
    source_map = SqliteSourceMapStore(_SingleConnectionProvider(tmp_path / "index.db"))
    source_records = source_map.load()
    policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        record_manager.storage,
        source_records,
        source_map,
        GDriveReplacementJournal(tmp_path / "gdrive-replacements.json"),
    )

    asyncio.run(policy.replace((first, second)))

    saved = source_map.load()
    assert set(saved) == {
        canonical_gdrive_source_key(first),
        canonical_gdrive_source_key(second),
    }
    assert saved[canonical_gdrive_source_key(first)] == [first.storage_key]
    assert saved[canonical_gdrive_source_key(second)] == [second.storage_key]


def test_drive_policy_applies_tombstone_membership_update(
    record_manager,
    tmp_path: Path,
) -> None:
    """
    Retain remaining scope membership while removing a tombstoned scope.
    """
    active = _record("file-1:chunk-a", "shared body", scopes=("drive-a", "drive-b"))
    tombstone = _record("file-1", "ignored", deleted=True, scopes=("drive-a",))
    source_map = SqliteSourceMapStore(_SingleConnectionProvider(tmp_path / "index.db"))
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        record_manager.storage,
        source_map.load(),
        source_map,
        GDriveReplacementJournal(tmp_path / "gdrive-replacements.json", repository),
        repository,
    )

    asyncio.run(policy.replace((active,)))
    asyncio.run(policy.replace((tombstone,)))

    hydrated = record_manager.storage.hydrate_record(active.storage_key)
    assert hydrated is not None
    assert hydrated.metadata["scope_memberships"] == ["drive-b"]
    assert repository.memberships_for_source("gdrive", "workspace", "file-1") == (
        "drive-b",
    )


def test_delete_stale_keys_removes_stale_keys_from_storage_and_source_map(
    tmp_path: Path, record_manager
) -> None:
    """
    Drop a key no longer present in the new batch from storage and from the
    source's own retained-key list.
    """
    storage = _DeleteSpyStorage()
    source_records = {"source-a": ["key-1", "key-2", "key-3"]}
    policy = _policy_for_delete_stale_keys(tmp_path, record_manager, source_records, storage)

    reverse_index = policy._build_reverse_index()
    delta: dict[str, list[str] | None] = {}
    policy._delete_stale_keys(
        "source-a", ("key-1", "key-2", "key-3"), ("key-1", "key-3"), reverse_index, delta
    )

    assert storage.deleted == [("key-2",)]
    assert source_records["source-a"] == ["key-1", "key-3"]


def test_delete_stale_keys_strips_a_key_claimed_by_another_source(
    tmp_path: Path, record_manager
) -> None:
    """
    A key that a new batch now claims for source_key must be released from
    any other doc_id that was still carrying it (the same-source vs
    other-source asymmetry).
    """
    storage = _DeleteSpyStorage()
    source_records = {
        "source-a": ["key-1"],
        "other-doc": ["key-2", "shared-key"],
    }
    policy = _policy_for_delete_stale_keys(tmp_path, record_manager, source_records, storage)

    reverse_index = policy._build_reverse_index()
    delta: dict[str, list[str] | None] = {}
    policy._delete_stale_keys(
        "source-a", ("key-1",), ("key-1", "shared-key"), reverse_index, delta
    )

    assert source_records["other-doc"] == ["key-2"]
    assert source_records["source-a"] == ["key-1"]


def test_delete_stale_keys_strips_own_stale_key_even_when_old_keys_omits_it(
    tmp_path: Path, record_manager
) -> None:
    """
    For the entry belonging to source_key itself, any key absent from
    new_keys is dropped even if the caller-supplied old_keys did not
    report it as stale -- but it is not sent to storage.delete(), since
    only keys in stale_keys (old_keys - new_keys) are deleted there.
    """
    storage = _DeleteSpyStorage()
    source_records = {"source-a": ["key-1", "ghost-key"]}
    policy = _policy_for_delete_stale_keys(tmp_path, record_manager, source_records, storage)

    reverse_index = policy._build_reverse_index()
    delta: dict[str, list[str] | None] = {}
    policy._delete_stale_keys("source-a", ("key-1",), ("key-1",), reverse_index, delta)

    assert source_records["source-a"] == ["key-1"]
    assert storage.deleted == []


def test_delete_stale_keys_deduplicates_preserving_first_occurrence_order(
    tmp_path: Path, record_manager
) -> None:
    """
    Retained keys are de-duplicated, keeping each key's first-seen position.
    """
    storage = _DeleteSpyStorage()
    source_records = {"source-a": ["key-2", "key-1", "key-2", "key-1"]}
    policy = _policy_for_delete_stale_keys(tmp_path, record_manager, source_records, storage)

    reverse_index = policy._build_reverse_index()
    delta: dict[str, list[str] | None] = {}
    policy._delete_stale_keys(
        "source-a", ("key-1", "key-2"), ("key-1", "key-2"), reverse_index, delta
    )

    assert source_records["source-a"] == ["key-2", "key-1"]


def test_delete_stale_keys_pops_doc_id_when_nothing_is_retained(
    tmp_path: Path, record_manager
) -> None:
    """
    Remove a doc_id from the source map entirely once every one of its keys
    is stale.
    """
    storage = _DeleteSpyStorage()
    source_records = {"source-a": ["key-1", "key-2"]}
    policy = _policy_for_delete_stale_keys(tmp_path, record_manager, source_records, storage)

    reverse_index = policy._build_reverse_index()
    delta: dict[str, list[str] | None] = {}
    policy._delete_stale_keys("source-a", ("key-1", "key-2"), (), reverse_index, delta)

    assert "source-a" not in source_records
    assert storage.deleted == [("key-1", "key-2")]


def test_drive_policy_recovers_indexed_journal_entry(tmp_path: Path, record_manager) -> None:
    """
    Complete stale cleanup and source-map repair from an indexed journal entry.
    """
    old = _record("file-1:chunk-a", "old body")
    new = _record("file-1:chunk-b", "new body")
    asyncio.run(record_manager.ingestor.index_records((old, new)))
    source_map = SqliteSourceMapStore(_SingleConnectionProvider(tmp_path / "index.db"))
    source_records = {canonical_gdrive_source_key(old): [old.storage_key]}
    journal = GDriveReplacementJournal(tmp_path / "gdrive-replacements.json")
    entry = journal.prepare(
        canonical_gdrive_source_key(new),
        (old.storage_key,),
        (new.storage_key,),
    )
    journal.mark_indexed(entry)
    policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        record_manager.storage,
        source_records,
        source_map,
        journal,
    )

    assert policy.recover() is True

    assert record_manager.storage.hydrate_record(old.storage_key) is None
    assert record_manager.storage.hydrate_record(new.storage_key) is not None
    assert source_map.load() == {
        canonical_gdrive_source_key(new): [new.storage_key]
    }
    assert journal.load() == ()


def test_drive_replace_hydrates_no_existing_keys_when_every_key_is_already_tracked(
    record_manager,
    tmp_path: Path,
) -> None:
    """
    Look up existing keys for every source without hydrating any record when
    every candidate key is already tracked in the source map (the common
    case): `_record_keys_by_source` recovers each key's source key from
    `_source_records` itself, only hydrating on genuine drift.
    """
    setup_journal = GDriveReplacementJournal(tmp_path / "setup-gdrive-replacements.json")
    setup_source_map = SqliteSourceMapStore(_SingleConnectionProvider(tmp_path / "index.db"))
    setup_policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        record_manager.storage,
        setup_source_map.load(),
        setup_source_map,
        setup_journal,
    )
    first = _record("file-1:chunk-a", "first body")
    second = _record("file-2:chunk-a", "second body")
    second = replace(
        second, metadata={**second.metadata, "gdrive_source_id": "file-2"}
    )
    third = _record("file-3:chunk-a", "third body")
    third = replace(third, metadata={**third.metadata, "gdrive_source_id": "file-3"})
    asyncio.run(setup_policy.replace((first, second, third)))

    counting_storage = _CountingRecordStorage(record_manager.storage)
    journal = GDriveReplacementJournal(tmp_path / "gdrive-replacements.json")
    policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        counting_storage,
        setup_source_map.load(),
        setup_source_map,
        journal,
    )
    updated_first = _record("file-1:chunk-a", "updated first body")
    updated_second = replace(
        _record("file-2:chunk-a", "updated second body"),
        metadata={**second.metadata, "gdrive_source_id": "file-2"},
    )
    updated_third = replace(
        _record("file-3:chunk-a", "updated third body"),
        metadata={**third.metadata, "gdrive_source_id": "file-3"},
    )

    asyncio.run(policy.replace((updated_first, updated_second, updated_third)))

    assert counting_storage.hydrate_records_calls == 0
    assert counting_storage.hydrate_record_calls == 0

    saved = setup_source_map.load()
    assert saved[canonical_gdrive_source_key(updated_first)] == [
        updated_first.storage_key
    ]
    assert saved[canonical_gdrive_source_key(updated_second)] == [
        updated_second.storage_key
    ]
    assert saved[canonical_gdrive_source_key(updated_third)] == [
        updated_third.storage_key
    ]


def test_record_keys_by_source_matches_hydrated_mapping_with_and_without_drift(
    record_manager,
    tmp_path: Path,
) -> None:
    """
    _record_keys_by_source must return the same source_key -> keys mapping a
    full hydrate-and-group pass would produce, whether or not every Drive key
    is already tracked in _source_records. It should need zero hydrations
    when everything is already tracked, and hydrate only the untracked
    (drifted) keys otherwise.
    """
    journal = GDriveReplacementJournal(tmp_path / "gdrive-replacements.json")
    source_map = SqliteSourceMapStore(_SingleConnectionProvider(tmp_path / "index.db"))
    policy = GDriveReplacementPolicy(
        record_manager.ingestor, record_manager.storage, source_map.load(), source_map, journal
    )
    first = _record("file-1:chunk-a", "first body")
    second = _record("file-2:chunk-a", "second body")
    second = replace(second, metadata={**second.metadata, "gdrive_source_id": "file-2"})
    asyncio.run(policy.replace((first, second)))

    counting_storage = _CountingRecordStorage(record_manager.storage)
    tracked_policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        counting_storage,
        policy._source_records,
        source_map,
        journal,
    )

    baseline = tracked_policy._record_keys_by_source()

    assert counting_storage.hydrate_records_calls == 0
    assert baseline == {
        canonical_gdrive_source_key(first): (first.storage_key,),
        canonical_gdrive_source_key(second): (second.storage_key,),
    }

    # Simulate drift: a Drive record indexed directly, bypassing the policy,
    # so _source_records has no idea what source key it belongs to.
    third = _record("file-3:chunk-a", "third body")
    third = replace(third, metadata={**third.metadata, "gdrive_source_id": "file-3"})
    asyncio.run(record_manager.ingestor.index_records((third,)))
    counting_storage.hydrate_records_calls = 0

    with_drift = tracked_policy._record_keys_by_source()

    assert counting_storage.hydrate_records_calls == 1
    assert with_drift == {
        **baseline,
        canonical_gdrive_source_key(third): (third.storage_key,),
    }


def test_drive_replace_applies_incremental_source_map_delta_not_full_rewrite(
    record_manager,
    tmp_path: Path,
) -> None:
    """
    A replace() touching one source must write only that source's row to the
    source map, not rewrite every row in the table.
    """
    connection_provider = _SingleConnectionProvider(tmp_path / "index.db")
    spy_store = _SourceMapWriteSpy(connection_provider)
    journal = GDriveReplacementJournal(tmp_path / "gdrive-replacements.json")
    policy = GDriveReplacementPolicy(
        record_manager.ingestor, record_manager.storage, spy_store.load(), spy_store, journal
    )
    first = _record("file-1:chunk-a", "first body")
    second = _record("file-2:chunk-a", "second body")
    second = replace(second, metadata={**second.metadata, "gdrive_source_id": "file-2"})
    asyncio.run(policy.replace((first, second)))
    spy_store.save_calls.clear()
    spy_store.apply_delta_calls.clear()

    updated_first = _record("file-1:chunk-a", "updated first body")
    asyncio.run(policy.replace((updated_first,)))

    assert spy_store.save_calls == []
    assert len(spy_store.apply_delta_calls) == 1
    upserts, removals = spy_store.apply_delta_calls[0]
    assert set(upserts) == {canonical_gdrive_source_key(updated_first)}
    assert removals == []
    assert spy_store.load() == {
        canonical_gdrive_source_key(updated_first): [updated_first.storage_key],
        canonical_gdrive_source_key(second): [second.storage_key],
    }


def test_journal_entries_survive_a_simulated_process_restart(tmp_path: Path) -> None:
    """
    A fresh journal instance over the same path sees every durable write.
    """
    path = tmp_path / "gdrive-replacements.json"
    journal = GDriveReplacementJournal(path)
    first = journal.prepare("source-a", ("old-a",), ("new-a",))
    journal.mark_indexed(first)
    journal.prepare("source-b", (), ("new-b",))

    restarted = GDriveReplacementJournal(path)

    assert restarted.load() == (
        replace(first, phase="indexed"),
        GDriveReplacementEntry("source-b", (), ("new-b",)),
    )


def test_journal_interleaves_prepare_mark_indexed_complete_across_sources(
    tmp_path: Path,
) -> None:
    """
    Independent source keys progress through their phases without clobbering
    each other's state.
    """
    path = tmp_path / "gdrive-replacements.json"
    journal = GDriveReplacementJournal(path)

    entry_a = journal.prepare("source-a", ("old-a",), ("new-a",))
    entry_b = journal.prepare("source-b", ("old-b",), ("new-b",))
    journal.mark_indexed(entry_a)
    journal.complete("source-a")
    entry_c = journal.prepare("source-c", (), ("new-c",))
    journal.mark_indexed(entry_b)
    journal.mark_indexed(entry_c)
    journal.complete("source-b")
    journal.complete("source-c")

    assert journal.load() == ()
    assert GDriveReplacementJournal(path).load() == ()


def test_journal_writes_do_not_reread_the_journal_file_per_entry(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Recording many replacement entries in one process lifetime must not cost
    one full journal read+parse per entry: that turns a resync touching K
    Drive sources into O(K^2) journal I/O. At most one read should occur, to
    hydrate the in-process cache the first time it is needed.
    """
    from mcp_markdown_ragdocs.gdrive.json_record_store import JsonEnvelopeStore

    read_calls = 0
    original_read = JsonEnvelopeStore.read

    def counting_read(self, expected_type):
        nonlocal read_calls
        read_calls += 1
        return original_read(self, expected_type)

    monkeypatch.setattr(JsonEnvelopeStore, "read", counting_read)

    journal = GDriveReplacementJournal(tmp_path / "gdrive-replacements.json")
    for index in range(50):
        entry = journal.prepare(f"source-{index}", (), (f"key-{index}",))
        journal.mark_indexed(entry)
        journal.complete(f"source-{index}")

    assert read_calls <= 1
