"""Tests for idempotent Google Drive record replacement."""

import asyncio
import json
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path

from searchkernel.api import (
    GraphEdge,
    Record,
    RecordStatus,
    build_local_record_kernel,
)

from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementJournal,
    canonical_gdrive_source_key,
)
from mcp_markdown_ragdocs.gdrive.state import (
    GDriveScopeIdentity,
    GDriveStateRepository,
)
from mcp_markdown_ragdocs.gdrive.replacement_policy import GDriveReplacementPolicy
from mcp_markdown_ragdocs.indexing.record_ports import JsonSourceMapStore
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager


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

    assert record_manager.kernel.backend.hydrate_record(first[1].storage_key) is None
    assert record_manager.kernel.backend.hydrate_record(replacement[0].storage_key) is not None
    source_key = canonical_gdrive_source_key(replacement[0])
    source_map = json.loads(
        (Path(record_manager.index_path) / "record-sources.json").read_text()
    )
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

    monkeypatch.setattr(record_manager.kernel.vector_store, "delete", lambda _keys: None)
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
    source_map = json.loads(
        (Path(record_manager.index_path) / "record-sources.json").read_text()
    )
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

    assert record_manager.kernel.backend.hydrate_record(active.storage_key) is None
    assert record_manager.count_records("gdrive") == 0


def test_drive_tombstone_keeps_a_source_visible_in_remaining_scope(record_manager) -> None:
    """
    Remove one Drive scope membership without deleting a still-visible source.
    """
    active = _record("file-1:chunk-a", "shared body", scopes=("drive-a", "drive-b"))
    tombstone = _record("file-1", "ignored", deleted=True, scopes=("drive-a",))

    assert record_manager.index_record(active) is True
    assert record_manager.index_record(tombstone) is True

    hydrated = record_manager.kernel.backend.hydrate_record(active.storage_key)
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

    assert restarted.kernel.backend.hydrate_record(old.storage_key) is None
    assert restarted.kernel.backend.hydrate_record(new.storage_key) is not None
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

    assert restarted.kernel.backend.hydrate_record(old.storage_key) is not None
    assert restarted.kernel.backend.hydrate_record(new.storage_key) is None
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
    source_map = JsonSourceMapStore(tmp_path / "record-sources.json")
    source_records = source_map.load()
    policy = GDriveReplacementPolicy(
        record_manager.ingestor,
        record_manager.storage,
        source_records,
        source_map,
        GDriveReplacementJournal(tmp_path / "gdrive-replacements.json"),
    )

    asyncio.run(policy.replace((first, second)))

    saved = json.loads((tmp_path / "record-sources.json").read_text())
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
    source_map = JsonSourceMapStore(tmp_path / "record-sources.json")
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


def test_drive_policy_recovers_indexed_journal_entry(tmp_path: Path, record_manager) -> None:
    """
    Complete stale cleanup and source-map repair from an indexed journal entry.
    """
    old = _record("file-1:chunk-a", "old body")
    new = _record("file-1:chunk-b", "new body")
    asyncio.run(record_manager.ingestor.index_records((old, new)))
    source_map = JsonSourceMapStore(tmp_path / "record-sources.json")
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
    assert json.loads((tmp_path / "record-sources.json").read_text()) == {
        canonical_gdrive_source_key(new): [new.storage_key]
    }
    assert journal.load() == ()
