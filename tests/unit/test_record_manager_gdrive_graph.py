"""Tests for graph-rebuild scoping when indexing Google Drive record batches.

Google Drive sync (gdrive/sync.py) persists after every ~100-record batch, and
persist() blocks on the debounced graph rebuild. RecordIndexManager used to
force a *full* graph rebuild (every live doc_id) on every such batch instead
of scoping the rebuild to the doc_ids the batch actually touched.
"""

from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import Record, RecordStatus

from tests.conftest import create_test_document


def _gdrive_record(source_id: str, body: str, *, links: list[str] | None = None) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    metadata: dict[str, object] = {
        "gdrive_source_id": source_id,
        "scope_memberships": ["shared-with-me"],
        "deleted": False,
        "extraction_status": "indexed",
    }
    if links is not None:
        metadata["links"] = links
    return Record(
        source_kind="gdrive",
        source_id=source_id,
        workspace_id="workspace",
        title="Drive note",
        body=body,
        indexed_text=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=metadata,
        status=RecordStatus.ACTIVE,
    )


def test_gdrive_batch_links_to_an_existing_note_form_graph_edges(
    record_manager,
    tmp_path: Path,
) -> None:
    """A Drive record's own links resolve to an already-indexed local note.

    This is the direction a scoped (non-full) rebuild always covers directly,
    since the newly dirtied Drive doc_id recomputes its own outgoing links
    from current source_records - the graph must still end up correct after
    the fix removes the forced full rebuild.
    """
    note_path = create_test_document(
        tmp_path / "docs", "note-a", "# Note A\n\nExisting local note."
    )
    assert record_manager.index_document(note_path) is True
    note_doc_id = record_manager._doc_id_for_path(note_path)
    note_key = record_manager._source_records[note_doc_id][0]

    drive_record = _gdrive_record("file-1:chunk-a", "Drive body", links=["note-a"])

    assert record_manager.index_records((drive_record,)) is True
    record_manager.persist()

    neighbors = record_manager.graph.neighbors(drive_record.identity)
    assert any(neighbor.identity.storage_key == note_key for neighbor in neighbors)


def test_gdrive_sync_pass_triggers_at_most_one_full_rebuild(
    record_manager,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Multiple gdrive batches in one sync pass do not each force a full rebuild.

    Mirrors gdrive/sync.py::_index_and_persist, which calls index_records()
    then persist() after every batch. Before the fix, each persist() forced a
    full rebuild.flush() over every live doc_id (here: every pre-existing
    note plus every previously indexed Drive doc), so the per-batch rebuild
    scope grew with total corpus size instead of staying bounded to the new
    batch.
    """
    note_count = 5
    for index in range(note_count):
        note_path = create_test_document(
            tmp_path / "docs", f"note-{index}", f"# Note {index}\n\nBody {index}."
        )
        assert record_manager.index_document(note_path) is True
    record_manager.persist()

    recomputed_sizes: list[int] = []
    original_recompute = record_manager._recompute_graph_documents

    def counting_recompute(doc_ids, source_records):
        recomputed_sizes.append(len(doc_ids))
        return original_recompute(doc_ids, source_records)

    monkeypatch.setattr(record_manager, "_recompute_graph_documents", counting_recompute)

    batch_count = 3
    for batch_index in range(batch_count):
        drive_record = _gdrive_record(
            f"file-{batch_index}:chunk-a", f"Drive body {batch_index}"
        )
        assert record_manager.index_records((drive_record,)) is True
        record_manager.persist()

    # A full rebuild per batch would recompute the whole growing corpus each
    # time (note_count + 1, note_count + 2, ... doc_ids per batch). A scoped
    # rebuild only recomputes the doc_id(s) each batch actually touched.
    full_rebuild_cost = sum(note_count + offset for offset in range(1, batch_count + 1))
    assert sum(recomputed_sizes) < full_rebuild_cost
    assert sum(recomputed_sizes) <= batch_count * 2
