from pathlib import Path

from tests.conftest import create_test_document


def test_reindexing_changed_markdown_replaces_stale_records(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    doc_path = Path(create_test_document(docs_dir, "guide", "# Guide\n\nOriginal body"))

    assert record_manager.index_document(str(doc_path)) is True
    old_id = record_manager.prepare_document(str(doc_path)).document.id
    old_keys = set(record_manager._source_records[old_id])

    doc_path.write_text("# Guide\n\nUpdated body")
    assert record_manager.index_document(str(doc_path)) is True

    new_keys = set(record_manager._source_records[old_id])
    assert new_keys == old_keys
    hydrated = [record_manager.kernel.backend.hydrate_record(key) for key in new_keys]
    assert [record.body for record in hydrated if record is not None] == [
        "Guide\n\nUpdated body"
    ]
