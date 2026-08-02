from pathlib import Path

from tests.conftest import create_test_document


def test_reconcile_after_file_move_removes_old_source_and_indexes_new_path(
    record_manager,
) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    old_path = Path(create_test_document(docs_dir, "original", "# Title\n\nSame body"))
    assert record_manager.index_document(str(old_path)) is True
    old_id = record_manager.prepare_document(str(old_path)).document.id

    new_path = docs_dir / "renamed.md"
    old_path.rename(new_path)
    result = record_manager.reconcile_indices([str(new_path)], docs_dir)

    new_id = record_manager.prepare_document(str(new_path)).document.id
    assert result.added_count == 1
    assert result.removed_count == 1
    assert old_id not in record_manager._source_records
    assert new_id in record_manager._source_records
    assert record_manager.describe_documents()[0]["file_path"] == str(new_path)
