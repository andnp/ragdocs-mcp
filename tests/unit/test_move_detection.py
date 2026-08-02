from pathlib import Path

from tests.conftest import create_test_document


def test_moved_markdown_remains_searchable_after_reconciliation(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    old_path = Path(create_test_document(docs_dir, "old", "# Guide\n\nMove me"))
    assert record_manager.index_document(str(old_path)) is True

    new_path = docs_dir / "new.md"
    old_path.rename(new_path)
    record_manager.reconcile_indices([str(new_path)], docs_dir)

    assert record_manager.keyword.search("Move me", 5)
    assert record_manager.describe_documents()[0]["file_path"] == str(new_path)
