from pathlib import Path

from tests.conftest import create_test_document


def test_reindex_cycle_restores_canonical_records(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    doc_path = create_test_document(docs_dir, "reindex_me", "# Title\n\nBody")

    assert record_manager.index_document(doc_path) is True
    doc_id = record_manager.prepare_document(doc_path).document.id
    record_manager.remove_document(doc_id)
    assert record_manager.get_document_count() == 0

    assert record_manager.index_document(doc_path, force=True) is True
    assert record_manager.get_document_count() == 1
    assert record_manager.describe_documents()[0]["file_path"] == doc_path


def test_reconcile_indices_adds_and_removes_current_files(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    first = Path(create_test_document(docs_dir, "first", "# First\n\nBody"))
    second = Path(create_test_document(docs_dir, "second", "# Second\n\nBody"))

    result = record_manager.reconcile_indices([str(first), str(second)], docs_dir)
    assert result.added_count == 2
    assert record_manager.get_document_count() == 2

    first.unlink()
    result = record_manager.reconcile_indices([str(second)], docs_dir)
    assert result.removed_count == 1
    assert record_manager.get_document_count() == 1
