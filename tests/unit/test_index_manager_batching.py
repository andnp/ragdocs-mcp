from pathlib import Path

from tests.conftest import create_test_document


def test_index_documents_indexes_the_complete_batch(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    first = create_test_document(docs_dir, "guide", "# Guide\n\nFirst batch document")
    second = create_test_document(docs_dir, "api", "# API\n\nSecond batch document")

    record_manager.index_documents([first, second], persist=True)

    assert record_manager.get_document_count() == 2
    assert len(record_manager.describe_documents()) == 2
    assert (Path(record_manager.index_path) / "record-sources.json").exists()


def test_remove_documents_removes_the_complete_batch(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    first = create_test_document(docs_dir, "guide", "# Guide\n\nFirst batch document")
    second = create_test_document(docs_dir, "api", "# API\n\nSecond batch document")
    record_manager.index_documents([first, second], persist=True)

    first_id = record_manager.prepare_document(first).document.id
    second_id = record_manager.prepare_document(second).document.id
    record_manager.remove_documents([first_id, second_id], persist=True)

    assert record_manager.get_document_count() == 0
    assert record_manager.describe_documents() == []
