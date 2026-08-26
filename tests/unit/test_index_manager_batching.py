from pathlib import Path

from tests.conftest import create_test_document


def test_index_documents_indexes_the_complete_batch(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    first = create_test_document(docs_dir, "guide", "# Guide\n\nFirst batch document")
    second = create_test_document(docs_dir, "api", "# API\n\nSecond batch document")

    record_manager.index_documents([first, second], persist=True)

    assert record_manager.get_document_count() == 2
    assert len(record_manager.describe_documents()) == 2
    assert record_manager._source_map_store.load()


def test_index_documents_rebuilds_graph_once(record_manager, monkeypatch) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    first = create_test_document(docs_dir, "guide", "# Guide\n\nFirst batch document")
    second = create_test_document(docs_dir, "api", "# API\n\nSecond batch document")
    rebuild_calls = 0
    rebuild_graph = record_manager._rebuild_graph

    def tracked_rebuild_graph() -> None:
        nonlocal rebuild_calls
        rebuild_calls += 1
        rebuild_graph()

    monkeypatch.setattr(record_manager, "_rebuild_graph", tracked_rebuild_graph)
    record_manager.index_documents([first, second])

    assert rebuild_calls == 1


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


def test_describe_documents_reports_one_description_per_duplicated_chunk(
    record_manager,
) -> None:
    """A doc_id whose key list contains a duplicate still describes once."""
    docs_dir = Path(record_manager._config.indexing.documents_path)
    document = create_test_document(docs_dir, "guide", "# Guide\n\nContent")
    record_manager.index_document(document)

    source_id, keys = next(iter(record_manager._source_records.items()))
    keys.extend(keys[:1])

    descriptions = record_manager.describe_documents()

    assert descriptions[0]["doc_id"] == source_id
