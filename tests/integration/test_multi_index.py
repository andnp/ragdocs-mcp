"""Integration coverage for the canonical record-backed index stack."""

from pathlib import Path

from mcp_markdown_ragdocs.config import Config, IndexingConfig
from tests.integration._canonical import make_record_index_manager


def _manager(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return make_record_index_manager(
        Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index")))
    )


def test_index_document_updates_record_stores(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "sample.md"
    document.write_text("# Sample\n\nSearchable record content.")

    assert manager.index_document(str(document))
    assert manager.get_document_count() == 1
    assert manager.count_records("note") >= 1
    assert manager.keyword.search("searchable record", 5)
    assert manager.vector.search(
        manager.embedding_provider.embed(["searchable record"])[0],
        5,
        model_name=manager.embedding_provider.model_name,
        dim=manager.embedding_provider.dim,
    )


def test_index_document_preserves_heading_retrieval_fields(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "headings.md"
    document.write_text("# Guide\n\n## Authentication\n\nUse bearer tokens.")

    assert manager.index_document(str(document))

    records = [
        manager.kernel.backend.hydrate_record(key)
        for key in manager._source_records[manager.prepare_document(str(document)).document.id]
    ]
    authentication = next(
        record for record in records if record and record.metadata.get("header_path") == "Guide > Authentication"
    )

    assert authentication.title == "Guide > Authentication"
    assert authentication.indexed_text == authentication.body


def test_remove_document_removes_record_from_all_local_stores(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "remove.md"
    document.write_text("# Remove\n\nContent to remove.")
    manager.index_document(str(document))
    doc_id = str(manager.describe_documents()[0]["doc_id"])

    manager.remove_document(doc_id)

    assert manager.get_document_count() == 0
    assert manager.count_records("note") == 0
    assert manager.keyword.search("content", 5) == []


def test_links_are_written_to_canonical_graph_store(tmp_path):
    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    target = docs / "target.md"
    source = docs / "source.md"
    target.write_text("# Target\n\nTarget content.")
    source.write_text("# Source\n\nSee [target](target.md).")
    manager.index_document(str(target))
    manager.index_document(str(source))

    assert manager.graph.graph_integrity_errors() == []


def test_markdown_links_retrieve_indexed_target_chunks(tmp_path):
    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    source = docs / "source.md"
    target = docs / "target.md"
    source.write_text("# Source\n\nSee [target](target.md).")
    target.write_text(
        "# Target\n\n"
        "This target has enough content to produce a canonical chunk identity."
    )

    manager.index_document(str(source))
    manager.index_document(str(target))

    source_record = manager.prepare_document(str(source)).records[0]
    neighbors = manager.graph.neighbors(source_record.identity)

    assert neighbors
    assert all(neighbor.identity.source_id.startswith("target_chunk_") for neighbor in neighbors)
    assert manager.graph.graph_integrity_errors() == []


def test_empty_document_does_not_break_record_manager(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "empty.md"
    document.write_text("")

    assert manager.index_document(str(document))
    assert manager.is_ready()
