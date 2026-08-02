"""Integration coverage for canonical local record-manager persistence."""

from pathlib import Path

from searchkernel.api import IndexManifest, load_manifest, save_manifest, should_rebuild

from mcp_markdown_ragdocs.config import Config, IndexingConfig
from tests.integration._canonical import make_record_index_manager


def _config(tmp_path: Path) -> Config:
    docs = tmp_path / "docs"
    docs.mkdir()
    return Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index")))


def test_missing_manifest_requires_rebuild_and_persists_records(tmp_path):
    config = _config(tmp_path)
    manager = make_record_index_manager(config)
    document = Path(config.indexing.documents_path) / "one.md"
    document.write_text("# One\n\nCanonical lifecycle content.")
    manifest = IndexManifest(spec_version="1", embedding_model="test", chunking_config={})

    assert should_rebuild(manifest, load_manifest(Path(config.indexing.index_path)))
    assert manager.index_document(str(document))
    manager.persist()
    save_manifest(Path(config.indexing.index_path), manifest)

    assert manager.get_document_count() == 1
    saved = load_manifest(Path(config.indexing.index_path))
    assert saved is not None
    assert saved.spec_version == "1"


def test_matching_manifest_can_reload_record_source_map(tmp_path):
    config = _config(tmp_path)
    document = Path(config.indexing.documents_path) / "one.md"
    document.write_text("# One\n\nReloadable content.")
    manager = make_record_index_manager(config)
    manager.index_document(str(document))
    manager.persist()

    reloaded = make_record_index_manager(config)
    reloaded.load()
    assert reloaded.get_document_count() == 1
    assert reloaded.describe_documents()[0]["file_path"] == str(document)


def test_manifest_mismatch_requires_rebuild(tmp_path):
    first = IndexManifest(spec_version="1", embedding_model="old", chunking_config={})
    current = IndexManifest(spec_version="1", embedding_model="new", chunking_config={})
    assert should_rebuild(current, first)


def test_remove_document_updates_persisted_record_map(tmp_path):
    config = _config(tmp_path)
    document = Path(config.indexing.documents_path) / "remove.md"
    document.write_text("# Remove\n\nTransient content.")
    manager = make_record_index_manager(config)
    manager.index_document(str(document))
    doc_id = str(manager.describe_documents()[0]["doc_id"])
    manager.remove_document(doc_id)
    manager.persist()

    reloaded = make_record_index_manager(config)
    reloaded.load()
    assert reloaded.get_document_count() == 0
