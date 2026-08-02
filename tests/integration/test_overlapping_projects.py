"""Integration coverage for isolated canonical local record kernels."""

from pathlib import Path

from searchkernel.api import IndexManifest, load_manifest, save_manifest

from mcp_markdown_ragdocs.config import Config, IndexingConfig
from tests.integration._canonical import make_record_index_manager


def _config(docs: Path, index: Path) -> Config:
    return Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(index)))


def test_overlapping_projects_use_separate_record_databases(tmp_path):
    shared = tmp_path / "shared"
    project_a = shared / "project_a"
    project_b = shared / "project_b"
    project_a.mkdir(parents=True)
    project_b.mkdir()
    (project_a / "doc.md").write_text("# Project A")
    (project_b / "doc.md").write_text("# Project B")

    index_a = tmp_path / "indices" / "a"
    index_b = tmp_path / "indices" / "b"
    manager_a = make_record_index_manager(_config(project_a, index_a))
    manager_b = make_record_index_manager(_config(project_b, index_b))
    manager_a.index_document(str(project_a / "doc.md"))
    manager_b.index_document(str(project_b / "doc.md"))

    assert manager_a.get_document_count() == 1
    assert manager_b.get_document_count() == 1
    assert (index_a / "index.db").exists()
    assert (index_b / "index.db").exists()


def test_nested_projects_keep_document_roots_isolated(tmp_path):
    parent = tmp_path / "parent"
    nested = parent / "nested"
    nested.mkdir(parents=True)
    (parent / "parent.md").write_text("# Parent")
    (nested / "nested.md").write_text("# Nested")

    parent_manager = make_record_index_manager(
        _config(parent, tmp_path / "indices" / "parent")
    )
    nested_manager = make_record_index_manager(
        _config(nested, tmp_path / "indices" / "nested")
    )
    parent_manager.index_document(str(parent / "parent.md"))
    nested_manager.index_document(str(nested / "nested.md"))

    assert parent_manager.describe_documents()[0]["file_path"] == str(parent / "parent.md")
    assert nested_manager.describe_documents()[0]["file_path"] == str(nested / "nested.md")


def test_overlapping_project_manifests_are_isolated(tmp_path):
    index_a = tmp_path / "indices" / "a"
    index_b = tmp_path / "indices" / "b"
    save_manifest(index_a, IndexManifest(spec_version="1", embedding_model="a", chunking_config={}))
    save_manifest(index_b, IndexManifest(spec_version="1", embedding_model="b", chunking_config={}))

    loaded_a = load_manifest(index_a)
    loaded_b = load_manifest(index_b)
    assert loaded_a is not None and loaded_a.embedding_model == "a"
    assert loaded_b is not None and loaded_b.embedding_model == "b"
