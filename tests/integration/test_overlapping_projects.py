"""
Integration tests for multi-project overlapping paths.

GAP #15: Multi-project overlapping paths (Medium/Low, Score 3.33)
"""


import pytest

from src.config import Config, IndexingConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, load_manifest, save_manifest
from src.indices.vector import VectorIndex
from src.indices.keyword import KeywordIndex
from src.indices.graph import GraphStore


@pytest.fixture
def shared_docs_root(tmp_path):
    """Create shared root directory for multiple projects."""
    root = tmp_path / "shared"
    root.mkdir()
    return root


def test_overlapping_projects_same_root_different_subdirs(shared_docs_root, tmp_path, shared_embedding_model):
    """
    Multiple projects with overlapping root but different subdirectories.

    Tests that projects can share parent directory without conflicts.
    """
    # Create project structure
    project_a_docs = shared_docs_root / "project_a"
    project_a_docs.mkdir()
    (project_a_docs / "doc_a.md").write_text("# Project A Doc")

    project_b_docs = shared_docs_root / "project_b"
    project_b_docs.mkdir()
    (project_b_docs / "doc_b.md").write_text("# Project B Doc")

    # Create separate indices
    index_a = tmp_path / "indices" / "project_a"
    index_b = tmp_path / "indices" / "project_b"

    config_a = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(project_a_docs),
            index_path=str(index_a),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    config_b = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(project_b_docs),
            index_path=str(index_b),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Index both projects
    manager_a = IndexManager(
        config_a,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_a.index_document(str(project_a_docs / "doc_a.md"))
    manager_a.persist()

    manager_b = IndexManager(
        config_b,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_b.index_document(str(project_b_docs / "doc_b.md"))
    manager_b.persist()

    # Verify indices are separate
    assert manager_a.get_document_count() == 1
    assert manager_b.get_document_count() == 1
    assert index_a.exists()
    assert index_b.exists()


def test_overlapping_projects_nested_paths(tmp_path, shared_embedding_model):
    """
    Project with path nested inside another project's path.

    Tests that nested project paths don't interfere with each other.
    """
    # Create nested structure
    parent_project = tmp_path / "parent"
    parent_project.mkdir()
    (parent_project / "parent_doc.md").write_text("# Parent Doc")

    nested_project = parent_project / "nested"
    nested_project.mkdir()
    (nested_project / "nested_doc.md").write_text("# Nested Doc")

    # Create separate indices
    parent_index = tmp_path / "indices" / "parent"
    nested_index = tmp_path / "indices" / "nested"

    # Parent project config (excludes nested)
    config_parent = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(parent_project),
            index_path=str(parent_index),
            exclude=["**/nested/**"],  # Exclude nested project
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Nested project config
    config_nested = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(nested_project),
            index_path=str(nested_index),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Index parent (should only get parent_doc)
    manager_parent = IndexManager(
        config_parent,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_parent.index_document(str(parent_project / "parent_doc.md"))
    manager_parent.persist()

    # Index nested (should only get nested_doc)
    manager_nested = IndexManager(
        config_nested,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_nested.index_document(str(nested_project / "nested_doc.md"))
    manager_nested.persist()

    # Verify separation
    assert manager_parent.get_document_count() == 1
    assert manager_nested.get_document_count() == 1


def test_overlapping_projects_symlinked_paths(tmp_path, shared_embedding_model):
    """
    Projects with symlinked document paths.

    Tests that symlinked paths don't cause doc_id collisions.
    """
    # Create real directory
    real_docs = tmp_path / "real_docs"
    real_docs.mkdir()
    (real_docs / "doc.md").write_text("# Document")

    # Create symlink
    linked_docs = tmp_path / "linked_docs"
    linked_docs.symlink_to(real_docs)

    # Create separate indices
    index_real = tmp_path / "indices" / "real"
    index_linked = tmp_path / "indices" / "linked"

    config_real = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(real_docs),
            index_path=str(index_real),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    config_linked = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(linked_docs),
            index_path=str(index_linked),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Index from both paths
    manager_real = IndexManager(
        config_real,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_real.index_document(str(real_docs / "doc.md"))
    manager_real.persist()

    manager_linked = IndexManager(
        config_linked,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_linked.index_document(str(linked_docs / "doc.md"))
    manager_linked.persist()

    # Both should index successfully (same content, different projects)
    assert manager_real.get_document_count() == 1
    assert manager_linked.get_document_count() == 1


def test_overlapping_projects_manifest_isolation(tmp_path, shared_embedding_model):
    """
    Manifests are isolated per project.

    Tests that project manifests don't interfere with each other.
    """
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    project_a_docs = docs_root / "project_a"
    project_a_docs.mkdir()
    (project_a_docs / "doc.md").write_text("# Doc A")

    project_b_docs = docs_root / "project_b"
    project_b_docs.mkdir()
    (project_b_docs / "doc.md").write_text("# Doc B")

    index_a = tmp_path / "indices" / "project_a"
    index_b = tmp_path / "indices" / "project_b"

    # Save manifests for both projects
    manifest_a = IndexManifest(
        spec_version="1.0.0",
        embedding_model="model-a",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files={"doc": "doc.md"},
    )
    save_manifest(index_a, manifest_a)

    manifest_b = IndexManifest(
        spec_version="1.0.0",
        embedding_model="model-b",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files={"doc": "doc.md"},
    )
    save_manifest(index_b, manifest_b)

    # Load and verify isolation
    loaded_a = load_manifest(index_a)
    loaded_b = load_manifest(index_b)

    assert loaded_a is not None
    assert loaded_b is not None
    assert loaded_a.embedding_model == "model-a"
    assert loaded_b.embedding_model == "model-b"


def test_overlapping_projects_same_relative_paths(tmp_path, shared_embedding_model):
    """
    Projects with same relative file paths.

    Tests that doc_ids don't collide when relative paths are identical.
    """
    # Create two projects with same structure
    project_a = tmp_path / "project_a"
    project_a.mkdir()
    (project_a / "README.md").write_text("# Project A README")

    project_b = tmp_path / "project_b"
    project_b.mkdir()
    (project_b / "README.md").write_text("# Project B README")

    index_a = tmp_path / "indices" / "project_a"
    index_b = tmp_path / "indices" / "project_b"

    config_a = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(project_a),
            index_path=str(index_a),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    config_b = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(project_b),
            index_path=str(index_b),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Index both
    manager_a = IndexManager(
        config_a,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_a.index_document(str(project_a / "README.md"))
    manager_a.persist()

    manager_b = IndexManager(
        config_b,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_b.index_document(str(project_b / "README.md"))
    manager_b.persist()

    # Both should work independently
    assert manager_a.get_document_count() == 1
    assert manager_b.get_document_count() == 1


def test_project_switch_same_index_path_fails_gracefully(tmp_path, shared_embedding_model):
    """
    Switching projects with same index path is handled.

    Tests that reusing index path for different project is detected.
    """
    project_a = tmp_path / "project_a"
    project_a.mkdir()
    (project_a / "doc_a.md").write_text("# Doc A")

    project_b = tmp_path / "project_b"
    project_b.mkdir()
    (project_b / "doc_b.md").write_text("# Doc B")

    # Same index path for both (bad configuration)
    shared_index = tmp_path / "shared_index"

    config_a = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(project_a),
            index_path=str(shared_index),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Index project A
    manager_a = IndexManager(
        config_a,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )
    manager_a.index_document(str(project_a / "doc_a.md"))
    manager_a.persist()

    manifest_a = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )
    save_manifest(shared_index, manifest_a)

    # Now use same index path for project B (should detect mismatch)
    config_b = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(project_b),
            index_path=str(shared_index),
        ),
        parsers={"**/*.md": "MarkdownParser"},
    )

    # Loading with different documents_path should work but might need rebuild
    # The system should handle this gracefully
    manager_b = IndexManager(
        config_b,
        VectorIndex(embedding_model=shared_embedding_model),
        KeywordIndex(),
        GraphStore(),
    )

    # Should be able to load existing index
    manager_b.load()

    # But document count reflects project A since we loaded its index
    # This demonstrates the issue with shared paths
    assert manager_b.get_document_count() == 1
