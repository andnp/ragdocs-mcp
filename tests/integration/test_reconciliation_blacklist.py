"""
Integration tests for reconciliation blacklist cleanup.

Tests the system's ability to detect and remove indexed files that are now
excluded by the blacklist configuration. This handles scenarios like:
- Accidentally indexed .venv files being cleaned up
- Config changes adding new exclude patterns
- Changes to exclude_hidden_dirs setting
"""

from pathlib import Path

import pytest

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, save_manifest
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex


@pytest.fixture
def base_config(tmp_path):
    """
    Create test configuration with temporary paths.

    Uses tmp_path for isolated test storage to avoid conflicts.
    """
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
            include=["**/*"],
            exclude=["**/.venv/**", "**/node_modules/**", "**/build/**"],
            exclude_hidden_dirs=True,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(),
    )


@pytest.fixture
def indices(shared_embedding_model):
    """Create real index instances."""
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    return vector, keyword, graph


@pytest.fixture
def manager(base_config, indices):
    """Create IndexManager with real indices."""
    vector, keyword, graph = indices
    return IndexManager(base_config, vector, keyword, graph)


def test_reconciliation_removes_newly_blacklisted_venv_files(
    base_config, manager, tmp_path
):
    """
    Test that reconciliation removes .venv files after blacklist is enforced.

    Simulates the scenario where .venv files were accidentally indexed
    (perhaps before the exclude pattern was in place), and now need to
    be cleaned up during reconciliation.
    """
    docs_path = Path(base_config.indexing.documents_path)
    index_path = Path(base_config.indexing.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # Create a valid document and index it
    (docs_path / "README.md").write_text("# Project Documentation\n\nMain docs.")
    manager.index_document(str(docs_path / "README.md"))
    manager.persist()

    # Create a manifest that includes accidentally indexed .venv files
    # (simulating they were indexed before the blacklist was effective)
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers=base_config.parsers,
        chunking_config={},
        indexed_files={
            "README": "README.md",
            ".venv/lib/python3.13/site-packages/flax/README": ".venv/lib/python3.13/site-packages/flax/README.md",
            ".venv/lib/python3.13/site-packages/orbax/README": ".venv/lib/python3.13/site-packages/orbax/README.md",
        }
    )
    save_manifest(index_path, manifest)

    # Simulate the .venv files existing on disk but should be excluded
    venv_dir = docs_path / ".venv" / "lib" / "python3.13" / "site-packages"
    (venv_dir / "flax").mkdir(parents=True)
    (venv_dir / "flax" / "README.md").write_text("# Flax\n\nML library.")
    (venv_dir / "orbax").mkdir(parents=True)
    (venv_dir / "orbax" / "README.md").write_text("# Orbax\n\nCheckpointing.")

    # Discover files (will not include .venv due to exclude patterns)
    from src.utils import should_include_file
    import glob

    pattern = str(docs_path / "**" / "*.md")
    all_files = glob.glob(pattern, recursive=True)
    discovered_files = [
        f for f in all_files
        if should_include_file(
            f,
            base_config.indexing.include,
            base_config.indexing.exclude,
            base_config.indexing.exclude_hidden_dirs,
        )
    ]

    # Verify .venv files are not discovered
    assert len(discovered_files) == 1
    assert str(docs_path / "README.md") in discovered_files

    # Run reconciliation
    result = manager.reconcile_indices(discovered_files, docs_path)

    # Verify the .venv files were marked for removal
    assert result.removed_count == 2
    assert result.added_count == 0
    assert result.failed_count == 0


def test_reconciliation_handles_blacklist_config_change(
    base_config, manager, tmp_path
):
    """
    Test that changing blacklist config leads to cleanup on next reconciliation.

    Simulates a user adding '**/vendor/**' to their exclude patterns after
    having already indexed vendor files.
    """
    docs_path = Path(base_config.indexing.documents_path)
    index_path = Path(base_config.indexing.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # Create and index valid docs
    (docs_path / "main.md").write_text("# Main\n\nMain documentation.")
    manager.index_document(str(docs_path / "main.md"))
    manager.persist()

    # Create manifest with vendor files (indexed before vendor was excluded)
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers=base_config.parsers,
        chunking_config={},
        indexed_files={
            "main": "main.md",
            "vendor/package/README": "vendor/package/README.md",
        }
    )
    save_manifest(index_path, manifest)

    # Create vendor directory with files
    vendor_dir = docs_path / "vendor" / "package"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "README.md").write_text("# Vendor Package\n\nThird-party code.")

    # Update config to add vendor to exclude patterns
    updated_config = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
            include=["**/*"],
            exclude=["**/.venv/**", "**/node_modules/**", "**/build/**", "**/vendor/**"],
            exclude_hidden_dirs=True,
        ),
        parsers=base_config.parsers,
        search=SearchConfig(),
        llm=LLMConfig(),
    )

    # Create new manager with updated config
    from src.indices.vector import VectorIndex
    from src.indices.keyword import KeywordIndex
    from src.indices.graph import GraphStore

    vector_new = VectorIndex()
    keyword_new = KeywordIndex()
    graph_new = GraphStore()
    manager_new = IndexManager(updated_config, vector_new, keyword_new, graph_new)
    manager_new.load()

    # Discover files with new config
    from src.utils import should_include_file
    import glob

    pattern = str(docs_path / "**" / "*.md")
    all_files = glob.glob(pattern, recursive=True)
    discovered_files = [
        f for f in all_files
        if should_include_file(
            f,
            updated_config.indexing.include,
            updated_config.indexing.exclude,
            updated_config.indexing.exclude_hidden_dirs,
        )
    ]

    # Verify vendor is not discovered
    assert str(docs_path / "main.md") in discovered_files
    assert str(vendor_dir / "README.md") not in discovered_files

    # Run reconciliation with new config
    result = manager_new.reconcile_indices(discovered_files, docs_path)

    # Vendor file should be removed
    assert result.removed_count == 1
    assert result.added_count == 0


def test_reconciliation_respects_exclude_hidden_dirs_change(
    base_config, manager, tmp_path
):
    """
    Test that changing exclude_hidden_dirs setting triggers cleanup.

    Simulates enabling exclude_hidden_dirs after having indexed hidden
    directory files.
    """
    docs_path = Path(base_config.indexing.documents_path)
    index_path = Path(base_config.indexing.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # Create and index valid docs
    (docs_path / "visible.md").write_text("# Visible\n\nPublic documentation.")
    manager.index_document(str(docs_path / "visible.md"))
    manager.persist()

    # Create manifest with hidden directory files
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers=base_config.parsers,
        chunking_config={},
        indexed_files={
            "visible": "visible.md",
            ".hidden/secret": ".hidden/secret.md",
            ".cache/data": ".cache/data.md",
        }
    )
    save_manifest(index_path, manifest)

    # Create hidden directories with files
    hidden_dir = docs_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.md").write_text("# Secret\n\nHidden content.")

    cache_dir = docs_path / ".cache"
    cache_dir.mkdir()
    (cache_dir / "data.md").write_text("# Cache Data\n\nTemporary data.")

    # Discover files (hidden dirs excluded by default config)
    from src.utils import should_include_file
    import glob

    pattern = str(docs_path / "**" / "*.md")
    all_files = glob.glob(pattern, recursive=True)
    discovered_files = [
        f for f in all_files
        if should_include_file(
            f,
            base_config.indexing.include,
            base_config.indexing.exclude,
            base_config.indexing.exclude_hidden_dirs,
        )
    ]

    # Verify hidden dirs not discovered
    assert len(discovered_files) == 1
    assert str(docs_path / "visible.md") in discovered_files

    # Run reconciliation
    result = manager.reconcile_indices(discovered_files, docs_path)

    # Both hidden directory files should be removed
    assert result.removed_count == 2
    assert result.added_count == 0


def test_reconciliation_logs_distinct_messages_for_excluded_vs_missing(
    base_config, manager, tmp_path, caplog
):
    """
    Test that reconciliation provides distinct log messages for:
    - Files excluded by pattern (still exist on disk)
    - Files that are missing (deleted from disk)

    This helps users understand WHY files are being removed.
    """
    import logging
    caplog.set_level(logging.INFO)

    docs_path = Path(base_config.indexing.documents_path)
    index_path = Path(base_config.indexing.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # Create and index a valid doc
    (docs_path / "active.md").write_text("# Active\n\nActive documentation.")
    manager.index_document(str(docs_path / "active.md"))
    manager.persist()

    # Create manifest with:
    # 1. A file that still exists but is excluded (.venv)
    # 2. A file that is truly missing (deleted)
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="local",
        parsers=base_config.parsers,
        chunking_config={},
        indexed_files={
            "active": "active.md",
            ".venv/lib/README": ".venv/lib/README.md",  # Excluded by pattern
            "deleted": "deleted.md",  # File doesn't exist
        }
    )
    save_manifest(index_path, manifest)

    # Create the .venv file (exists but excluded)
    venv_dir = docs_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "README.md").write_text("# Package\n\nPackage docs.")

    # Note: "deleted.md" doesn't exist - simulates deleted file

    # Discover files
    from src.utils import should_include_file
    import glob

    pattern = str(docs_path / "**" / "*.md")
    all_files = glob.glob(pattern, recursive=True)
    discovered_files = [
        f for f in all_files
        if should_include_file(
            f,
            base_config.indexing.include,
            base_config.indexing.exclude,
            base_config.indexing.exclude_hidden_dirs,
        )
    ]

    # Run reconciliation
    result = manager.reconcile_indices(discovered_files, docs_path)

    # Both should be removed
    assert result.removed_count == 2

    # Check for distinct log messages
    log_text = caplog.text
    assert "excluded by pattern" in log_text
    assert "file missing" in log_text or "Stale entry" in log_text
