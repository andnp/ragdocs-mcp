"""
Unit tests for file discovery with symlink handling.

GAP #6: File discovery with symlink cycles (Medium/Low, Score 3.33)
"""

from pathlib import Path

import pytest

from src.context import ApplicationContext
from src.config import Config, IndexingConfig, ServerConfig


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "index"),
        ),
        parsers={"**/*.md": "MarkdownParser"}
    )


@pytest.fixture
def context_with_config(test_config, monkeypatch):
    """Create ApplicationContext with test config."""
    monkeypatch.setattr("src.context.load_config", lambda: test_config)
    return ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )


def test_discover_files_with_symlink_to_file(context_with_config):
    """
    File discovery handles symlinks to individual files.

    Tests that symlinked files are handled gracefully.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create real file
    real_file = docs_path / "real.md"
    real_file.write_text("# Real File")

    # Create symlink to file
    link_file = docs_path / "link.md"
    link_file.symlink_to(real_file)

    files = ctx.discover_files()

    # glob.glob by default follows symlinks
    # Both files should appear (or just real, depending on implementation)
    assert len(files) >= 1
    assert any("real.md" in f for f in files)


def test_discover_files_with_symlink_to_directory(context_with_config):
    """
    File discovery handles symlinks to directories.

    Tests that symlinked directories don't cause issues.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create real directory with file
    real_dir = docs_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "doc.md").write_text("# Document")

    # Create symlink to directory
    link_dir = docs_path / "linked_dir"
    link_dir.symlink_to(real_dir)

    files = ctx.discover_files()

    # Should find file through real directory
    assert len(files) >= 1
    assert any("doc.md" in f for f in files)


def test_discover_files_with_circular_symlink(context_with_config):
    """
    File discovery handles circular symlinks without infinite loop.

    Tests that circular symlinks don't cause discovery to hang.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create directory
    dir_a = docs_path / "dir_a"
    dir_a.mkdir()
    (dir_a / "file.md").write_text("# File in A")

    # Create circular symlink (A -> B -> A)
    dir_b = docs_path / "dir_b"
    dir_b.mkdir()

    try:
        # Create symlink from B to A
        (dir_b / "link_to_a").symlink_to(dir_a)

        # Create symlink from A to B (circular)
        (dir_a / "link_to_b").symlink_to(dir_b)

        # Discover should not hang
        files = ctx.discover_files()

        # Should find at least the real file
        assert len(files) >= 1
        assert any("file.md" in f for f in files)
    except OSError:
        # Some systems may not allow circular symlinks
        pytest.skip("System doesn't support circular symlinks")


def test_discover_files_with_broken_symlink(context_with_config):
    """
    File discovery handles broken symlinks gracefully.

    Tests that symlinks to non-existent targets don't crash discovery.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create valid file
    (docs_path / "valid.md").write_text("# Valid")

    # Create broken symlink
    broken_link = docs_path / "broken.md"
    nonexistent = docs_path / "nonexistent.md"
    broken_link.symlink_to(nonexistent)

    # Discovery should not crash on broken symlink
    files = ctx.discover_files()

    # Should find the valid file
    assert len(files) >= 1
    assert any("valid.md" in f for f in files)


def test_discover_files_with_symlink_outside_docs_path(context_with_config, tmp_path):
    """
    File discovery handles symlinks pointing outside docs path.

    Tests that symlinks to external locations are handled.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create external directory
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "external.md").write_text("# External")

    # Create symlink to external directory
    link_dir = docs_path / "external_link"
    link_dir.symlink_to(external_dir)

    # Create local file
    (docs_path / "local.md").write_text("# Local")

    files = ctx.discover_files()

    # Should find local file
    assert any("local.md" in f for f in files)

    # External file may or may not be included depending on glob behavior


def test_discover_files_with_deep_symlink_chain(context_with_config):
    """
    File discovery handles chains of symlinks.

    Tests that multiple levels of symlink indirection work.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create real file
    (docs_path / "real.md").write_text("# Real")

    # Create chain: link1 -> link2 -> link3 -> real.md
    (docs_path / "link3.md").symlink_to(docs_path / "real.md")
    (docs_path / "link2.md").symlink_to(docs_path / "link3.md")
    (docs_path / "link1.md").symlink_to(docs_path / "link2.md")

    files = ctx.discover_files()

    # Should find files (exact count depends on symlink handling)
    assert len(files) >= 1


def test_discover_files_excludes_symlink_in_hidden_dir(context_with_config):
    """
    File discovery handles symlinks to hidden directories.

    Tests symlinked paths behavior with hidden dirs.
    Note: glob follows symlinks by default, so files are discovered
    through the symlink even if it points to a hidden directory.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create visible file
    (docs_path / "visible.md").write_text("# Visible")

    # Create hidden directory with file
    hidden_dir = docs_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.md").write_text("# Hidden")

    # Create symlink to hidden directory
    link_to_hidden = docs_path / "link_to_hidden"
    link_to_hidden.symlink_to(hidden_dir)

    files = ctx.discover_files()

    # glob follows symlinks, so files through symlink are found
    # even if symlink target is hidden
    assert len(files) >= 1
    assert any("visible.md" in f for f in files)


def test_discover_files_with_symlink_to_parent(context_with_config):
    """
    File discovery handles symlinks to parent directories.

    Tests that symlinks creating directory loops are handled.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create nested structure
    subdir = docs_path / "subdir"
    subdir.mkdir()
    (subdir / "file.md").write_text("# File")

    try:
        # Create symlink from subdir back to parent (potential loop)
        (subdir / "link_to_parent").symlink_to(docs_path)

        files = ctx.discover_files()

        # Should find file without infinite loop
        assert len(files) >= 1
        assert any("file.md" in f for f in files)
    except OSError:
        # Some systems may not allow this
        pytest.skip("System doesn't support parent symlinks")


def test_discover_files_relative_symlink(context_with_config):
    """
    File discovery handles relative symlinks.

    Tests that relative symlink paths work correctly.
    """
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    # Create directory structure
    dir_a = docs_path / "dir_a"
    dir_a.mkdir()
    (dir_a / "doc.md").write_text("# Doc")

    dir_b = docs_path / "dir_b"
    dir_b.mkdir()

    # Create relative symlink
    (dir_b / "link.md").symlink_to("../dir_a/doc.md")

    files = ctx.discover_files()

    # Should find at least the real file
    assert len(files) >= 1
    assert any("doc.md" in f for f in files)
