"""
Unit tests for manifest version migration.

GAP #8: Manifest version migration (Medium/Low, Score 3.33)
"""



from src.indexing.manifest import (
    IndexManifest,
    load_manifest,
    save_manifest,
    should_rebuild,
)


def test_should_rebuild_triggers_on_version_upgrade(tmp_path):
    """
    Trigger rebuild when spec version increases.

    Tests that version upgrades trigger rebuild.
    """
    # Old version manifest
    old_manifest = IndexManifest(
        spec_version="0.9.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )

    # Current version manifest
    current_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )

    assert should_rebuild(current_manifest, old_manifest) is True


def test_should_rebuild_triggers_on_version_downgrade(tmp_path):
    """
    Trigger rebuild when spec version decreases.

    Tests that version downgrades also trigger rebuild.
    """
    # Newer version manifest (simulating rollback scenario)
    newer_manifest = IndexManifest(
        spec_version="2.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )

    # Current version manifest
    current_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )

    assert should_rebuild(current_manifest, newer_manifest) is True


def test_should_not_rebuild_when_versions_match(tmp_path):
    """
    Skip rebuild when spec versions match.

    Tests that identical versions don't trigger rebuild.
    """
    manifest1 = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files={},  # Add indexed_files to avoid rebuild trigger
    )

    manifest2 = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files={},
    )

    assert should_rebuild(manifest1, manifest2) is False


def test_version_format_variations(tmp_path):
    """
    Handle different version format variations.

    Tests that version format differences trigger rebuild.
    """
    manifest_v1 = IndexManifest(
        spec_version="1.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
    )

    manifest_v2 = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
    )

    # Different string representations should trigger rebuild
    assert should_rebuild(manifest_v1, manifest_v2) is True


def test_should_rebuild_with_missing_indexed_files_field():
    """
    Trigger rebuild when indexed_files is None.

    Tests migration from old manifests without indexed_files.
    """
    # Old manifest without indexed_files field
    old_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
        indexed_files=None,
    )

    current_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
        indexed_files={},
    )

    # Should trigger rebuild to populate indexed_files
    assert should_rebuild(current_manifest, old_manifest) is True


def test_load_old_manifest_without_indexed_files(tmp_path):
    """
    Load old manifest format without indexed_files.

    Tests backward compatibility with older manifest formats.
    """
    manifest_path = tmp_path / "index.manifest.json"

    # Write old format manifest (missing indexed_files)
    import json
    old_format = {
        "spec_version": "1.0.0",
        "embedding_model": "test-model",
        "parsers": {"**/*.md": "MarkdownParser"},
        "chunking_config": {},
        # No indexed_files field
    }
    manifest_path.write_text(json.dumps(old_format))

    loaded = load_manifest(tmp_path)

    assert loaded is not None
    assert loaded.spec_version == "1.0.0"
    assert loaded.indexed_files is None  # Should handle missing field


def test_save_and_load_manifest_preserves_version(tmp_path):
    """
    Save and load preserves exact version string.

    Tests that version strings are not modified during save/load.
    """
    manifest = IndexManifest(
        spec_version="1.2.3-beta",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
    )

    save_manifest(tmp_path, manifest)
    loaded = load_manifest(tmp_path)

    assert loaded is not None
    assert loaded.spec_version == "1.2.3-beta"


def test_should_rebuild_with_chunking_config_changes():
    """
    Trigger rebuild when chunking config changes.

    Tests that chunking parameter changes trigger rebuild.
    """
    old_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={
            "strategy": "hierarchical",
            "max_chunk_chars": 1000,
        },
    )

    new_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={
            "strategy": "hierarchical",
            "max_chunk_chars": 2000,  # Changed
        },
    )

    assert should_rebuild(new_manifest, old_manifest) is True


def test_should_rebuild_with_empty_chunking_configs():
    """
    Handle empty chunking configs correctly.

    Tests that empty configs are handled consistently.
    """
    manifest1 = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
        indexed_files={},
    )

    manifest2 = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
        chunking_config={},
        indexed_files={},
    )

    assert should_rebuild(manifest1, manifest2) is False


def test_should_rebuild_with_parser_changes():
    """
    Trigger rebuild when parser configuration changes.

    Tests that parser changes trigger rebuild.
    """
    old_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
    )

    new_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={
            "**/*.md": "MarkdownParser",
            "**/*.txt": "PlainTextParser",  # Added new parser
        },
        chunking_config={},
    )

    assert should_rebuild(new_manifest, old_manifest) is True


def test_manifest_migration_path_from_0_9_to_1_0(tmp_path):
    """
    Simulate migration from v0.9 to v1.0.

    Tests complete migration workflow.
    """
    # Save old manifest
    old_manifest = IndexManifest(
        spec_version="0.9.0",
        embedding_model="old-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files=None,  # Old format didn't have this
    )
    save_manifest(tmp_path, old_manifest)

    # Load and check if rebuild needed
    loaded = load_manifest(tmp_path)
    assert loaded is not None

    current_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="old-model",
        parsers={"**/*.md": "MarkdownParser"},
        chunking_config={},
        indexed_files={},
    )

    # Should trigger rebuild due to version change
    assert should_rebuild(current_manifest, loaded) is True

    # After rebuild, save new manifest
    save_manifest(tmp_path, current_manifest)

    # Load again and verify no rebuild needed
    loaded_after_rebuild = load_manifest(tmp_path)
    assert should_rebuild(current_manifest, loaded_after_rebuild) is False
