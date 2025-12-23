import pytest

from src.indexing.manifest import (
    IndexManifest,
    load_manifest,
    save_manifest,
    should_rebuild,
)


@pytest.fixture
def sample_manifest():
    return IndexManifest(
        spec_version="1.0.0",
        embedding_model="all-MiniLM-L6-v2",
        parsers={"**/*.md": "MarkdownParser"},
    )


def test_save_and_load_manifest(tmp_path, sample_manifest):
    """
    Save manifest to JSON and load it back.

    Verifies round-trip serialization of manifest data.
    """
    save_manifest(tmp_path, sample_manifest)

    loaded = load_manifest(tmp_path)

    assert loaded is not None
    assert loaded.spec_version == "1.0.0"
    assert loaded.embedding_model == "all-MiniLM-L6-v2"
    assert loaded.parsers == {"**/*.md": "MarkdownParser"}


def test_load_manifest_missing_file(tmp_path):
    """
    Return None when manifest file does not exist.

    Ensures graceful handling of missing manifests.
    """
    loaded = load_manifest(tmp_path)

    assert loaded is None


def test_load_manifest_corrupted_json(tmp_path):
    """
    Return None when manifest JSON is corrupted.

    Handles malformed JSON gracefully.
    """
    manifest_path = tmp_path / "index.manifest.json"
    manifest_path.write_text("{ invalid json")

    loaded = load_manifest(tmp_path)

    assert loaded is None


def test_load_manifest_missing_required_fields(tmp_path):
    """
    Return None when manifest is missing required fields.

    Validates manifest structure on load.
    """
    manifest_path = tmp_path / "index.manifest.json"
    manifest_path.write_text('{"spec_version": "1.0.0"}')

    loaded = load_manifest(tmp_path)

    assert loaded is None


def test_should_rebuild_when_no_saved_manifest(sample_manifest):
    """
    Trigger rebuild when no saved manifest exists.

    First-time indexing should always build.
    """
    result = should_rebuild(sample_manifest, None)

    assert result is True


def test_should_rebuild_when_spec_version_differs(sample_manifest):
    """
    Trigger rebuild when spec version changes.

    Version changes indicate incompatible index formats.
    """
    saved = IndexManifest(
        spec_version="0.9.0",
        embedding_model="all-MiniLM-L6-v2",
        parsers={"**/*.md": "MarkdownParser"},
    )

    result = should_rebuild(sample_manifest, saved)

    assert result is True


def test_should_rebuild_when_embedding_model_differs(sample_manifest):
    """
    Trigger rebuild when embedding model changes.

    Different models produce incompatible vectors.
    """
    saved = IndexManifest(
        spec_version="1.0.0",
        embedding_model="different-model",
        parsers={"**/*.md": "MarkdownParser"},
    )

    result = should_rebuild(sample_manifest, saved)

    assert result is True


def test_should_rebuild_when_parsers_differ(sample_manifest):
    """
    Trigger rebuild when parser configuration changes.

    Parser changes affect document processing.
    """
    saved = IndexManifest(
        spec_version="1.0.0",
        embedding_model="all-MiniLM-L6-v2",
        parsers={"**/*.txt": "TextParser"},
    )

    result = should_rebuild(sample_manifest, saved)

    assert result is True


def test_should_not_rebuild_when_manifests_identical(sample_manifest):
    """
    Skip rebuild when manifests are identical.

    Avoids unnecessary reindexing work.
    """
    saved = IndexManifest(
        spec_version="1.0.0",
        embedding_model="all-MiniLM-L6-v2",
        parsers={"**/*.md": "MarkdownParser"},
    )

    result = should_rebuild(sample_manifest, saved)

    assert result is False


def test_save_manifest_creates_parent_directory(tmp_path):
    """
    Create parent directories when saving manifest.

    Ensures manifest can be saved to nested paths.
    """
    nested_path = tmp_path / "nested" / "path"
    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model="test-model",
        parsers={},
    )

    save_manifest(nested_path, manifest)

    assert (nested_path / "index.manifest.json").exists()
