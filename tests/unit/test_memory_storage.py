"""
Unit tests for memory storage path resolution and utilities.

Tests path resolution for "project" and "user" storage strategies,
memory ID computation, and directory management.
"""

from pathlib import Path


from src.memory.storage import (
    compute_memory_id,
    ensure_memory_dirs,
    get_indices_path,
    get_memory_file_path,
    get_trash_path,
    list_memory_files,
)


# ============================================================================
# ensure_memory_dirs Tests
# ============================================================================


class TestEnsureMemoryDirs:

    def test_creates_memory_directory(self, tmp_path: Path):
        """
        Verify ensure_memory_dirs creates the main memory directory.
        """
        memory_path = tmp_path / ".memories"
        assert not memory_path.exists()

        ensure_memory_dirs(memory_path)

        assert memory_path.exists()
        assert memory_path.is_dir()

    def test_creates_indices_subdirectory(self, tmp_path: Path):
        """
        Verify ensure_memory_dirs creates the indices subdirectory.

        The indices directory stores vector, keyword, and graph indices.
        """
        memory_path = tmp_path / ".memories"

        ensure_memory_dirs(memory_path)

        indices_path = memory_path / "indices"
        assert indices_path.exists()
        assert indices_path.is_dir()

    def test_creates_trash_subdirectory(self, tmp_path: Path):
        """
        Verify ensure_memory_dirs creates the .trash subdirectory.

        Deleted memories are moved to .trash for safety (soft delete).
        """
        memory_path = tmp_path / ".memories"

        ensure_memory_dirs(memory_path)

        trash_path = memory_path / ".trash"
        assert trash_path.exists()
        assert trash_path.is_dir()

    def test_idempotent_when_dirs_exist(self, tmp_path: Path):
        """
        Verify ensure_memory_dirs is idempotent.

        Calling it multiple times should not raise errors.
        """
        memory_path = tmp_path / ".memories"

        ensure_memory_dirs(memory_path)
        ensure_memory_dirs(memory_path)
        ensure_memory_dirs(memory_path)

        assert memory_path.exists()
        assert (memory_path / "indices").exists()
        assert (memory_path / ".trash").exists()

    def test_creates_nested_parent_directories(self, tmp_path: Path):
        """
        Verify ensure_memory_dirs creates nested parent directories.

        Similar to 'mkdir -p' behavior.
        """
        memory_path = tmp_path / "deep" / "nested" / "path" / ".memories"

        ensure_memory_dirs(memory_path)

        assert memory_path.exists()


# ============================================================================
# get_memory_file_path Tests
# ============================================================================


class TestGetMemoryFilePath:

    def test_returns_path_with_md_extension(self, tmp_path: Path):
        """
        Verify get_memory_file_path adds .md extension if missing.
        """
        result = get_memory_file_path(tmp_path, "my-note")

        assert result == tmp_path / "my-note.md"

    def test_preserves_existing_md_extension(self, tmp_path: Path):
        """
        Verify get_memory_file_path doesn't double .md extension.
        """
        result = get_memory_file_path(tmp_path, "my-note.md")

        assert result == tmp_path / "my-note.md"

    def test_handles_filename_with_spaces(self, tmp_path: Path):
        """
        Verify get_memory_file_path handles filenames with spaces.
        """
        result = get_memory_file_path(tmp_path, "my note file")

        assert result == tmp_path / "my note file.md"

    def test_handles_filename_with_subdirectory(self, tmp_path: Path):
        """
        Verify get_memory_file_path handles subdirectory in filename.
        """
        result = get_memory_file_path(tmp_path, "subdir/note")

        assert result == tmp_path / "subdir/note.md"


# ============================================================================
# get_trash_path Tests
# ============================================================================


class TestGetTrashPath:

    def test_returns_trash_subdirectory(self, tmp_path: Path):
        """
        Verify get_trash_path returns .trash subdirectory.
        """
        result = get_trash_path(tmp_path)

        assert result == tmp_path / ".trash"


# ============================================================================
# get_indices_path Tests
# ============================================================================


class TestGetIndicesPath:

    def test_returns_indices_subdirectory(self, tmp_path: Path):
        """
        Verify get_indices_path returns indices subdirectory.
        """
        result = get_indices_path(tmp_path)

        assert result == tmp_path / "indices"


# ============================================================================
# list_memory_files Tests
# ============================================================================


class TestListMemoryFiles:

    def test_returns_empty_when_no_files(self, tmp_path: Path):
        """
        Verify list_memory_files returns empty list for empty directory.
        """
        result = list_memory_files(tmp_path)

        assert result == []

    def test_returns_empty_when_path_not_exists(self, tmp_path: Path):
        """
        Verify list_memory_files returns empty list for non-existent path.

        This provides graceful handling before memory directory is created.
        """
        nonexistent = tmp_path / "does_not_exist"

        result = list_memory_files(nonexistent)

        assert result == []

    def test_lists_markdown_files(self, tmp_path: Path):
        """
        Verify list_memory_files finds .md files in directory.
        """
        (tmp_path / "note1.md").write_text("Note 1")
        (tmp_path / "note2.md").write_text("Note 2")

        result = list_memory_files(tmp_path)

        filenames = [f.name for f in result]
        assert "note1.md" in filenames
        assert "note2.md" in filenames
        assert len(result) == 2

    def test_excludes_hidden_files(self, tmp_path: Path):
        """
        Verify list_memory_files excludes files starting with dot.

        Hidden files like .metadata.md should not be listed.
        """
        (tmp_path / "visible.md").write_text("Visible")
        (tmp_path / ".hidden.md").write_text("Hidden")

        result = list_memory_files(tmp_path)

        filenames = [f.name for f in result]
        assert "visible.md" in filenames
        assert ".hidden.md" not in filenames

    def test_excludes_non_markdown_files(self, tmp_path: Path):
        """
        Verify list_memory_files only returns .md files.
        """
        (tmp_path / "note.md").write_text("Note")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "script.py").write_text("print('hi')")

        result = list_memory_files(tmp_path)

        filenames = [f.name for f in result]
        assert "note.md" in filenames
        assert "data.json" not in filenames
        assert "script.py" not in filenames

    def test_excludes_directories(self, tmp_path: Path):
        """
        Verify list_memory_files excludes directories.
        """
        (tmp_path / "note.md").write_text("Note")
        subdir = tmp_path / "subdir.md"
        subdir.mkdir()

        result = list_memory_files(tmp_path)

        filenames = [f.name for f in result]
        assert "note.md" in filenames
        assert "subdir.md" not in filenames


# ============================================================================
# compute_memory_id Tests
# ============================================================================


class TestComputeMemoryId:

    def test_computes_id_from_relative_path(self, tmp_path: Path):
        """
        Verify compute_memory_id creates memory: prefixed ID from relative path.

        The ID is based on the file path relative to memory_path, without extension.
        """
        memory_path = tmp_path / ".memories"
        memory_path.mkdir()
        file_path = memory_path / "my-note.md"

        result = compute_memory_id(memory_path, file_path)

        assert result == "memory:my-note"

    def test_strips_md_extension(self, tmp_path: Path):
        """
        Verify compute_memory_id removes .md extension from ID.
        """
        memory_path = tmp_path / ".memories"
        memory_path.mkdir()
        file_path = memory_path / "test-file.md"

        result = compute_memory_id(memory_path, file_path)

        assert result == "memory:test-file"
        assert ".md" not in result

    def test_handles_nested_file(self, tmp_path: Path):
        """
        Verify compute_memory_id handles files in subdirectories.
        """
        memory_path = tmp_path / ".memories"
        subdir = memory_path / "2025" / "january"
        subdir.mkdir(parents=True)
        file_path = subdir / "daily-note.md"

        result = compute_memory_id(memory_path, file_path)

        assert result == "memory:2025/january/daily-note"

    def test_handles_file_outside_memory_path(self, tmp_path: Path):
        """
        Verify compute_memory_id handles files outside memory_path gracefully.

        Falls back to using just the stem of the file.
        """
        memory_path = tmp_path / ".memories"
        memory_path.mkdir()
        file_path = tmp_path / "external" / "file.md"

        result = compute_memory_id(memory_path, file_path)

        assert result == "memory:file"

    def test_id_is_consistent_for_same_file(self, tmp_path: Path):
        """
        Verify compute_memory_id returns consistent ID for the same file.

        IDs must be deterministic for index updates.
        """
        memory_path = tmp_path / ".memories"
        memory_path.mkdir()
        file_path = memory_path / "consistent.md"

        id1 = compute_memory_id(memory_path, file_path)
        id2 = compute_memory_id(memory_path, file_path)
        id3 = compute_memory_id(memory_path, file_path)

        assert id1 == id2 == id3
