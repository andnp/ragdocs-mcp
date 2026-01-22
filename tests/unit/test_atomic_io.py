import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.atomic_io import (
    atomic_write_binary,
    atomic_write_json,
    atomic_write_text,
    fsync_path,
)


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path: Path):
        """
        Verifies atomic_write_json creates valid JSON file.
        """
        target = tmp_path / "test.json"
        data = {"key": "value", "nested": {"list": [1, 2, 3]}}

        atomic_write_json(target, data)

        assert target.exists()
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path: Path):
        """
        Verifies atomic_write_json creates parent directories.
        """
        target = tmp_path / "deep" / "nested" / "dir" / "test.json"

        atomic_write_json(target, {"data": True})

        assert target.exists()

    def test_overwrites_existing_file(self, tmp_path: Path):
        """
        Verifies atomic_write_json overwrites existing files atomically.
        """
        target = tmp_path / "test.json"
        target.write_text('{"old": "data"}')

        atomic_write_json(target, {"new": "data"})

        with open(target) as f:
            loaded = json.load(f)
        assert loaded == {"new": "data"}

    def test_cleans_up_temp_file_on_json_error(self, tmp_path: Path):
        """
        Verifies temp file is removed when JSON serialization fails.
        """
        target = tmp_path / "test.json"

        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(target, NotSerializable())

        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0
        assert not target.exists()

    def test_no_partial_writes_on_replace_failure(self, tmp_path: Path):
        """
        Verifies original file is not corrupted if replace fails.

        Simulates interrupt by patching os.replace to fail.
        """
        target = tmp_path / "test.json"
        original_data = {"original": "data"}
        target.write_text(json.dumps(original_data))

        with patch("src.utils.atomic_io.os.replace", side_effect=OSError("Simulated failure")):
            with pytest.raises(OSError):
                atomic_write_json(target, {"new": "data"})

        with open(target) as f:
            loaded = json.load(f)
        assert loaded == original_data


class TestAtomicWriteText:
    def test_writes_text_content(self, tmp_path: Path):
        """
        Verifies atomic_write_text creates text file with correct content.
        """
        target = tmp_path / "test.txt"
        content = "Hello, World!\nLine 2"

        atomic_write_text(target, content)

        assert target.read_text() == content

    def test_handles_unicode(self, tmp_path: Path):
        """
        Verifies atomic_write_text handles Unicode correctly.
        """
        target = tmp_path / "test.txt"
        content = "Unicode: 你好世界 🎉"

        atomic_write_text(target, content)

        assert target.read_text(encoding="utf-8") == content

    def test_creates_parent_directories(self, tmp_path: Path):
        """
        Verifies atomic_write_text creates parent directories.
        """
        target = tmp_path / "a" / "b" / "c" / "test.txt"

        atomic_write_text(target, "content")

        assert target.exists()
        assert target.read_text() == "content"


class TestAtomicWriteBinary:
    def test_writes_binary_content(self, tmp_path: Path):
        """
        Verifies atomic_write_binary creates binary file.
        """
        target = tmp_path / "test.bin"
        data = b"\x00\x01\x02\xff\xfe"

        atomic_write_binary(target, data)

        assert target.read_bytes() == data

    def test_creates_parent_directories(self, tmp_path: Path):
        """
        Verifies atomic_write_binary creates parent directories.
        """
        target = tmp_path / "x" / "y" / "test.bin"

        atomic_write_binary(target, b"data")

        assert target.exists()


class TestFsyncPath:
    def test_fsyncs_existing_file(self, tmp_path: Path):
        """
        Verifies fsync_path works on existing files.
        """
        target = tmp_path / "test.txt"
        target.write_text("content")

        fsync_path(target)

    def test_raises_on_missing_file(self, tmp_path: Path):
        """
        Verifies fsync_path raises FileNotFoundError for missing files.
        """
        target = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            fsync_path(target)


class TestFsyncParameter:
    def test_skips_fsync_when_disabled(self, tmp_path: Path):
        """
        Verifies fsync can be disabled for performance testing.
        """
        target = tmp_path / "test.json"

        with patch("src.utils.atomic_io.os.fsync") as mock_fsync:
            atomic_write_json(target, {"data": True}, fsync=False)
            mock_fsync.assert_not_called()

    def test_calls_fsync_by_default(self, tmp_path: Path):
        """
        Verifies fsync is called by default.
        """
        target = tmp_path / "test.json"

        with patch("src.utils.atomic_io.os.fsync") as mock_fsync:
            atomic_write_json(target, {"data": True}, fsync=True)
            mock_fsync.assert_called_once()
