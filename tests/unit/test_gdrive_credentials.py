"""Tests for Google Drive credential path validation."""

import os
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.gdrive.credentials import validate_gdrive_credentials_path


def _write_credentials(path: Path, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"type":"authorized_user"}', encoding="utf-8")
    os.chmod(path, mode)
    return path


def test_validates_owner_readable_credentials_outside_source_tree(tmp_path: Path):
    """
    Accept a regular owner-only credential file outside the source tree.
    Return its resolved path for the later session boundary.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    credentials = _write_credentials(tmp_path / "state" / "credentials.json")

    result = validate_gdrive_credentials_path(credentials, source_root)

    assert result == credentials.resolve()


def test_rejects_credentials_inside_source_tree(tmp_path: Path):
    """
    Refuse credentials stored below the source tree even when readable.
    Keep secrets out of repository files and generated artifacts.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    credentials = _write_credentials(source_root / "credentials.json")

    with pytest.raises(ValueError, match="outside the source tree"):
        validate_gdrive_credentials_path(credentials, source_root)


def test_rejects_symlink_inside_source_tree(tmp_path: Path):
    """
    Refuse a source-tree symlink to an otherwise secure external file.
    The configured path itself must remain outside the source tree.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    credentials = _write_credentials(tmp_path / "state" / "credentials.json")
    link = source_root / "credentials.json"
    link.symlink_to(credentials)

    with pytest.raises(ValueError, match="outside the source tree"):
        validate_gdrive_credentials_path(link, source_root)


@pytest.mark.parametrize("mode", [0o640, 0o600 | 0o004])
def test_rejects_group_or_other_permissions(tmp_path: Path, mode: int):
    """
    Reject credential files readable by group or other users.
    Owner-only access is required even when the file is otherwise readable.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    credentials = _write_credentials(tmp_path / "credentials.json", mode)

    with pytest.raises(ValueError, match="owner-only"):
        validate_gdrive_credentials_path(credentials, source_root)


def test_rejects_missing_credentials(tmp_path: Path):
    """
    Report a missing configured credential file as unreadable.
    Avoid allowing the OAuth session to fail later with an opaque file error.
    """
    with pytest.raises(ValueError, match="not readable"):
        validate_gdrive_credentials_path(
            tmp_path / "missing.json",
            tmp_path / "source",
        )
