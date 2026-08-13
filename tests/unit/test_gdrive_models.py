"""Tests for typed Google Drive provider models."""

from mcp_markdown_ragdocs.gdrive.models import (
    DriveChange,
    DriveFile,
    DriveScope,
    DriveWorkspace,
)


def test_drive_file_parses_provider_metadata_and_shortcut() -> None:
    """
    Preserve the provider fields needed for stable identity and later mapping.
    Normalize numeric size and nested shortcut metadata into typed values.
    """
    file = DriveFile.from_api(
        {
            "id": "file-1",
            "name": "Notes",
            "mimeType": "text/plain",
            "size": "12",
            "parents": ["folder-1"],
            "shortcutDetails": {"targetId": "target-1", "targetMimeType": "application/pdf"},
        }
    )

    assert file.id == "file-1"
    assert file.size == 12
    assert file.parents == ("folder-1",)
    assert file.shortcut_target_id == "target-1"


def test_drive_change_uses_file_id_when_file_body_is_missing() -> None:
    """
    Retain removal changes even when Drive omits the deleted file body.
    """
    change = DriveChange.from_api({"fileId": "file-1", "removed": True})

    assert change.file_id == "file-1"
    assert change.removed is True
    assert change.file is None


def test_drive_workspace_keeps_scope_identity_separate() -> None:
    """
    Represent one workspace with explicit scopes without conflating identities.
    """
    scope = DriveScope("workspace-1", shared_drive_id="drive-1")
    workspace = DriveWorkspace("workspace-1", scopes=(scope,))

    assert workspace.workspace_id == scope.workspace_id
    assert scope.is_shared_drive is True
