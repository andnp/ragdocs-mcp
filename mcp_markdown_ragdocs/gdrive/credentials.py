"""Validation for ragdocs-owned Google Drive credential files."""

import os
import stat
from pathlib import Path


def validate_gdrive_credentials_path(
    credentials_path: str | Path,
    source_root: str | Path,
) -> Path:
    """Return a secure credential path outside the configured source tree."""
    configured_path = Path(credentials_path).expanduser()
    if not configured_path.is_absolute():
        configured_path = Path.cwd() / configured_path
    configured_path = configured_path.absolute()
    resolved_path = configured_path.resolve(strict=False)
    resolved_source_root = Path(source_root).expanduser().resolve()

    for candidate in (configured_path, resolved_path):
        try:
            candidate.relative_to(resolved_source_root)
        except ValueError:
            continue
        raise ValueError(
            f"Google Drive credentials must be outside the source tree: {candidate}"
        )

    try:
        file_stat = resolved_path.stat()
    except OSError as error:
        raise ValueError(
            f"Google Drive credentials are not readable: {resolved_path}"
        ) from error

    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(
            f"Google Drive credentials must be a regular file: {resolved_path}"
        )
    if not (file_stat.st_mode & stat.S_IRUSR) or not os.access(
        resolved_path, os.R_OK
    ):
        raise ValueError(
            f"Google Drive credentials are not readable: {resolved_path}"
        )
    if file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(
            "Google Drive credentials must be owner-only: "
            f"{resolved_path}"
        )

    return resolved_path
