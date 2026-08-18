"""Verify the runtime factory forwards gate concurrency settings from config."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

import mcp_markdown_ragdocs.gdrive.gate as gate_module
from mcp_markdown_ragdocs.app.runtime_factory import build_gdrive_source
from mcp_markdown_ragdocs.config import GoogleDriveConfig, load_config


def test_build_gdrive_source_forwards_gate_settings_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_gdrive_source must construct DriveRequestGate with the
    configured min_interval_seconds and max_concurrent, not the class's
    own conservative defaults.
    """
    credentials_path = tmp_path / "creds" / "authorized-user.json"
    credentials_path.parent.mkdir()
    credentials_path.write_text("{}")
    credentials_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    source_root = tmp_path / "source"
    source_root.mkdir()
    index_path = tmp_path / "index"
    index_path.mkdir()

    real_gate = gate_module.DriveRequestGate
    captured_kwargs: dict[str, Any] = {}

    class _SpyGate(real_gate):  # type: ignore[misc, valid-type]
        def __init__(self, path: Path, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(path, **kwargs)

    monkeypatch.setattr(gate_module, "DriveRequestGate", _SpyGate)  # type: ignore[attr-defined]

    config = load_config()
    config.gdrive = GoogleDriveConfig(
        enabled=True,
        credentials_path=str(credentials_path),
        request_min_interval_seconds=0.05,
        request_max_concurrent=7,
    )

    build_gdrive_source(config, source_root=source_root, index_path=index_path)

    assert captured_kwargs == {"min_interval_seconds": 0.05, "max_concurrent": 7}
