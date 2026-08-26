from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.config import LoggingConfig
from mcp_markdown_ragdocs.runtime_logging import configure_file_logging


def test_logging_config_rejects_invalid_retention() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        LoggingConfig(max_bytes=0)
    with pytest.raises(ValueError, match="backup_count"):
        LoggingConfig(backup_count=0)


def test_file_logging_retains_configured_backups(tmp_path: Path) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    log_path = tmp_path / "worker.log"

    try:
        configure_file_logging(
            log_path,
            LoggingConfig(max_bytes=128, backup_count=2),
        )
        test_logger = logging.getLogger("rotation-test")
        for index in range(10):
            test_logger.info("entry-%d %s", index, "x" * 180)
        for handler in root_logger.handlers:
            handler.flush()

        log_files = sorted(tmp_path.glob("worker.log*"))
        assert len(log_files) == 3
        assert log_path.exists()
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_file_logging_suppresses_googleapiclient_discovery_cache_noise(
    tmp_path: Path,
) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    noisy_logger = logging.getLogger("googleapiclient.discovery_cache")
    original_noisy_level = noisy_logger.level
    log_path = tmp_path / "worker.log"

    try:
        configure_file_logging(log_path, LoggingConfig())

        assert noisy_logger.getEffectiveLevel() == logging.WARNING
        # httpx must remain suppressed too -- this must be additive, not a replacement.
        assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
        # Suppression is targeted: unrelated loggers still inherit root's INFO level.
        assert logging.getLogger("googleapiclient.other").getEffectiveLevel() == logging.INFO

        noisy_logger.info("file_cache is only supported with oauth2client<4.0.0")
        noisy_logger.error("token refresh failed")
        for handler in root_logger.handlers:
            handler.flush()

        content = log_path.read_text(encoding="utf-8")
        assert "file_cache is only supported" not in content
        assert "token refresh failed" in content
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
        noisy_logger.setLevel(original_noisy_level)


@pytest.mark.parametrize(
    "message",
    [
        "File /tmp/file.md is outside configured document roots [/docs]",
        "File /tmp/file.md is outside the 15 configured document roots.",
    ],
)
def test_file_logging_coalesces_repeated_outside_root_warnings(
    tmp_path: Path, message: str
) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    log_path = tmp_path / "worker.log"

    try:
        configure_file_logging(log_path, LoggingConfig())
        warning_logger = logging.getLogger("searchkernel.search.path_utils")
        for _ in range(3):
            warning_logger.warning(message)
        for handler in root_logger.handlers:
            handler.flush()

        assert log_path.read_text(encoding="utf-8").count("document roots") == 1
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
