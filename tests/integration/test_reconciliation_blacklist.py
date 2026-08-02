"""Reconcile records that became excluded or disappeared from disk."""

import glob
import logging
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.config import Config, IndexingConfig, LLMConfig, SearchConfig
from tests.integration._canonical import make_record_index_manager


@pytest.fixture
def base_config(tmp_path: Path) -> Config:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
            include=["**/*"],
            exclude=["**/.venv/**", "**/node_modules/**", "**/build/**"],
            exclude_hidden_dirs=True,
        ),
        search=SearchConfig(),
        llm=LLMConfig(),
    )


@pytest.fixture
def manager(base_config: Config):
    return make_record_index_manager(base_config)


def _discover(config: Config) -> list[str]:
    from searchkernel.utils import should_include_file

    root = Path(config.indexing.documents_path)
    return [
        path
        for path in glob.glob(str(root / "**" / "*.md"), recursive=True)
        if should_include_file(
            path,
            config.indexing.include,
            config.indexing.exclude,
            config.indexing.exclude_hidden_dirs,
        )
    ]


def test_reconciliation_removes_newly_blacklisted_venv_files(
    base_config: Config, manager, tmp_path: Path
) -> None:
    del tmp_path
    docs_path = Path(base_config.indexing.documents_path)
    readme = docs_path / "README.md"
    flax = docs_path / ".venv" / "lib" / "flax" / "README.md"
    orbax = docs_path / ".venv" / "lib" / "orbax" / "README.md"
    flax.parent.mkdir(parents=True)
    orbax.parent.mkdir(parents=True)
    readme.write_text("# Project Documentation\n\nMain docs.")
    flax.write_text("# Flax\n\nML library.")
    orbax.write_text("# Orbax\n\nCheckpointing.")
    for path in (readme, flax, orbax):
        assert manager.index_document(str(path))
    manager.persist()

    result = manager.reconcile_indices(_discover(base_config), docs_path)

    assert result.removed_count == 2
    assert result.added_count == 0
    assert result.failed_count == 0


def test_reconciliation_handles_blacklist_config_change(
    base_config: Config, manager, tmp_path: Path
) -> None:
    docs_path = Path(base_config.indexing.documents_path)
    main = docs_path / "main.md"
    vendor = docs_path / "vendor" / "package" / "README.md"
    vendor.parent.mkdir(parents=True)
    main.write_text("# Main\n\nMain documentation.")
    vendor.write_text("# Vendor Package\n\nThird-party code.")
    for path in (main, vendor):
        assert manager.index_document(str(path))
    manager.persist()

    updated_config = Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "indices"),
            include=["**/*"],
            exclude=["**/.venv/**", "**/node_modules/**", "**/build/**", "**/vendor/**"],
            exclude_hidden_dirs=True,
        ),
        search=SearchConfig(),
        llm=LLMConfig(),
    )
    manager_new = make_record_index_manager(updated_config)
    manager_new.load()

    result = manager_new.reconcile_indices(_discover(updated_config), docs_path)

    assert result.removed_count == 1
    assert result.added_count == 0


def test_reconciliation_respects_exclude_hidden_dirs_change(
    base_config: Config, manager, tmp_path: Path
) -> None:
    del tmp_path
    docs_path = Path(base_config.indexing.documents_path)
    visible = docs_path / "visible.md"
    secret = docs_path / ".hidden" / "secret.md"
    cache = docs_path / ".cache" / "data.md"
    secret.parent.mkdir(parents=True)
    cache.parent.mkdir(parents=True)
    visible.write_text("# Visible\n\nPublic documentation.")
    secret.write_text("# Secret\n\nHidden content.")
    cache.write_text("# Cache Data\n\nTemporary data.")
    for path in (visible, secret, cache):
        assert manager.index_document(str(path))
    manager.persist()

    result = manager.reconcile_indices(_discover(base_config), docs_path)

    assert result.removed_count == 2
    assert result.added_count == 0


def test_reconciliation_logs_distinct_messages_for_excluded_vs_missing(
    base_config: Config, manager, caplog: pytest.LogCaptureFixture
) -> None:
    docs_path = Path(base_config.indexing.documents_path)
    active = docs_path / "active.md"
    excluded = docs_path / ".venv" / "lib" / "README.md"
    deleted = docs_path / "deleted.md"
    excluded.parent.mkdir(parents=True)
    active.write_text("# Active\n\nActive documentation.")
    excluded.write_text("# Package\n\nPackage docs.")
    deleted.write_text("# Deleted\n\nRemoved documentation.")
    for path in (active, excluded, deleted):
        assert manager.index_document(str(path))
    manager.persist()
    deleted.unlink()

    caplog.set_level(logging.INFO)
    result = manager.reconcile_indices(_discover(base_config), docs_path)

    assert result.removed_count == 2
    assert "excluded by pattern" in caplog.text
    assert "file missing" in caplog.text
