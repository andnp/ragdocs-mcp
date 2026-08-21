from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from mcp_markdown_ragdocs import cli as cli_module


def _fake_runtime_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )


def _write_bloated_db(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE t(x TEXT)")
    connection.executemany(
        "INSERT INTO t VALUES (?)", [(f"row-{i}" * 200,) for i in range(2000)]
    )
    connection.execute("DELETE FROM t WHERE rowid % 2 = 0")
    connection.commit()
    connection.close()


def test_vacuum_enable_requires_yes() -> None:
    runner = CliRunner()

    result = runner.invoke(cli_module.cli, ["records", "vacuum-enable"])

    assert result.exit_code == 2
    assert "Refusing to vacuum without --yes." in result.output


def test_vacuum_enable_refuses_while_daemon_running(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli_module.RuntimePaths, "resolve", lambda: _fake_runtime_paths(tmp_path)
    )
    monkeypatch.setattr(
        cli_module, "inspect_daemon", lambda paths=None: SimpleNamespace(running=True)
    )
    runner = CliRunner()

    result = runner.invoke(cli_module.cli, ["records", "vacuum-enable", "--yes"])

    assert result.exit_code == 1
    assert "stop the daemon first" in result.output


def test_vacuum_enable_errors_when_db_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli_module.RuntimePaths, "resolve", lambda: _fake_runtime_paths(tmp_path)
    )
    monkeypatch.setattr(
        cli_module, "inspect_daemon", lambda paths=None: SimpleNamespace(running=False)
    )
    runner = CliRunner()

    result = runner.invoke(cli_module.cli, ["records", "vacuum-enable", "--yes"])

    assert result.exit_code == 1
    assert "index database not found" in result.output


def test_vacuum_enable_migrates_auto_vacuum_and_shrinks_file(
    monkeypatch, tmp_path
) -> None:
    runtime_paths = _fake_runtime_paths(tmp_path)
    _write_bloated_db(runtime_paths.index_db_path)
    before_size = runtime_paths.index_db_path.stat().st_size

    monkeypatch.setattr(cli_module.RuntimePaths, "resolve", lambda: runtime_paths)
    monkeypatch.setattr(
        cli_module, "inspect_daemon", lambda paths=None: SimpleNamespace(running=False)
    )
    runner = CliRunner()

    result = runner.invoke(cli_module.cli, ["records", "vacuum-enable", "--yes"])

    assert result.exit_code == 0
    assert "auto_vacuum: 0 -> 2 (incremental)" in result.output

    after_size = runtime_paths.index_db_path.stat().st_size
    assert after_size < before_size

    connection = sqlite3.connect(str(runtime_paths.index_db_path))
    try:
        assert connection.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    finally:
        connection.close()
