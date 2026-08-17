import sqlite3
from pathlib import Path

from mcp_markdown_ragdocs.daemon.storage_diagnostics import sqlite_storage_diagnostics


def test_sqlite_storage_diagnostics_reports_read_only_database_details(
    tmp_path: Path,
) -> None:
    """
    Report SQLite pragmas, sidecar sizes, and filesystem capacity for a database.
    """
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE records (body TEXT)")
        connection.execute("INSERT INTO records VALUES ('body')")

    payload = sqlite_storage_diagnostics(db_path)

    assert payload["status"] == "ready"
    assert payload["exists"] is True
    db_size = payload["db_size_bytes"]
    wal_size = payload["wal_size_bytes"]
    sqlite_details = payload["sqlite"]
    filesystem = payload["filesystem"]
    assert isinstance(db_size, int) and db_size > 0
    assert isinstance(wal_size, int) and wal_size >= 0
    assert isinstance(sqlite_details, dict)
    assert sqlite_details["journal_mode"] == "wal"
    page_count = sqlite_details["page_count"]
    assert isinstance(page_count, int) and page_count > 0
    assert isinstance(filesystem, dict)
    free_bytes = filesystem["free_bytes"]
    assert isinstance(free_bytes, int) and free_bytes > 0


def test_sqlite_storage_diagnostics_reports_missing_database(tmp_path: Path) -> None:
    """
    Identify an absent database without creating it as a diagnostic side effect.
    """
    db_path = tmp_path / "missing.db"

    payload = sqlite_storage_diagnostics(db_path)

    assert payload["status"] == "missing"
    assert payload["exists"] is False
    assert not db_path.exists()
