import json
import sqlite3
from pathlib import Path

from mcp_markdown_ragdocs.indexing.record_ports import SqliteSourceMapStore


class _SingleConnectionProvider:
    """Minimal real-connection SQLiteConnectionProvider for tests."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(str(path))

    def get_connection(self) -> sqlite3.Connection:
        return self._connection


def _write_legacy_json(path: Path, records: dict[str, list[str]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_sqlite_source_map_store_round_trips_membership(tmp_path: Path) -> None:
    """Preserve source membership across the sqlite-backed storage boundary."""
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider)
    records = {"source-1": ("key-a", "key-b")}

    store.save(records)

    assert store.load() == {"source-1": ["key-a", "key-b"]}


def test_sqlite_source_map_store_save_replaces_prior_snapshot(tmp_path: Path) -> None:
    """A later save() fully replaces membership rather than merging with it."""
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider)
    store.save({"source-1": ["key-a"], "source-2": ["key-b"]})

    store.save({"source-1": ["key-a"]})

    assert store.load() == {"source-1": ["key-a"]}


def test_sqlite_source_map_store_applies_targeted_delta(tmp_path: Path) -> None:
    """
    Update selected sources while preserving unrelated membership rows.
    """
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider)
    store.save({"source-1": ["key-a"], "source-2": ["key-b"]})

    store.apply_delta({"source-1": ["key-c"]}, ["source-2"])

    assert store.load() == {"source-1": ["key-c"]}


def test_sqlite_source_map_store_applies_delta_as_one_transaction(
    tmp_path: Path,
) -> None:
    """
    Leave prior membership intact when a delta cannot be committed.
    """
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider)
    store.save({"source-1": ["key-a"]})
    connection = provider.get_connection()
    connection.execute(
        """
        CREATE TRIGGER reject_source_2 BEFORE INSERT ON source_map
        WHEN NEW.doc_id = 'source-2'
        BEGIN SELECT RAISE(ABORT, 'reject'); END
        """
    )

    try:
        store.apply_delta({"source-1": ["key-c"], "source-2": ["key-b"]}, [])
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("expected the delta transaction to fail")

    assert store.load() == {"source-1": ["key-a"]}


def test_sqlite_source_map_store_skips_a_malformed_row(tmp_path: Path) -> None:
    """Skip a row with malformed keys_json rather than failing the whole load."""
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider)
    connection = provider.get_connection()
    connection.execute(
        "INSERT INTO source_map (doc_id, keys_json) VALUES (?, ?)",
        ("broken", "not-json"),
    )
    connection.execute(
        "INSERT INTO source_map (doc_id, keys_json) VALUES (?, ?)",
        ("ok", '["key-a"]'),
    )
    connection.commit()

    assert store.load() == {"ok": ["key-a"]}


def test_sqlite_source_map_store_degrades_gracefully_when_table_unavailable(
    tmp_path: Path,
) -> None:
    """Treat a locked or missing backing table as empty membership."""
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider)
    provider.get_connection().execute("DROP TABLE source_map")

    assert store.load() == {}


def test_sqlite_source_map_store_migrates_legacy_json_once(tmp_path: Path) -> None:
    """Import an existing record-sources.json into the table on first use."""
    legacy_path = tmp_path / "record-sources.json"
    _write_legacy_json(legacy_path, {"source-1": ["key-a"]})
    provider = _SingleConnectionProvider(tmp_path / "index.db")

    store = SqliteSourceMapStore(provider, legacy_path)

    assert store.load() == {"source-1": ["key-a"]}


def test_sqlite_source_map_store_migration_runs_only_once(tmp_path: Path) -> None:
    """Never re-import the legacy file on a later construction, even if edited."""
    legacy_path = tmp_path / "record-sources.json"
    _write_legacy_json(legacy_path, {"source-1": ["key-a"]})
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    SqliteSourceMapStore(provider, legacy_path)

    _write_legacy_json(legacy_path, {"source-1": ["key-z"]})
    second_store = SqliteSourceMapStore(provider, legacy_path)

    assert second_store.load() == {"source-1": ["key-a"]}


def test_sqlite_source_map_store_does_not_remigrate_an_emptied_table(
    tmp_path: Path,
) -> None:
    """A table legitimately emptied afterward (e.g. clear_documents) stays empty."""
    legacy_path = tmp_path / "record-sources.json"
    _write_legacy_json(legacy_path, {"source-1": ["key-a"]})
    provider = _SingleConnectionProvider(tmp_path / "index.db")
    store = SqliteSourceMapStore(provider, legacy_path)
    store.save({})

    second_store = SqliteSourceMapStore(provider, legacy_path)

    assert second_store.load() == {}
