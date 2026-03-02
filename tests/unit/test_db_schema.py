import sqlite3
import threading
from pathlib import Path

import pytest

from src.storage.db import DatabaseManager


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table') ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


class TestSchemaAllTablesExist:
    def test_all_expected_tables_present(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        tables = _table_names(conn)
        expected = {
            "documents",
            "chunks",
            "kv_store",
            "search_index",
            "graph_nodes",
            "graph_edges",
            "tasks",
            "system1_journal",
            "system_state",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"


class TestGraphNodesColumns:
    def test_columns(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        cols = _column_names(conn, "graph_nodes")
        assert "node_id" in cols
        assert "metadata" in cols

    def test_node_id_is_primary_key(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        rows = conn.execute("PRAGMA table_info(graph_nodes)").fetchall()
        pk_col = [r for r in rows if r[5] == 1]  # pk column index
        assert len(pk_col) == 1
        assert pk_col[0][1] == "node_id"


class TestGraphEdgesColumns:
    def test_columns(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        cols = _column_names(conn, "graph_edges")
        assert "source" in cols
        assert "target" in cols
        assert "edge_type" in cols
        assert "edge_context" in cols


class TestGraphEdgesCompositePK:
    def test_duplicate_exact_edge_raises(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('A')")
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('B')")
        conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) VALUES ('A', 'B', 'link')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO graph_edges (source, target, edge_type) VALUES ('A', 'B', 'link')"
            )

    def test_different_edge_type_on_same_pair_succeeds(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('A')")
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('B')")
        conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) VALUES ('A', 'B', 'link')"
        )
        conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) VALUES ('A', 'B', 'sibling')"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        assert count == 2


class TestGraphEdgesCascadeDelete:
    def test_deleting_source_node_removes_edges(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('A')")
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('B')")
        conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) VALUES ('A', 'B', 'link')"
        )
        conn.commit()
        conn.execute("DELETE FROM graph_nodes WHERE node_id = 'A'")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        assert count == 0

    def test_deleting_target_node_removes_edges(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('A')")
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('B')")
        conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type) VALUES ('A', 'B', 'link')"
        )
        conn.commit()
        conn.execute("DELETE FROM graph_nodes WHERE node_id = 'B'")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        assert count == 0


class TestGraphEdgesIndexes:
    def test_indexes_exist(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        indexes = _index_names(conn)
        assert "idx_graph_edges_source" in indexes
        assert "idx_graph_edges_target" in indexes
        assert "idx_graph_edges_type" in indexes


class TestTasksTable:
    def test_columns(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        cols = _column_names(conn, "tasks")
        for col in ("id", "task_name", "data", "status", "created_at", "updated_at"):
            assert col in cols, f"Missing column: {col}"

    def test_default_status_is_pending(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO tasks (id, task_name, created_at) VALUES ('t1', 'test_task', 1000.0)"
        )
        conn.commit()
        row = conn.execute("SELECT status FROM tasks WHERE id = 't1'").fetchone()
        assert row[0] == "pending"

    def test_status_index_exists(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        indexes = _index_names(conn)
        assert "idx_tasks_status" in indexes


class TestSystem1Journal:
    def test_autoincrement(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp) VALUES ('a', 1.0)"
        )
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp) VALUES ('b', 2.0)"
        )
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp) VALUES ('c', 3.0)"
        )
        # Delete the middle row
        conn.execute("DELETE FROM system1_journal WHERE id = 2")
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp) VALUES ('d', 4.0)"
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM system1_journal ORDER BY id DESC LIMIT 1"
        ).fetchone()
        # AUTOINCREMENT ensures new id > max previous id (which was 3)
        assert row[0] > 3

    def test_default_status(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp) VALUES ('test', 1.0)"
        )
        conn.commit()
        row = conn.execute("SELECT status FROM system1_journal WHERE id = 1").fetchone()
        assert row[0] == "pending"


class TestSystemState:
    def test_upsert(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES ('k1', 'v1')"
        )
        conn.commit()
        row = conn.execute("SELECT value FROM system_state WHERE key = 'k1'").fetchone()
        assert row[0] == "v1"

        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES ('k1', 'v2')"
        )
        conn.commit()
        row = conn.execute("SELECT value FROM system_state WHERE key = 'k1'").fetchone()
        assert row[0] == "v2"


class TestConcurrentWAL:
    def test_concurrent_writes_no_locking_errors(self, db: DatabaseManager) -> None:
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                conn = db.get_connection()
                for i in range(200):
                    conn.execute(
                        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                        (f"thread{thread_id}_key{i}", f"value{i}"),
                    )
                    conn.commit()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent writes: {errors}"

        conn = db.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
        assert count == 1000

    def test_readers_dont_block_writers(self, db: DatabaseManager) -> None:
        read_errors: list[Exception] = []
        write_errors: list[Exception] = []

        def writer() -> None:
            try:
                conn = db.get_connection()
                for i in range(500):
                    conn.execute(
                        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                        (f"key{i}", f"value{i}"),
                    )
                    conn.commit()
            except Exception as e:
                write_errors.append(e)

        def reader(thread_id: int) -> None:
            try:
                conn = db.get_connection()
                for _ in range(500):
                    conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()
            except Exception as e:
                read_errors.append(e)

        writer_t = threading.Thread(target=writer)
        reader_ts = [threading.Thread(target=reader, args=(i,)) for i in range(3)]

        writer_t.start()
        for t in reader_ts:
            t.start()

        writer_t.join(timeout=30)
        for t in reader_ts:
            t.join(timeout=30)

        assert not write_errors, f"Writer errors: {write_errors}"
        assert not read_errors, f"Reader errors: {read_errors}"

        conn = db.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
        assert count == 500


class TestForeignKeysEnabled:
    def test_pragma_returns_one(self, db: DatabaseManager) -> None:
        conn = db.get_connection()
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert result == 1


class TestSchemaIdempotent:
    def test_double_init_no_error(self, db: DatabaseManager) -> None:
        db.initialize_schema()
        db.initialize_schema()
        conn = db.get_connection()
        conn.execute("INSERT INTO graph_nodes (node_id) VALUES ('test')")
        conn.commit()
        row = conn.execute(
            "SELECT node_id FROM graph_nodes WHERE node_id = 'test'"
        ).fetchone()
        assert row[0] == "test"


class TestFreshDbOnTmpPath:
    def test_fresh_db(self, tmp_path: Path) -> None:
        db = DatabaseManager(tmp_path / "subdir" / "fresh.db")
        conn = db.get_connection()
        tables = _table_names(conn)
        assert "graph_nodes" in tables
        assert "graph_edges" in tables
        assert "tasks" in tables
        assert "system1_journal" in tables
        assert "system_state" in tables
