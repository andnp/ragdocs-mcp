"""Test cross-process consistency via SQLite WAL mode.

Verifies that writes from one process are immediately visible to a reader
in a separate process without any explicit synchronization.
"""

import multiprocessing
from pathlib import Path

import pytest

from src.storage.db import DatabaseManager


def _writer(db_path: str, done_event) -> None:
    """Run in child process: write a kv_store row to the database."""
    db = DatabaseManager(Path(db_path))
    conn = db.get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        ("cross_proc_doc", "/test/cross.md"),
    )
    conn.commit()
    db.close()
    done_event.set()


def _kv_writer(db_path_str: str, process_id: int, done_event) -> None:
    """Run in child process: write kv_store entries."""
    db = DatabaseManager(Path(db_path_str))
    conn = db.get_connection()
    for i in range(50):
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (f"proc{process_id}_key{i}", f"value{i}"),
        )
        conn.commit()
    db.close()
    done_event.set()


@pytest.mark.serial
class TestCrossProcessConsistency:
    def test_writer_process_visible_to_reader(self, tmp_path: Path) -> None:
        """A kv_store row written by a child process is immediately visible
        to a reader on the main process without any sync calls."""
        db_path = tmp_path / "shared.db"
        db = DatabaseManager(db_path)

        # Pre-check: no kv row
        conn = db.get_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM kv_store WHERE key = ?", ("cross_proc_doc",)
        ).fetchone()[0]
        assert count == 0

        # Spawn writer in a separate process
        ctx = multiprocessing.get_context("spawn")
        done = ctx.Event()
        proc = ctx.Process(target=_writer, args=(str(db_path), done))
        proc.start()
        done.wait(timeout=15)
        proc.join(timeout=15)

        assert proc.exitcode == 0, (
            f"Writer process failed with exit code {proc.exitcode}"
        )

        # Reader: verify the row is visible WITHOUT any explicit sync
        row = conn.execute(
            "SELECT key, value FROM kv_store WHERE key = ?",
            ("cross_proc_doc",),
        ).fetchone()
        assert row is not None, (
            "Row written by child process not visible to reader"
        )
        assert row[0] == "cross_proc_doc"
        assert row[1] == "/test/cross.md"

        db.close()

    def test_concurrent_cross_process_writes(self, tmp_path: Path) -> None:
        """Multiple child processes can write concurrently without errors."""
        db_path = tmp_path / "concurrent.db"
        db = DatabaseManager(db_path)

        ctx = multiprocessing.get_context("spawn")
        events = [ctx.Event() for _ in range(3)]
        procs = [
            ctx.Process(target=_kv_writer, args=(str(db_path), pid, events[pid]))
            for pid in range(3)
        ]

        for p in procs:
            p.start()
        for e in events:
            e.wait(timeout=30)
        for p in procs:
            p.join(timeout=30)

        for p in procs:
            assert p.exitcode == 0

        conn = db.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
        assert count == 150  # 3 processes × 50 keys

        db.close()
