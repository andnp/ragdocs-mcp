I've created the detailed plan in `docs/plans/sqlite-refactor.md`.

**Plan Summary: SQLite-based Coordination**

1.  **Architecture**: Move from file-based snapshots to a shared **SQLite database (`index.db`)** in WAL mode.
    -   **Main Process**: Reads directly from DB (FTS5) and loads vectors into memory for FAISS. Zero-latency startup.
    -   **Worker Process**: Polls a DB-backed task queue to handle parsing/embedding in the background.
2.  **Concurrency**: Leverages SQLite's Multi-Version Concurrency Control (MVCC) to allow the worker to write while the server reads.
3.  **Schema**:
    -   `documents`: File metadata.
    -   `chunks`: Text + Vector BLOBs.
    -   `tasks`: Indexing job queue.
    -   `search_index`: FTS5 virtual table (replaces Whoosh).

**Next Steps:**
1.  Scaffold `src/storage/db.py` to handle the SQLite connection and schema.
2.  Prototype the `tasks` table integration.
