# Plan: SQLite-based Coordination & Storage Refactor

## 1. Goal
Replace the complex, file-based **Index Snapshot System** with a **SQLite-based architecture**. This allows the MCP server to start up immediately and serve queries while a background worker process handles heavy indexing tasks (parsing, embedding), coordinating via a shared database.

## 2. Motivation
- **Parallelism:** The user requires the MCP server to be responsive immediately (Main Process) while indexing happens in the background (Worker Process).
- **Complexity:** The current snapshot system involves copying entire index directories, causing high disk I/O and race conditions.
- **Robustness:** A database (SQLite) provides ACID transactions, simple locking, and a single source of truth, eliminating the fragile state synchronization logic.

## 3. Architecture

### 3.1. The Shared Database (`index.db`)
A single SQLite file in WAL (Write-Ahead Log) mode serves as the coordination hub and data store.

**Core Tables:**
1.  **`documents`**: Tracks file state (hash, mtime, status). Replaces `manifest.py`.
2.  **`chunks`**: Stores text, metadata, and **vectors** (as BLOBs).
3.  **`tasks`**: A persistent queue for indexing jobs (`index_file`, `remove_file`).
4.  **`kv_store`**: specific application state or simple key-value config.
5.  **`search_index` (FTS5)**: A virtual table for full-text keyword search. Replaces `Whoosh`.

### 3.2. Process Roles

#### **Main Process (MCP Server)**
- **Startup:** Opens `index.db`.
    - **Keyword Search:** Queries SQLite FTS5 directly (Zero startup time).
    - **Vector Search:** Loads vectors from `chunks` table into an in-memory FAISS index. (Fast sequential read).
- **Runtime:**
    - Serves queries.
    - Periodically checks DB for updates (e.g., `last_updated` timestamp) to refresh the in-memory FAISS index.
    - Adds "indexing tasks" to the DB when file changes are detected (if running in single-process mode, or if the watcher is in the main process).

#### **Worker Process**
- **Startup:** Connects to `index.db`.
- **Runtime:**
    - Polls `tasks` table for pending work.
    - **Action:** Reads file -> Parses -> Chunks -> Embeds.
    - **Write:** Opens a transaction to:
        1.  Update `chunks` table (text + vectors).
        2.  Update `search_index` (FTS5).
        3.  Update `documents` status.
        4.  Delete task.

## 4. Design Constraints
- **Startup Speed:** The Main Process must not block on indexing. It should be able to answer "No results" immediately if the DB is empty, rather than waiting for the first file to be indexed.
- **Concurrency:** Must use SQLite `WAL` mode to allow simultaneous readers (Main) and writers (Worker).
- **Memory:** FAISS index is kept in memory in the Main Process. We need to ensure we don't duplicate the vector data unnecessarily in Python objects if possible (using `sqlite-vec` in the future could be an option, but standard BLOB retrieval -> FAISS is a safe first step).

## 5. Implementation Steps

### Phase 1: Database Layer & Schema
1.  **Dependency:** Ensure `sqlite3` is available (standard lib).
2.  **`src/storage/db.py`**: Create a `DatabaseManager` class handling connections, schema migration, and WAL mode setup.
3.  **Schema Definition**: Define the SQL for the tables mentioned above.

### Phase 2: Refactor Indices
1.  **Keyword Index**: Rewrite `src/indices/keyword.py` to wrap SQL FTS5 queries instead of `Whoosh`.
2.  **Vector Index**: Rewrite `src/indices/vector.py`.
    -   *Write path:* Save vectors to DB blobs.
    -   *Read path:* Load vectors from DB into `faiss.IndexFlatIP`.
3.  **Document Store**: Update `src/indexing/manager.py` to write document metadata to SQLite instead of `manifest.json`.

### Phase 3: Task Queue & Worker
1.  **Task Queue**: Implement a simple DB-backed queue in `src/coordination/task_queue.py`.
2.  **Worker Logic**: Update `src/worker/process.py` to consume from this DB queue instead of the multiprocessing `Queue`.
3.  **Watcher**: Update `src/indexing/watcher.py` to simply insert rows into the `tasks` table.

### Phase 4: Cleanup
1.  Remove `src/ipc/` (Snapshot syncing logic).
2.  Remove `snapshots/` directory logic.
3.  Remove `Whoosh` dependency.

## 6. Next Steps
1.  Create `src/storage/` directory.
2.  Prototype the SQLite schema and connection manager.
3.  Test FTS5 performance on the current documentation set.
