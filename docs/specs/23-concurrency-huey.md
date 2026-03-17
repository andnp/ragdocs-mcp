# Feature Spec 23: Concurrency Model & Task Queue (Huey)

**Status:** Draft
**Related:** [sqlite-refactor.md](../plans/sqlite-refactor.md), [../plans/04-daemon-zmq-control-plane-contract.md](../plans/04-daemon-zmq-control-plane-contract.md)

## 1. Overview
The current multiprocess architecture relies on Python's `multiprocessing.Queue` and a complex file-based snapshot system. When multiple MCP clients (e.g., VS Code extension and Claude Desktop) open the same project simultaneously, they spawn competing MCP server instances. If each spawns its own background worker, they contend for system resources (CPU/GPU for embeddings), file locks, and watch the same directories redundantly.

This specification replaces the transient multiprocessing queues with **Huey**, a robust, SQLite-backed task queue. This provides a unified coordination layer where multiple MCP instances can enqueue work, but only a single background worker processes it.

## 2. Goals
1.  **Multi-Instance Safety**: Allow multiple MCP servers to run concurrently on the same project without stepping on each other's toes.
2.  **Resource Efficiency**: Ensure only one worker process is active per project to conserve memory (embedding models) and CPU.
3.  **Persistence**: Task queues survive MCP server restarts. Pending indexing jobs are not lost.
4.  **Simplicity**: Replace complex IPC and snapshot synchronization with a simple database-backed queue.

## 3. Architecture

### 3.1. The Queue Database (`queue.db`)
We will integrate `huey` using its `SqliteHuey` storage backend. This keeps the deployment entirely local-first with zero external infrastructure (no Redis/RabbitMQ required).

-   **Location**: `.ragdocs/queue.db` (or `~/.local/share/mcp-markdown-ragdocs/projects/<hash>/queue.db`).
-   **Purpose**: Stores pending tasks, scheduled tasks (e.g., cron jobs for memory maintenance), and task results.

### 3.2. Process Roles

#### A. Main Process (The MCP Server)
-   **Responsibility**: Responds to MCP JSON-RPC queries instantly.
-   **Operation**: 
    -   Reads the core `index.db` (SQLite) to serve `query_documents` and `search_git_history`.
    -   Detects file changes (via its own lightweight watcher) or user commands, and enqueues tasks to `SqliteHuey`.
    -   **Leader Election**: Upon startup, it attempts to acquire an exclusive file lock (e.g., `worker.lock`). 
        -   If it acquires the lock, it spawns the background Huey Consumer (Worker Process).
        -   If it fails (lock held by another instance), it runs as a "Replica" and skips spawning the worker, confident that another instance is handling the queue.

#### B. Worker Process (The Huey Consumer)
-   **Responsibility**: Heavy lifting (parsing, chunking, embedding generation) and scheduled maintenance.
-   **Operation**:
    -   Runs the standard `huey_consumer` loop.
    -   Pulls tasks from `queue.db`.
    -   Executes tasks and writes the resulting indices/vectors to the shared `index.db`.
    -   Runs periodic cron tasks (e.g., memory consolidation).

## 4. Implementation Plan

### Phase 1: Huey Integration
1.  **Dependencies**: Add `huey` to `pyproject.toml`.
2.  **Configuration**: Create `src/coordination/queue.py` defining the `SqliteHuey` instance.
    ```python
    from huey import SqliteHuey
    huey = SqliteHuey(filename='queue.db')
    ```
3.  **Task Definitions**: Define task functions decorated with `@huey.task()`.
    -   `index_document_task(file_path: str)`
    -   `delete_document_task(file_path: str)`

### Phase 2: Refactor Worker & IPC
1.  **Replace Multiprocessing Queues**: Remove the custom `QueueManager` and `command_queue` / `response_queue` from `src/ipc/`.
2.  **Worker Entrypoint**: Update `src/worker/process.py` to simply invoke the Huey consumer programmatically:
    ```python
    from huey.bin.huey_consumer import consumer_main
    # Start consumer pointing to our huey instance
    ```

### Phase 3: Leader Election & Startup
1.  **File Lock**: Introduce `filelock` (or standard `fcntl`) in `LifecycleCoordinator`.
2.  **Startup Logic**:
    ```python
    try:
        lock = FileLock("worker.lock", timeout=0.1)
        lock.acquire()
        # We are the leader. Start the Huey consumer subprocess.
        start_huey_worker()
    except Timeout:
        # We are a replica. Do nothing, just use the queue.
        logger.info("Worker already running in another instance.")
    ```
3.  **Graceful Shutdown**: Ensure the lock is released when the primary MCP server shuts down. If the primary crashes, the OS releases the file lock, allowing the next MCP instance to take over.

## 5. Security & Safety
- **Database Locks**: SQLite handles its own internal locking. `index.db` (using WAL mode) allows concurrent readers (MCP servers) while the worker writes.
- **Task Idempotency**: Indexing tasks must be idempotent. If a file is queued twice, the second operation should cleanly overwrite the first without duplicating chunks.