# Plan: Remove Index Snapshot System & Multiprocess Worker

## 1. Goal

Simplify the architecture of `mcp-markdown-ragdocs` by removing the multiprocess worker and the file-based index snapshot system. The project will revert to a **single-process architecture** where the main application context handles both indexing (via asyncio/background tasks) and searching.

## 2. Motivation

The current multiprocess architecture (`Main Process` + `Worker Process`) synchronized via file-based snapshots was introduced to ensure zero-downtime indexing. However, it introduces significant complexity:
- **Code Overhead:** `src/ipc/`, `src/worker/`, `src/reader/context.py`, and complex synchronization logic.
- **Resource Usage:** Duplicate index loading (2x memory), disk usage for snapshots (2x storage).
- **Latency:** Polling intervals and file system events add delay to index updates.
- **Maintenance:** Harder to debug and test.

For a local-first tool, a well-optimized single-process model using `asyncio` and thread-safe indices (which we now have) is sufficient and far simpler.

## 3. Scope of Changes

### 3.1. Components to Remove
- **IPC Layer:** `src/ipc/` (IndexSyncPublisher, IndexSyncReceiver, QueueManager).
- **Worker Process:** `src/worker/` (WorkerProcess, WorkerState).
- **Reader Context:** `src/reader/context.py` (ReadOnlyContext).
- **Snapshot Logic:** All code related to creating, versioning, and polling `snapshots/` directories.
- **Documentation:** `docs/specs/22-index-snapshot-system.md`, `docs/specs/21-multiprocess-architecture.md`.

### 3.2. Components to Refactor
- **Application Context (`src/context.py`):** 
  - Ensure it's the sole entry point for the application.
  - Verify it correctly initializes `IndexManager` and `SearchOrchestrator` in the same process.
- **MCP Server (`src/mcp_server.py`):** 
  - Switch from `ReadOnlyContext` to `ApplicationContext`.
  - Remove `[tool.ragdocs.worker]` configuration.
- **Lifecycle (`src/lifecycle.py`):** 
  - Remove worker process management (start/stop/health checks).
  - Remove snapshot directory path resolution.

### 3.3. Configuration Changes
- Remove `[tool.ragdocs.worker]` section from `pyproject.toml` and `config.py`.
- Remove `worker` related fields from `Config` model.

## 4. Implementation Steps

### Step 1: Remove Tests (Validation First)
Delete tests that verify the to-be-removed components to prevent noise during refactoring.
- `tests/unit/test_index_sync.py`
- `tests/unit/test_reader_context.py`
- `tests/unit/test_ipc_commands.py`
- `tests/unit/test_queue_manager.py`
- `tests/unit/test_worker_progressive_snapshots.py`
- `tests/unit/test_lifecycle_worker_init.py`
- `tests/integration/test_progressive_snapshots.py`

### Step 2: Remove Code Modules
Delete the core modules responsible for multiprocess/snapshots.
- `rm -rf src/ipc/`
- `rm -rf src/worker/`
- `rm src/reader/context.py`

### Step 3: Refactor Core Application
- **`src/config.py`**: Remove `WorkerConfig` and `SnapshotConfig`.
- **`src/context.py`**: Ensure `ApplicationContext` is robust for single-process use.
- **`src/mcp_server.py`**: 
  - Import `ApplicationContext` instead of `ReadOnlyContext`.
  - Initialize `self.ctx` as `ApplicationContext`.
- **`src/lifecycle.py`**: Remove `WorkerProcess` startup logic.

### Step 4: Clean Up Index Manager
- **`src/indexing/manager.py`**: 
  - Remove `_snapshot_version`, `_publish_snapshot`, and any callback logic related to snapshots.
  - Ensure `persist()` saves directly to the live index directory (thread-safely).

### Step 5: Update Documentation
- Delete obsolete specs.
- Update `AGENTS.md` to reflect the single-process architecture.
- Update `README.md` if it mentions the worker process.

## 5. Risks & Mitigation

- **Risk:** Blocking search during indexing.
  - **Mitigation:** Ensure `VectorIndex` and other heavy operations release the GIL or use threads where appropriate. The current `VectorIndex` implementation already uses a read/write lock (`_index_lock`), which should be sufficient.
- **Risk:** Data corruption if crash during write.
  - **Mitigation:** Rely on existing atomic write patterns (temp file + rename) used in `persist()`.

## 6. Verification
- Run `pytest` to ensure no regressions in core search/indexing functionality.
- Manual test: Start server, index a new file, query it immediately to verify "live" updates work without snapshots.
