# Master Implementation Plan: Next-Gen Architecture (REVISED)

**Context:** This document synthesized the transition from a file-based, GIL-bound architecture to a concurrent, SQLite-backed system. Several components (FTS5 KeywordIndex, basic DatabaseManager, Query Classifier) are already implemented. This plan focuses on completing the migration.

---

## Epic 1: SQLite-backed Core Storage & Simplification
*Goal: Consolidate all content and indices into `ragdocs.db`. Eliminate Whoosh and NetworkX JSON dependencies.*

**Commit 1.1: Extend SQLite Schema**
- **Action:** Update `src/storage/db.py` to add missing tables:
    - `graph_edges` (source, target, type, context) to replace NetworkX JSON.
    - `tasks` (id, task_name, data, status) for Huey.
    - `system1_journal` (id, content, timestamp, status) for memory scratchpad.
    - `system_state` (key, value) for leader election and maintenance metadata.
- **Verification:**
    - `test_schema_initialization`: Verify all tables exist and have correct columns after `db.initialize_schema()`.
    - `test_concurrent_wal_access`: Spawn 5 threads writing to `kv_store` while a main thread performs 1000 reads; verify zero `database is locked` errors.

**Commit 1.2: Refactor GraphStore to SQLite**
- **Action:** Rewrite `src/indices/graph.py` to use `ragdocs.db` for all edge/node operations.
- **Details:** Remove `networkx` dependency. Graph traversals should become simple SQL joins.
- **Verification:**
    - `test_graph_persistence`: Add nodes and edges, close DB, reopen, and verify connectivity matches.
    - `test_neighbor_lookup_performance`: Verify 2-hop neighbor lookup on a 10,000-node graph completes in < 10ms.

**Commit 1.3: Refactor VectorIndex to SQLite Persistence**
- **Action:** Update `src/indices/vector.py` to store chunk vectors and metadata as BLOBs/JSON in the `chunks` table.
- **Verification:**
    - `test_vector_load_on_startup`: Verify that after persisting 100 chunks to SQLite, the `VectorIndex` successfully populates its in-memory FAISS index with matching IDs and dimensions on reload.

**Commit 1.4: Eliminate the Snapshot System**
- **Action:** Delete `src/ipc/` directory (if it still contains snapshot logic).
- **Verification:**
    - `test_cross_process_consistency`: Start a separate process that writes a document to `ragdocs.db`; verify the Main process sees it immediately without any "sync" calls.

---

## Epic 2: Lifecycle Coordinator & Leader Election
*Goal: Centralize process state and ensure only one worker runs per project.*

**Commit 2.1: Refactor LifecycleCoordinator**
- **Action:** Update `src/lifecycle.py` to become the absolute source of truth for the process state.
- **Verification:**
    - `test_lifecycle_state_machine`: Verify transitions: `Uninitialized -> Starting -> Ready -> ShuttingDown -> Terminated`.
    - `test_shutdown_timeouts`: Mock a hanging cleanup task and verify the `emergency_timer` triggers `os._exit` within 4 seconds.

**Commit 2.2: Add SQLite Leader Election**
- **Action:** Implement a locking mechanism in `LifecycleCoordinator.start()`.
- **Details:** Use the `system_state` table or a file-based lock (`worker.lock`) to determine if the current instance is the **Primary** (spawns Huey worker) or a **Replica** (read-only).
- **Verification:**
    - `test_exclusive_leader_election`: Start two `LifecycleCoordinator` instances simultaneously; verify one becomes `READY_PRIMARY` and the other `READY_REPLICA`.
    - `test_leader_handoff`: Shut down the Primary; verify the Replica can successfully acquire the lock and promote itself to Primary on its next health check.

---

## Epic 3: Huey Task Queue Integration
*Goal: Move heavy lifting to a persistent, database-backed queue.*

**Commit 3.1: Setup SqliteHuey**
- **Action:** Add `huey` dependency. Create `src/coordination/queue.py` defining the `SqliteHuey` instance.
- **Verification:**
    - `test_task_persistence`: Enqueue 5 tasks, kill the process, restart, and verify the tasks are still in `queue.db`.

**Commit 3.2: Worker Process Refactor**
- **Action:** Update `src/worker/process.py` to launch the `huey_consumer`.
- **Verification:**
    - `test_worker_spawns_on_leader`: Verify the Huey consumer process only exists when the Lifecycle state is `READY_PRIMARY`.

**Commit 3.3: Convert Indexing to Huey Tasks**
- **Action:** Refactor `watcher.py` and `manager.py` to enqueue Huey tasks.
- **Verification:**
    - `test_end_to_end_indexing_flow`: Touch a file -> Verify `index_document` task is enqueued -> Verify document appears in `ragdocs.db` after worker processes it.

---

## Epic 4: Autonomous Agentic Memory (System 1 & System 2)
*Goal: Flush raw thoughts into a refined memory graph autonomously.*

**Commit 4.1: record_thought MCP Tool**
- **Action:** Implement `record_thought(text: str)` in `src/mcp/tools/memory_tools.py`.
- **Verification:**
    - `test_record_thought_persistence`: Call tool via MCP; verify entry exists in `system1_journal` with correct timestamp.

**Commit 4.2: AI CLI Provider Layer**
- **Action:** Create `src/memory/providers.py` with `GeminiCLIProvider` (using `gemini ask --json`).
- **Verification:**
    - `test_provider_json_parsing`: Mock the `gemini` subprocess output with valid/invalid JSON; verify the provider handles errors and returns typed dicts.

**Commit 4.3: Consolidation Tasks**
- **Action:** Implement the "Fast" (flush cache) and "Slow" (Strategy Roulette) Huey tasks.
- **Verification:**
    - `test_fast_consolidation_logic`: Provide 3 related scratchpad entries; verify the agent correctly proposes a `merge` or `create` operation and marks entries as `merged` in System 1.
    - `test_soft_delete_safety`: Trigger a consolidation that deletes a memory; verify the file moved to `.memories/.trash/` and is recoverable.

**Commit 4.4: System Status Tool**
- **Action:** Add `get_system_status()` MCP tool.
- **Verification:**
    - `test_status_report`: Verify tool returns the correct "Last Consolidated" timestamp and current Huey queue depth.
