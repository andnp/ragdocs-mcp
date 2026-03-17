# Feature Spec 23: Concurrency Model & Task Queue (Huey)

**Status:** Partially implemented — active contract for remaining task/admin work
**Related:** [sqlite-refactor.md](../plans/sqlite-refactor.md), [../plans/04-daemon-zmq-control-plane-contract.md](../plans/04-daemon-zmq-control-plane-contract.md)

## 1. Overview
Ragdocs now uses Huey as the durable task layer behind the daemon-backed control plane. The multiprocess queue/snapshot design that originally motivated this spec is no longer the active architecture. What remains relevant is the task contract: the daemon owns Huey supervision, thin clients enqueue work, and task payloads must remain durable and idempotent.

Current implementation status:

- daemon startup initializes `SqliteHuey`
- the daemon supervises a dedicated worker subprocess for the Huey consumer
- file watching enqueues indexing/removal work in task mode
- git refresh work is also routed through task infrastructure
- queue status is exposed from the CLI
- daemon requests reload persisted indices lazily when worker writes advance the store state

Still planned:

- richer task inspection surfaces
- broader task families
- tighter admin payloads around task history and diagnostics

## 2. Goals
1.  **Single daemon ownership**: Only the daemon owns recurring/background work.
2.  **Thin-client safety**: MCP and CLI clients may enqueue work without owning consumers.
3.  **Persistence**: Task queues survive daemon restarts. Pending indexing jobs are not lost.
4.  **Idempotency**: Repeated document delete/index/git-refresh tasks must remain safe.

## 3. Architecture

### 3.1. The Queue Database (`queue.db`)

Ragdocs uses `SqliteHuey` to keep the deployment entirely local-first with no Redis/RabbitMQ dependency.

- **Location**: the daemon runtime path (`RuntimePaths.queue_db_path`)
- **Purpose**: stores pending tasks, scheduled tasks, failures, and task metadata needed for queue inspection

### 3.2. Process Roles

#### A. Thin Clients (MCP + CLI)

- **Responsibility**: attach to the daemon, forward user requests, and request daemon startup when required
- **Operation**:
    - forward MCP tool discovery/tool calls to the daemon
    - forward CLI query/admin requests to the daemon
    - never own the Huey consumer or background watcher lifecycle

#### B. Global Daemon

- **Responsibility**: own lifecycle, task supervision, search-serving, and management payloads
- **Operation**:
    - starts and stops the Huey worker subprocess
    - runs file and git watchers
    - enqueues document/git maintenance work
    - writes index/task/runtime state to the shared local store

## 4. Implementation Plan

## 4. Remaining work

### Phase 1 — richer task inspection

Planned additions:

- per-task inspection by id
- better recent failure payloads
- deeper admin summaries beyond `queue status`

### Phase 2 — broaden task families

Planned additions:

- reconciliation/integrity tasks where not already covered
- explicit rebuild/maintenance tasks
- clearer retry/failure semantics surfaced to operators

### Phase 3 — converge docs/contracts

Planned additions:

- align this spec with the admin/control-plane contract
- keep path names and payload expectations consistent with `docs/plans/04-daemon-zmq-control-plane-contract.md`

## 5. Security & Safety
- **Database Locks**: SQLite handles its own internal locking. The daemon remains the authority for writes.
- **Task Idempotency**: Indexing, delete, and git-refresh tasks must behave safely when repeated.