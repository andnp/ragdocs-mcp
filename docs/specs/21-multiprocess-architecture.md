# 21. Multiprocess Architecture

**Version:** 1.0.0
**Date:** 2026-01-23
**Status:** Implemented

---

## Executive Summary

**Purpose:** Separate indexing operations from query processing using a two-process architecture to eliminate GIL contention, improve startup time, and enable non-blocking searches.

**Scope:** Main process handles MCP protocol and read-only searches; worker process handles file watching, indexing, and index updates. Processes communicate via multiprocessing queues and file-based snapshots.

**Decision:** Implement process separation with snapshot-based index synchronization rather than shared memory or RPC-based approaches.

---

## 1. Goals & Non-Goals

### Goals

1. **Non-blocking MCP:** Query processing never blocked by indexing operations
2. **Fast Perceived Startup:** MCP server responds to protocol immediately, indices load in background
3. **Isolated Failures:** Worker crash does not terminate MCP server; automatic restart with backoff
4. **Zero-Copy Index Sync:** Indices transferred via file snapshots, not serialized over IPC
5. **Backward Compatible:** Single-process mode remains available via configuration

### Non-Goals

1. **Distributed Processing:** No multi-machine deployment support
2. **Shared Memory Indices:** Complexity outweighs benefits for local-first use case
3. **Real-time Index Updates:** Sub-second index propagation not required (~100ms polling acceptable)
4. **Worker Pool:** Single worker process sufficient for typical documentation corpus size

---

## 2. Current State Analysis (Before)

### 2.1. Single-Process Architecture

```
┌─────────────────────────────────────────┐
│           ApplicationContext            │
│  ┌─────────────────────────────────┐   │
│  │ Embedding Model (loads on start)│   │ ◀── Blocks MCP for 5-30s
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ VectorIndex, KeywordIndex, Graph│   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ FileWatcher (background thread) │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ IndexManager (blocking updates) │   │ ◀── Blocks queries during reindex
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2.2. Problems with Single-Process

| Problem | Impact | Frequency |
|---------|--------|-----------|
| **Slow startup** | MCP client times out waiting for server | Every startup |
| **GIL contention** | Indexing blocks query threads | During file changes |
| **Memory pressure** | Single process holds all state | Continuous |
| **No fault isolation** | Indexing error can crash server | Rare but catastrophic |

---

## 3. Proposed Solution

### 3.1. Two-Process Architecture

```
MAIN PROCESS (MCP Server)              WORKER PROCESS (Indexer)
┌────────────────────────┐             ┌────────────────────────┐
│ LifecycleCoordinator   │             │ worker_main()          │
│   └─ start_with_worker │── spawn ──▶│   └─ WorkerState       │
├────────────────────────┤             ├────────────────────────┤
│ ReadOnlyContext        │             │ IndexManager           │
│   ├─ VectorIndex       │             │   ├─ VectorIndex       │
│   ├─ KeywordIndex      │             │   ├─ KeywordIndex      │
│   ├─ GraphStore        │             │   └─ GraphStore        │
│   └─ SearchOrchestrator│             │ FileWatcher            │
├────────────────────────┤             │ GitWatcher             │
│ IndexSyncReceiver      │◀── poll ───│ IndexSyncPublisher     │
│   └─ reload_callback() │             │   └─ publish()         │
├────────────────────────┤             ├────────────────────────┤
│ command_queue ─────────┼──────────▶│ command_queue          │
│ response_queue ◀───────┼────────────│ response_queue         │
└────────────────────────┘             └────────────────────────┘
         │                                       │
         ▼                                       ▼
    MCP Protocol                         File System Watchers
    (query_documents)                    (index updates)
```

### 3.2. Component Responsibilities

| Component | Process | Responsibility |
|-----------|---------|---------------|
| `LifecycleCoordinator` | Main | Spawn worker, health checks, restart on failure |
| `ReadOnlyContext` | Main | Hold read-only indices, execute searches |
| `IndexSyncReceiver` | Main | Poll for snapshots, trigger hot-reload |
| `QueueManager` | Both | Async-friendly multiprocessing.Queue wrapper |
| `WorkerState` | Worker | Hold mutable indices, watchers, config |
| `IndexSyncPublisher` | Worker | Persist indices to versioned snapshots |
| `worker_main()` | Worker | Entry point, command loop, shutdown |

### 3.3. Communication Channels

```
┌──────────────────┐         ┌──────────────────┐
│   MAIN PROCESS   │         │  WORKER PROCESS  │
├──────────────────┤         ├──────────────────┤
│                  │         │                  │
│  command_queue ──┼────────▶│── command_queue  │
│                  │  Queue  │                  │
│  response_queue◀─┼─────────┼── response_queue │
│                  │         │                  │
│  snapshots/ ◀────┼─ File ──┼── snapshots/     │
│  version.bin     │  I/O    │   version.bin    │
│                  │         │                  │
└──────────────────┘         └──────────────────┘
```

**Command Queue (main → worker):**
- `ShutdownCommand`: Request graceful/forced shutdown
- `HealthCheckCommand`: Request health status
- `ReindexDocumentCommand`: Request specific document reindex

**Response Queue (worker → main):**
- `InitCompleteNotification`: Worker initialization finished
- `IndexUpdatedNotification`: New snapshot published
- `HealthStatusResponse`: Health metrics

**File-Based Sync (worker → main):**
- `snapshots/version.bin`: Binary uint32 version number
- `snapshots/v{N}/`: Versioned index directories

### 3.4. Index Synchronization Flow

```
Worker Process                           Main Process
      │                                       │
      │  1. FileWatcher detects change        │
      ▼                                       │
┌─────────────┐                              │
│ index_doc() │                              │
└──────┬──────┘                              │
       │                                      │
       │  2. Persist to snapshot             │
       ▼                                      │
┌─────────────────┐                          │
│ publish(v{N+1}) │                          │
│  ├─ vector/     │                          │
│  ├─ keyword/    │                          │
│  └─ graph/      │                          │
└────────┬────────┘                          │
         │                                    │
         │  3. Write version.bin atomically   │
         ▼                                    │
┌─────────────────┐                          │
│ version.bin = N+1                          │
└─────────────────┘                          │
                                              │
         │  4. Send notification              │
         ▼                                    │
┌─────────────────┐                          │
│ IndexUpdated    │                          │
│ Notification    ├──────────────────────────▶│
└─────────────────┘                          │
                                              │  5. Poll detects version change
                                              ▼
                                    ┌─────────────────┐
                                    │ check_for_update│
                                    │ version > curr  │
                                    └────────┬────────┘
                                              │
                                              │  6. Hot-reload indices
                                              ▼
                                    ┌─────────────────┐
                                    │ reload_callback │
                                    │  load_from(v{N})│
                                    └─────────────────┘
```

**Atomicity Guarantee:**
- `version.bin` written via temp file + rename (atomic on POSIX)
- Main process only reads complete snapshots
- Old snapshots cleaned up after new version confirmed

---

## 4. Implementation Details

### 4.1. IPC Commands (src/ipc/commands.py)

```python
@dataclass(frozen=True)
class ShutdownCommand:
    graceful: bool = True
    timeout: float = 5.0

@dataclass(frozen=True)
class HealthCheckCommand:
    pass

@dataclass(frozen=True)
class IndexUpdatedNotification:
    version: int
    doc_count: int
    timestamp: float = field(default_factory=time.time)

@dataclass(frozen=True)
class InitCompleteNotification:
    version: int
    doc_count: int
    timestamp: float = field(default_factory=time.time)

@dataclass(frozen=True)
class HealthStatusResponse:
    healthy: bool
    queue_depth: int
    last_index_time: float | None
    doc_count: int
```

### 4.2. QueueManager (src/ipc/queue_manager.py)

```python
class QueueManager:
    """Async-friendly wrapper for multiprocessing.Queue."""

    async def get(self, timeout: float = 1.0) -> IPCMessage | None:
        """Poll-based async get with timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        poll_interval = 0.01

        while True:
            message = self.get_nowait()
            if message is not None:
                return message

            if asyncio.get_event_loop().time() >= deadline:
                return None

            await asyncio.sleep(poll_interval)
```

### 4.3. IndexSyncPublisher (src/ipc/index_sync.py)

```python
class IndexSyncPublisher:
    def publish(self, persist_callback: Callable[[Path], None]) -> int:
        """Publish new snapshot version."""
        new_version = self._version + 1
        snapshot_dir = self._snapshot_base / f"v{new_version}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        try:
            persist_callback(snapshot_dir)

            # Atomic version file update
            temp_version_file = self._version_file.with_suffix(".tmp")
            temp_version_file.write_bytes(struct.pack("<I", new_version))
            temp_version_file.replace(self._version_file)

            self._version = new_version
            self._cleanup_old_snapshots(keep=2)
            return new_version
        except Exception:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise
```

### 4.4. Worker Process Entry Point (src/worker/process.py)

```python
def worker_main(
    config_dict: dict,
    command_queue: Queue,
    response_queue: Queue,
    shutdown_event: EventType,
    snapshot_base: Path,
) -> None:
    """Worker process entry point."""
    asyncio.run(_worker_async_main(
        config_dict, command_queue, response_queue,
        shutdown_event, snapshot_base,
    ))

async def _worker_async_main(...) -> None:
    # 1. Install signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    # 2. Initialize worker state
    state = await _initialize_worker(...)

    # 3. Run command loop until shutdown
    await _run_command_loop(state)

    # 4. Clean shutdown
    await _shutdown_worker(state)
```

### 4.5. ReadOnlyContext (src/reader/context.py)

```python
@dataclass
class ReadOnlyContext:
    config: Config
    vector: VectorIndex
    keyword: KeywordIndex
    graph: GraphStore
    orchestrator: SearchOrchestrator
    sync_receiver: IndexSyncReceiver

    @classmethod
    async def create(cls, config: Config, snapshot_base: Path) -> ReadOnlyContext:
        """Create context with indices loaded from latest snapshot."""
        # Create empty indices
        vector = VectorIndex(...)
        keyword = KeywordIndex()
        graph = GraphStore()

        # Load from snapshot if available
        latest = _find_latest_snapshot(snapshot_base)
        if latest:
            _load_indices_from_snapshot(vector, keyword, graph, latest)

        # Create orchestrator for searches
        orchestrator = SearchOrchestrator(vector, keyword, graph, config, ...)

        # Create receiver with hot-reload callback
        def reload_callback(snapshot_dir: Path, version: int) -> None:
            _load_indices_from_snapshot(vector, keyword, graph, snapshot_dir)

        sync_receiver = IndexSyncReceiver(snapshot_base, reload_callback)

        return cls(...)
```

### 4.6. Lifecycle Coordinator Extensions (src/lifecycle.py)

```python
async def start_with_worker(self, readonly_ctx: ReadOnlyContext) -> None:
    """Start in multiprocess mode with worker process."""
    self._state = LifecycleState.STARTING
    self._readonly_ctx = readonly_ctx

    # Spawn worker process
    await self._start_worker()

    # Wait for initialization
    init_received = await self._wait_for_init(worker_config.startup_timeout)

    # Start sync watcher
    await readonly_ctx.start_sync_watcher()

    # Start health check loop
    self._health_check_task = asyncio.create_task(self._health_check_loop())

async def _health_check_loop(self) -> None:
    """Periodic health checks with automatic restart."""
    while True:
        await asyncio.sleep(interval)

        if not self._is_worker_alive():
            self._state = LifecycleState.DEGRADED
            await self._restart_worker()
            continue

        # Request health status
        self._command_queue.put_nowait(HealthCheckCommand())
        response = await asyncio.wait_for(...)

        if not response.healthy:
            self._state = LifecycleState.DEGRADED
```

---

## 5. Configuration

```toml
[tool.ragdocs.worker]
enabled = true                    # Enable multiprocessing (default: true)
startup_timeout = 30.0            # Max seconds to wait for worker init
shutdown_timeout = 5.0            # Max seconds for graceful shutdown
health_check_interval = 10.0      # Seconds between health checks
max_restart_attempts = 3          # Auto-restart limit
restart_backoff_base = 1.0        # Exponential backoff base (1s, 2s, 4s)
snapshot_keep_count = 2           # Old snapshots to retain
index_poll_interval = 0.1         # How often main checks for updates (seconds)
```

**Disabling Multiprocess Mode:**
```toml
[tool.ragdocs.worker]
enabled = false
```

When disabled, `ApplicationContext` is used directly (single-process mode) with blocking initialization.

---

## 6. Decision Matrix

| Option | Complexity | Memory | Latency | Fault Isolation | Chosen |
|--------|:----------:|:------:|:-------:|:---------------:|:------:|
| **Snapshot-based sync** | Medium | 2x indices | ~100ms | Full | ✓ |
| Shared memory (mmap) | High | 1x indices | <1ms | Partial | |
| RPC (gRPC/ZeroMQ) | High | 1x indices | <10ms | Full | |
| Single process | Low | 1x indices | 0ms | None | |

**Decision Rationale:**
- Snapshot-based sync chosen for simplicity and full fault isolation
- Memory overhead acceptable for local-first use case (typically <100MB total)
- 100ms latency acceptable for file-watching-triggered updates
- Shared memory rejected due to complexity of cross-process index structures
- RPC rejected due to serialization overhead for large index payloads

---

## 7. Testing Strategy

### Unit Tests

| Test File | Test Cases |
|-----------|------------|
| `tests/unit/test_ipc_commands.py` | Command serialization, frozen dataclass behavior |
| `tests/unit/test_queue_manager.py` | Async get/put, timeout handling, drain |
| `tests/unit/test_index_sync.py` | Version file atomicity, snapshot cleanup, reload callback |

### Integration Tests

| Test File | Test Cases |
|-----------|------------|
| `tests/integration/test_worker_process.py` | Worker startup, command processing, shutdown |
| `tests/integration/test_index_sync_e2e.py` | Publisher→Receiver flow, hot-reload verification |
| `tests/integration/test_lifecycle_worker.py` | Health checks, restart on failure, graceful shutdown |

### Test Patterns

**Worker Process Testing:**
```python
def test_worker_initialization(tmp_path):
    """Worker should publish initial snapshot after indexing."""
    config = create_test_config(tmp_path)
    cmd_queue, resp_queue = Queue(), Queue()
    shutdown_event = Event()

    # Start worker in subprocess
    process = Process(target=worker_main, args=(...))
    process.start()

    # Wait for init notification
    notification = resp_queue.get(timeout=30)
    assert isinstance(notification, InitCompleteNotification)
    assert notification.version == 1

    # Verify snapshot exists
    assert (tmp_path / "snapshots" / "v1" / "vector").exists()

    # Shutdown
    shutdown_event.set()
    process.join(timeout=5)
```

---

## 8. Observability

### Logging

Worker process logs prefixed with `[WORKER]`:
```
[WORKER INFO] src.worker.process: Initializing worker process
[WORKER INFO] src.worker.process: Loading embedding model: BAAI/bge-small-en-v1.5
[WORKER INFO] src.worker.process: Initial indexing complete: 42 documents
[WORKER INFO] src.ipc.index_sync: Published index snapshot v1 to .../snapshots/v1
[WORKER INFO] src.worker.process: Worker initialized: 42 documents, snapshot v1
```

Main process logs:
```
INFO src.lifecycle: Worker process started (pid=12345)
INFO src.lifecycle: Worker init complete: v1, 42 docs
INFO src.lifecycle: Lifecycle: READY (worker initialized)
INFO src.reader.context: Reloaded indices from snapshot v2
```

### Health Metrics

`HealthStatusResponse` includes:
- `healthy`: Overall health status
- `queue_depth`: Pending file watcher events
- `last_index_time`: Timestamp of last index operation
- `doc_count`: Current document count

---

## 9. Architecture Decision Records

### ADR-1: Snapshot vs. Shared Memory

**Status:** Accepted

**Context:** Indices must be accessible to main process for queries while worker process updates them.

**Decision:** Use file-based snapshots with version tracking.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Snapshots (selected)** | Simple, fault-isolated, portable | Memory overhead, I/O latency |
| Shared memory (mmap) | Zero-copy, low latency | Complex synchronization, crash risks |
| Message passing | Clean abstraction | Serialization overhead for large indices |

**Rationale:** Snapshots provide full fault isolation—worker crash leaves main process with last-known-good indices. The ~100ms polling latency is acceptable for file-change-triggered updates. Memory overhead (2x indices) is acceptable for typical documentation sizes (<100MB).

### ADR-2: Polling vs. File Notification

**Status:** Accepted

**Context:** Main process needs to detect when worker publishes new snapshot.

**Decision:** Poll `version.bin` at configurable interval (default 100ms).

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Polling (selected)** | Simple, cross-platform | Slight latency, CPU overhead |
| inotify/FSEvents | Instant notification | Platform-specific, complexity |
| Queue notification | Direct communication | Already have queue, adds message type |

**Rationale:** Polling is simple and reliable. The `IndexUpdatedNotification` via queue provides immediate notification; polling is backup for queue drops. 100ms poll interval adds negligible CPU overhead.

### ADR-3: Single Worker vs. Pool

**Status:** Accepted

**Context:** Should there be multiple worker processes for parallel indexing?

**Decision:** Single worker process.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Single worker (selected)** | Simple, predictable resource usage | No parallelism |
| Worker pool | Parallel indexing | Complex coordination, memory overhead |
| Thread pool in worker | Parallel within process | GIL limits effectiveness |

**Rationale:** Single worker is sufficient for typical documentation corpus sizes (hundreds to thousands of files). Embedding model is the bottleneck, and it already batches internally. Worker pool adds complexity without proportional benefit.

---

## 10. Implementation Files

| File | Role |
|------|------|
| [src/ipc/commands.py](../../src/ipc/commands.py) | IPC message dataclasses |
| [src/ipc/queue_manager.py](../../src/ipc/queue_manager.py) | Async-friendly queue wrapper |
| [src/ipc/index_sync.py](../../src/ipc/index_sync.py) | Snapshot publisher and receiver |
| [src/worker/process.py](../../src/worker/process.py) | Worker process entry point and command loop |
| [src/reader/context.py](../../src/reader/context.py) | Read-only context for main process |
| [src/lifecycle.py](../../src/lifecycle.py) | Lifecycle coordinator with worker management |
| [src/config.py](../../src/config.py) | WorkerConfig dataclass |

---

## 11. References

1. **Related Specs:**
   - [Spec 19: Self-Healing Index Infrastructure](19-self-healing-indices.md) - Index corruption recovery
   - [docs/architecture.md](../architecture.md) - Overall system architecture

2. **External References:**
   - Python multiprocessing documentation
   - POSIX atomic file operations (rename)
