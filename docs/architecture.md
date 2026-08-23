# Architecture

This document describes the current architecture of `mcp-markdown-ragdocs`.
The canonical design is the application composition around searchkernel's
public record, port, and factory APIs. Older snapshot-based and
`VectorIndex`-based designs are historical and are not implementation
guidance.

## Runtime composition

The application has one search use case and several transport adapters:

```text
                         ┌──────────────────────────┐
                         │ ApplicationSearchUseCase  │
                         │ app/search.py             │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ CanonicalSearchAdapter    │
                         │ search.py                 │
                         └────────────┬─────────────┘
                                      │ public searchkernel APIs
                         ┌────────────▼─────────────┐
                         │ Local record kernel       │
                         │ records + retrieval       │
                         └──────────────────────────┘

  MCP stdio ───────┐
  CLI ─────────────┼── daemon transport ──┐
  HTTP ────────────┘                       │
                                           ▼
                                  ApplicationContext
                                  lifecycle + composition

  Markdown/files ──► parsers/chunker ──► RecordIndexManager
                                           │
                                           ▼
                                  indexing ports and stores

  Durable tasks ───► Huey queue ───► worker process
```

The daemon is the normal runtime authority for CLI and MCP. The HTTP server
can compose an in-process context for deployments that do not use the daemon.
Both paths use the same application search use case rather than maintaining
separate ranking or filtering implementations.

### Google Drive ownership

The current Google Drive owner is `mcp-markdown-ragdocs`. Ragdocs owns Drive
credentials, synchronization, durable recovery state, records, embeddings,
health, and search results. Devkit consumes the authenticated ragdocs search
source for Drive while retaining its non-indexing Google Workspace workflows.
During migration, the old Devkit index is isolated from user-visible results
and retained for the documented rollback window. See the
[Google Drive operator runbook](guides/gdrive-operator-runbook.md) and the
[current migration status](plans/12-google-drive-source-ownership-migration.md)
for operating gates.

## Composition root

`mcp_markdown_ragdocs/app/composition.py`, `context.py`, and
`indexing/record_manager.py` compose the application:

1. Load and validate configuration.
2. Resolve document roots and the active project context.
3. Build the embedding provider and the local record kernel through public
   searchkernel factories.
4. Create `RecordIndexManager` for parsing, chunking, record identity, and
   ingestion.
5. Create `CanonicalSearchAdapter`, which owns the application-facing query
   shape and delegates to `ApplicationSearchUseCase`.
6. Attach watcher, lifecycle, daemon, and optional worker services.

Composition owns policy and lifecycle. Searchkernel owns record storage,
embedding, retrieval, and the lower-level search contracts.

## Shared query flow

Every query follows the same application path:

1. A transport validates its request and maps it to `SearchQuery`.
2. `ApplicationSearchUseCase` converts application policy into searchkernel
   filters and calculates the retrieval limit.
3. The canonical pipeline executes semantic, keyword, and graph retrieval
   through its public ports.
4. The use case applies project, source, score, exclusion, and per-document
   policies.
5. Results are mapped to the application's `ChunkResult` contract, with
   provenance and execution diagnostics preserved.
6. The transport formats the result as MCP content, CLI output, or an HTTP
   response.

The use case is deliberately transport-independent. MCP handlers and daemon
routes must not reimplement ranking, filtering, or result mapping.

### Search boundaries

- `app/search.py`: application query request, policy, and result mapping.
- `search.py`: compatibility adapter for the application-facing query tuple.
- `mcp/tools/document_tools.py`: MCP request and response formatting.
- `daemon/request_router.py`: daemon request routing and admin operations.
- `server.py`: HTTP adapter for the legacy in-process HTTP surface.
- `searchkernel.api`, `searchkernel.domain`, and `searchkernel.ports`:
  public integration boundary for the external search library.

Application modules may import searchkernel only through public modules. The
guard in `scripts/check_public_searchkernel_imports.py` and its corresponding
unit test enforce this boundary.

The application-facing transport boundary is enforced separately by
`import-linter` contracts in `pyproject.toml`. `app/search.py` may not import
CLI, daemon, MCP, HTTP server, or worker packages directly or indirectly.
`app/services.py` has the same direct-import restriction; its existing
`TYPE_CHECKING` annotation for `ApplicationContext` is intentionally not
treated as a transitive runtime dependency because the context composes the
daemon and task layers. These contracts do not restrict current domain,
configuration, git, or adapter dependencies.

### Final application ports

The runtime-facing ports are deliberately small and structural:

| Port | Owner | Responsibility |
| --- | --- | --- |
| `SearchKernelBoundary` | `app/search.py` | Execute canonical record search |
| `IndexingService` | `app/services.py` | Submit or perform indexing work |
| `LifecycleService` | `app/services.py` | Start, stop, and await readiness |
| daemon transport protocols | `daemon/transport.py` | Move request payloads across local IPC |
| router context protocols | `daemon/request_router.py` | Supply query, queue, and admin dependencies |

Lifecycle and task coordination ports are also enforced at their owned
boundary: `ContextIndexingPort`, `LifecycleContextPort`,
`GitIndexingContextPort`, `TaskQueuePort`, `TaskLeasePort`, and
`WorkIntentPort` remain `Protocol` definitions. Task registration modules
consume those ports and must not import or construct the concrete SQLite
`TaskLeaseStore` or `WorkIntentStore`; the AST guard in
`tests/unit/test_architecture_enforcement.py` keeps SQLite composition in the
application roots.

`PipelineSearchBoundary` is the compatibility adapter from the installed
searchkernel pipeline to `SearchKernelBoundary`. `CanonicalSearchAdapter`
retains the historical `query()` tuple for in-repo callers, but delegates to
`ApplicationSearchUseCase`; it is not a second search implementation.
`IndexManagerLike` and the task protocols are similarly structural seams for
worker execution. These seams are intentional compatibility surfaces while
the external searchkernel API remains the storage and retrieval authority.
`IndexingService.index_document()` and `index_record()` expose boolean success
outcomes. `SearchKernelBoundary` accepts a read-only mapping of canonical
filters and returns `RecordSearchOutcome`; the installed pipeline is consumed
through the matching `RecordSearchPipeline` shape.

### Score contract

The `score` returned by MCP, CLI, HTTP, and daemon query responses is the
canonical pipeline's raw weighted Reciprocal Rank Fusion (RRF) score. It is
used to order results and to apply an explicit `min_score` filter. It is not
a calibrated probability, confidence value, or percentage, and application
code must not normalize it a second time. A strong result may therefore have
a score well below `1.0`.

Search provenance retains the contributing strategies and their raw strategy
scores. `CompressionStats` and `SearchStrategyStats` describe the execution
path; they do not change the score contract. The golden parity and canonical
adapter tests enforce descending raw-RRF order and guard against accidental
score calibration at a transport boundary.

An optional `search.abstention_threshold` can supply a raw-score floor, but it
defaults to unset. Calibrate it against deterministic relevant, unrelated, and
low-score queries before activation; do not infer a universal threshold from
the score range of a single corpus.

## Indexing flow

The application owns source-specific ingestion:

1. File and git sources are discovered by the application.
2. `DocumentPlanner` parses Markdown into a `PreparedRecordDocument` with
   stable IDs, metadata, and chunk records.
3. `DocumentWriter` sends planned records through the ingestion port and
   removes stale canonical keys after successful indexing.
4. The record manager maintains source membership, failed-file state, graph
   timing, and state-version updates around those ports.
5. Searchkernel persists records, embeddings, and retrieval structures.

Google Drive replacements use `GDriveReplacementPolicy`, which groups records
by canonical source, preserves journal phases, applies scope membership and
tombstone rules, removes stale canonical keys, and repairs incomplete work
after restart. It depends on the generic record ingestion and storage ports;
the manager still owns graph finalization and state-version timing.

`RecordIndexManager` is the application adapter around this boundary. It
maintains source-to-record bookkeeping, reconciliation, failed-file state, and
application metadata without depending on searchkernel implementation modules.

### Worker separation

When task-backed indexing is enabled, the daemon handles protocol requests
and query serving while the worker process executes Huey indexing, git refresh,
and reindex tasks. Queue leases and task result payloads provide the durable
handoff. The worker does not define a second search contract; it invokes the
same indexing services and publishes state consumed by the daemon.

Single-process indexing remains available for HTTP deployments, tests, and
local debugging. It is an execution-mode choice, not a separate architecture.

### Durable work intent lifecycle

Indexing producers write a `WorkIntent` to SQLite before submitting a Huey
task. The operation plus canonical key is unique, so duplicate file or remove
requests converge on one durable intent. The normal lifecycle is:

```text
submit pending → claim claimed → start running → succeed | fail
                       │                 │
                       └─ release/recover stale claim → pending
```

Claims carry an opaque token and observation time. A producer must claim before
enqueueing and pass that token into the task. The worker starts and terminalizes
only the matching claim, so an old worker cannot complete a re-pended intent.
Failed intents can be reopened by a later submission; stale claims are
returned to `pending` after the configured timeout.

Huey dequeue leases are a separate execution guard. An active lease is
heartbeated during work, expired leases are reclaimed with their serialized
payload, and terminal lease state is owner-token guarded. The dequeue and lease
claim are still separate SQLite boundaries: a crash in that narrow gap remains
the documented at-most-once edge, while successfully claimed work is
at-least-once through payload requeue.

### Producer liveness

Managed producers write `producer.json` with PID, `/proc` start-time ticks,
status, and stop reason. `watcher_active` is true only when both the PID and
start time still match a live process; historical watcher statistics are not
liveness evidence. A managed restart records `stop_reason="restart"` so
diagnostics distinguish it from a normal shutdown.

Task-only producers use `GitWatcher(use_tasks=True,
allow_direct_refresh=False)`. They observe lightweight SQLite git state and
submit durable work instead of indexing directly. This keeps expensive writes
in the worker and prevents a watcher restart from creating an untracked
second indexing path.

## Transport adapters

### MCP

`mcp/server.py` is a daemon-backed stdio client in the normal mode.
`mcp/handlers.py` owns readiness and structured cold-start responses, while
`mcp/tools/` maps tool arguments to the shared query use case.

### CLI

`cli.py` provides thin commands for queries, git history, daemon management,
queue inspection, indexing, and reindex administration. Daemon-backed
commands use the local transport and receive the same payloads as other
administrative clients.

### HTTP

`server.py` exposes the optional FastAPI adapter. It composes an
`ApplicationContext` directly and delegates query behavior to the same
canonical adapter and use case used by daemon-backed paths.

## Lifecycle and control plane

`lifecycle.py` coordinates startup readiness, shutdown, and failure state.
`context.py` owns application resources and index state. The daemon modules
own local IPC, metadata, locking, health, and request routing.

The worker lifecycle in `worker/process.py` is intentionally narrow:

- identify the expected daemon parent;
- start and stop the worker subprocess;
- report PID, readiness, and heartbeat health;
- terminate stale workers for the same runtime;
- remove stale status after shutdown.

Administrative endpoints are routed by `daemon/request_router.py`. They may
inspect queue, rebuild, index, or reindex state, but they do not bypass the
application services for normal indexing or search.

### Request diagnostics

The local transport client generates a `request_id` for each request, and the
daemon preserves and echoes a caller-supplied ID. Error responses echo the ID
when one is available, allowing client logs to correlate validation, routing,
and worker failures without embedding transport details in domain objects.

Query responses include `query_execution_stats` from the shared use case. The
current payload reports whether search degraded and the messages for strategy
failures. Admin responses expose queue, worker, producer, and reindex status
separately, so `watcher_active`, queue activity, and index readiness are not
collapsed into one ambiguous health flag.

## Reindexing and persistence

Durable embedding-model migration is an application orchestration concern.
`indexing/reindex.py` coordinates model namespaces, checkpoints, validation,
activation, contract, and rollback through the public searchkernel reindex
contracts. The manifest and runtime status file expose progress to admin
clients.

This is distinct from the normal local record kernel startup path. A model
migration must not be inferred from an embedding configuration change alone;
the active model metadata and migration state are authoritative.

## Project and source policy

Projects are soft metadata used by query policy and ranking context. Records
remain in the shared canonical index and carry project metadata. Source
filters distinguish document records from git commit records and other
record kinds.

The application resolves project context at the transport boundary, then
passes canonical filters to `ApplicationSearchUseCase`. Storage should not be
duplicated per project merely to implement query scoping.

## Quality gates

Architecture quality is protected by:

- public searchkernel import checks;
- import-linter application transport contracts;
- Ruff C901 complexity checks with a repository ceiling of 30;
- tests for the shared query contract and transport routing;
- real-fixture indexing and reindex tests;
- worker, queue, lifecycle, and MCP seam coverage;
- Ruff and Pyrefly checks in CI;
- a coverage threshold enforced by the test workflow.

When extending the system, add behavior to the appropriate application
service or port adapter first, then cover the public seam. Do not introduce
private searchkernel imports, transport-specific ranking logic, or a second
snapshot/index abstraction.

## Acceptance criteria

An architecture change is considered complete when:

1. Production code crosses searchkernel only through its public API and the
   import-boundary check remains green.
2. Query transports use the shared application use case and preserve raw-RRF
   scores, provenance, and execution diagnostics.
3. Worker shutdown, lifecycle recovery, reindex status transitions, durable
   intent claims, and producer PID/start-time liveness have deterministic tests
   using `tmp_path`, real SQLite, and local fakes.
4. The production package passes Pyrefly and Ruff, and the full non-performance
   suite clears the CI coverage floor with measured margin.
5. Runtime acceptance tests cover daemon startup, readiness, restart
   idempotency, task-driven updates, and request diagnostics without external
   services.

Known deliberate gaps remain: legacy `faiss+sqlite` storage cannot perform a
durable model-scoped migration; the Huey dequeue/lease boundary cannot provide
strict exactly-once execution; and compatibility adapters still expose a few
historical method names. These are explicit contracts, not alternate
architecture directions.
