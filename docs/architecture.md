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

## Indexing flow

The application owns source-specific ingestion:

1. File and git sources are discovered by the application.
2. Parsers produce application `Document` or source records.
3. The record manager chunks documents and assigns stable source metadata.
4. Records cross the searchkernel ingestion port.
5. Searchkernel persists records, embeddings, and retrieval structures.

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
- tests for the shared query contract and transport routing;
- real-fixture indexing and reindex tests;
- worker, queue, lifecycle, and MCP seam coverage;
- Ruff and Pyright checks in CI;
- a coverage threshold enforced by the test workflow.

When extending the system, add behavior to the appropriate application
service or port adapter first, then cover the public seam. Do not introduce
private searchkernel imports, transport-specific ranking logic, or a second
snapshot/index abstraction.
