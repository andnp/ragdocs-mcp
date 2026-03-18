# Plan: Global Daemon, Huey Tasks, Admin Surface & Soft Projects

**Status:** Partially complete — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/specs/25-ragdocs-product-principles.md`, `docs/specs/23-concurrency-huey.md`, `docs/plans/04-daemon-zmq-control-plane-contract.md`, `docs/guides/multi-project-setup.md`

## Executive Summary

This plan replaces ragdocs' current coordination model with a global daemon and thin clients, uses Huey as the durable background task substrate, introduces a minimal operator/admin surface, and converts project semantics from hard partitioning to bounded ranking uplift. The design borrows the daemon and metadata principles from `mcp-memory` while intentionally keeping Huey as the task engine because ragdocs already has that direction documented and the user explicitly wants it.

## Verified Implementation Status (2026-03-17)

- **Phase 1 — daemon scaffolding:** substantially complete. `src/daemon/metadata.py`, `src/daemon/paths.py`, `src/daemon/lock.py`, `src/daemon/management.py`, and daemon CLI commands are present, and the current transport now uses ZMQ `ROUTER`/`DEALER` IPC via `src/daemon/transport.py`.
- **Thin-client behavior:** substantially complete. `src/mcp/server.py` and CLI query/admin commands forward work to the daemon over local IPC, the in-process fallback path is gone, and follow-up hardening landed explicit timeout errors, request IDs, attach/readiness fixes, and correct boot-lock release after startup.
- **Huey ownership:** substantially complete. Production daemon startup initializes the queue, registers indexing tasks, supervises a dedicated worker subprocess for Huey task execution, and now places file/git watcher execution behind that worker boundary. Task-persistence hardening, worker restart supervision, and daemon-side freshness reloads are implemented; richer task families and deeper admin visibility remain incomplete.
- **Admin surface:** partially complete. `daemon start|stop|status|restart`, `queue status`, `index stats`, `/api/admin/overview`, and `/internal/shutdown` now exist, and the admin/index/queue payloads now expose queue-pressure metadata, watcher debounce/backpressure counters, and current index state. Richer task inspection and longer-horizon throughput history remain pending.
- **Soft projects:** substantially complete. Indexed content and query results now carry `project_id` metadata, search applies the bounded `1.2x` active-project uplift, authoritative index storage is global, the default global-corpus discovery path spans all configured project roots unless the caller explicitly narrows scope, and manager/reconciler/watcher identity handling now resolves doc IDs against the real configured roots rather than relying solely on the common-root compatibility path. Follow-up work is mostly operational polish and broader docs cleanup.
- **Indexing controls:** in progress. Worker-side indexing now has bounded task backpressure, per-file last-request-wins debounce in the watcher, and capped torch thread configuration to reduce CPU saturation during embedding-heavy workloads.

## Goals

1. run one global daemon per user environment
2. make MCP sessions and CLI commands thin clients
3. route indexing and maintenance through Huey-backed durable tasks
4. expose daemon/task/index health via CLI and simple admin endpoints
5. treat projects as ranking metadata, not execution/storage boundaries

## Locked Decisions

1. daemon/store scope is global per user environment
2. admin v1 is CLI-only
3. active-project uplift starts at **1.2x**

## Non-Goals

1. preserve the old snapshot-based multiprocess architecture
2. depend on Redis, RabbitMQ, or remote services
3. build a large web application before core control-plane migration is stable

## Control Plane Design

### Runtime Roles

#### Thin Client

- MCP stdio server instance
- CLI command invocation
- optional HTTP request entry point

Thin clients do three things only:

1. resolve active user/project context
2. ensure the daemon exists or connect to it
3. forward requests to the daemon

#### Global Daemon

- owns watchers, embedding models, index manager, git refresh, and Huey consumer lifecycle
- owns durable task execution and scheduling
- owns local runtime metadata and health reporting

### ZMQ Contract

Adopt a local Unix-socket ZMQ transport.

- daemon side: `ROUTER`
- client side: `DEALER`
- endpoint: local socket path under the user cache/state directory
- permissions: owner-only socket file
- handshake includes daemon PID, version, socket path, uptime, and health snapshot

### Daemon Metadata

Persist a small metadata record describing the active daemon:

- PID
- started_at
- binary path
- version
- socket path
- runtime status

This metadata exists to support:

- client attach logic
- stale-daemon cleanup
- operator status commands

## Task Architecture

### Why Huey

Ragdocs already has a draft direction toward `SqliteHuey`. That should become the one durable task layer instead of a partial sidecar.

### Required Task Families

1. index document
2. delete document
3. reconcile filesystem state
4. refresh git history index
5. rebuild search artifacts
6. scheduled maintenance and integrity checks

### Queue Ownership Rule

- only the daemon runs the Huey consumer
- clients enqueue work only
- direct background indexing from client processes is forbidden once the daemon path is active

### Retry and Failure Rules

- every durable task must be idempotent
- failures must remain inspectable
- repeated tasks for the same document/path should be coalesced or safely overwrite prior work

## Admin Surface

### CLI Commands

Add a minimal operator command set:

- `ragdocs daemon start`
- `ragdocs daemon stop`
- `ragdocs daemon restart`
- `ragdocs daemon status`
- `ragdocs queue status`
- `ragdocs index stats`
- `ragdocs admin serve` or equivalent dashboard launcher

### Dashboard Scope

The dashboard is deferred. The first operator surface is CLI-only.

When a dashboard eventually exists, it should stay narrow:

- daemon health
- queue depth and recent failures
- document and chunk counts
- git index counts
- recent task runs
- search/index health indicators

## Soft Project Semantics

### Current Problem

Current multi-project behavior pushes the system toward project activation and separate index locations. That makes storage, startup, and control-plane logic heavier than necessary.

### Target Model

Projects become metadata stamped onto indexed content.

- every document/chunk may carry `project_id`
- active project context may apply a bounded score multiplier
- the corpus remains globally searchable by default
- explicit project filters remain available when requested

### Ranking Contract

Use bounded uplift rather than hard partitioning.

Example:

$$
score_{final} = score_{base} \times 1.2
$$

Where non-matching content keeps multiplier $1.0$.

For the first implementation pass, the uplift is intentionally conservative and fixed at $1.2$.

## Migration Plan

### Phase 1 — Daemon Scaffolding

1. define daemon metadata model and filesystem layout
2. add start/attach/status logic
3. add ZMQ client/server transport skeleton

### Phase 2 — Huey Ownership

1. move watcher-triggered work into task enqueue
2. make daemon own Huey consumer lifecycle
3. add task result and failure inspection surface

### Phase 3 — Thin MCP Client

1. convert MCP server path into request-forwarding thin client
2. keep clients strict and daemon-backed rather than preserving a local execution fallback
3. prove multiple clients can safely share one daemon

### Phase 4 — Admin Surface

1. add daemon CLI commands
2. add queue/index status commands
3. defer dashboard work until the CLI surface proves sufficient

### Phase 5 — Soft Projects

1. unify storage assumptions
2. stamp project metadata at ingest time ✅
3. add project uplift in ranking
4. rewrite multi-project docs around soft semantics

## Decision Matrix

| Decision Area | Option | Pros | Cons | Decision |
|---|---|---|---|---|
| Client/daemon transport | ZMQ over Unix socket | local, explicit, concurrency-friendly, proven in sibling repo | new plumbing to build | Adopt |
| Task engine | Huey + SQLite | durable, already aligned with existing ragdocs draft | daemon must own consumer carefully | Adopt |
| Project model | Hard segmentation | stronger locality | storage and runtime complexity | Reject |
| Project model | Soft uplift at 1.2x | simpler control plane, global-first search | requires ranking tuning | Adopt |
| Admin surface | CLI only | smallest scope | limited visibility | Adopt for v1 |

## Key Risks

| Risk | Severity | Mitigation |
|---|---|---|
| mixed old/new startup paths create duplicate work | High | define one authoritative daemon ownership rule and add migration guardrails |
| daemon failure leaves clients with unclear state | High | metadata + health probe + explicit status commands |
| Huey tasks duplicate or race on the same file | High | idempotent handlers and coalescing rules |
| project uplift harms ranking quality | Medium | keep uplift bounded and evaluate before increasing weight |
| admin surface balloons in scope | Medium | constrain v1 to status, queue, index health |

## Acceptance Criteria

1. one daemon can serve multiple clients safely
2. only the daemon runs background consumers/watchers
3. indexing and git refresh survive client restarts
4. operators can inspect daemon, queue, and index state
5. project context boosts ranking without default hard partitioning
