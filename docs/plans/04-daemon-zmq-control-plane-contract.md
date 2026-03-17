# Contract: Global Daemon & ZMQ Control Plane

**Status:** Draft
**Date:** 2026-03-16
**Related:** `docs/specs/25-ragdocs-product-principles.md`, `docs/plans/02-global-daemon-huey-and-soft-projects.md`, `docs/plans/03-tranche-implementation-roadmap.md`

## Executive Summary

This document defines the control-plane contract for Ragdocs V2: one global daemon, thin clients, boot locking, daemon metadata, and local ZMQ transport. The contract is intentionally modeled after the working `mcp-memory` patterns, but adapted to Ragdocs' reduced scope and Huey-based task ownership.

## Mandatory Reference Implementation

Future agents should study `/home/andy/Projects/personal/mcp-memory` before changing this contract.

Minimum references:

- `docs/specs/00-product-principles.md`
- `docs/plans/03-daemon-infrastructure-zmq-and-security.md`
- `src/mcp_memory/daemon.py`
- `src/mcp_memory/daemon_lifecycle.py`
- `src/mcp_memory/daemon_transport.py`
- `src/mcp_memory/daemon_models.py`

## Goals

1. ensure exactly one daemon owns the runtime authority
2. allow multiple MCP/CLI clients to attach safely
3. expose a stable local request/response contract for admin and search requests
4. make stale-daemon cleanup explicit and inspectable

## Non-Goals

1. expose ZMQ directly to end users as a public protocol
2. support remote or networked daemon access in v1
3. keep the snapshot-based multiprocess system alive in parallel

## Runtime Roles

### Thin Client

Thin clients include:

- MCP stdio sessions
- CLI commands
- optional HTTP façade if one survives later

Thin clients may:

- resolve active project context
- ensure the daemon is running
- forward requests to the daemon

Thin clients may **not**:

- own watchers
- own Huey consumers
- own embedding-model lifetime
- perform authoritative background indexing directly

### Global Daemon

The daemon owns:

- runtime state
- watcher lifecycle
- index manager lifecycle
- git refresh lifecycle
- Huey consumer lifecycle
- task/run metadata

## Filesystem Contract

The daemon must own three local artifacts:

1. **boot lock file** — exclusive lock for daemon startup
2. **metadata file** — JSON description of the active daemon
3. **socket file** — Unix-domain socket used by ZMQ transport

All three artifacts are global per user environment.

## Boot Lock Contract

### Lock Semantics

- lock type: filesystem lock using `fcntl`
- behavior: non-blocking acquisition with bounded retry loop
- timeout: configuration-backed; must fail cleanly with explicit error

### Lock Rules

1. only daemon boot/attach coordination uses the lock
2. project context must never key the lock path
3. stale clients do not release the lock; the owning process or OS does

## Daemon Metadata Contract

The daemon metadata record should be JSON and contain at minimum:

| Field | Type | Purpose |
|---|---|---|
| `pid` | int | process identity |
| `started_at` | float | startup timestamp |
| `status` | string | `starting`, `ready`, `degraded`, or `stopping` |
| `daemon_scope` | string | always `global` in v1 |
| `transport` | string | `zmq` |
| `socket_path` | string | transport endpoint backing file |
| `binary_path` | string\|null | executable provenance |
| `version` | string\|null | daemon version/protocol provenance |
| `index_db_path` | string | active shared store |
| `queue_db_path` | string | active Huey queue store |

### Metadata Rules

1. metadata is advisory until confirmed by a health probe
2. metadata mismatch with the live daemon invalidates attach
3. stale metadata must be cleaned during daemon recovery

## Transport Contract

### ZMQ Pattern

- daemon socket: `ROUTER`
- client socket: `DEALER`
- endpoint: `ipc://<socket_path>`
- permissions: owner-only socket file

### Request Envelope

Every client request should serialize to one JSON object:

```json
{
  "request_id": "uuid-or-monotonic-id",
  "path": "/api/search/query",
  "payload": {},
  "client": {
    "kind": "mcp|cli|http",
    "pid": 12345,
    "project_context": "optional-project-id"
  }
}
```

### Response Envelope

```json
{
  "request_id": "same-as-request",
  "status": "ok|error",
  "data": {},
  "error": null
}
```

### Transport Rules

1. path-based dispatch is the stable abstraction
2. payloads must remain JSON-serializable
3. timeouts must normalize to explicit client-visible timeout errors
4. health probing uses the same transport, not side channels

## Required Control Paths

The daemon must support these minimum paths:

| Path | Purpose |
|---|---|
| `/internal/health` | liveness and metadata verification |
| `/internal/shutdown` | graceful daemon stop |
| `/api/admin/overview` | summary status for CLI |
| `/api/admin/tasks` | queue/task inspection |
| `/api/admin/index` | corpus/index stats |
| `/api/search/query` | document search |
| `/api/search/git-history` | git history search |

## Health Payload Contract

The health payload should include at minimum:

- daemon status
- daemon scope
- runtime active flag
- client count
- queue enabled flag
- search/index health summary
- socket path
- PID
- version

This structure should follow the spirit of `mcp-memory`'s management payloads even though Ragdocs will omit memory-specific fields.

## Client Startup Contract

Client startup should follow this sequence:

1. read daemon metadata if present
2. probe `/internal/health` if metadata exists
3. if healthy, attach and proceed
4. if unhealthy or missing, acquire boot lock
5. under lock, re-check metadata/health to avoid thundering herd
6. spawn daemon if still absent
7. wait for healthy metadata and attach

## Failure Semantics

| Failure | Client Behavior |
|---|---|
| stale metadata, no live daemon | clean metadata, attempt startup |
| stale socket, no live daemon | remove socket, attempt startup |
| boot lock timeout | return explicit daemon-start timeout error |
| transport timeout | return explicit request timeout error |
| metadata/health mismatch | treat daemon as invalid and restart attach flow |

## Files Likely To Change In Ragdocs

- `src/cli.py`
- `src/mcp/server.py`
- `src/lifecycle.py`
- `src/context.py`
- new daemon/transport modules

## Acceptance Criteria

1. multiple clients attach to one daemon safely
2. project context never affects daemon identity
3. metadata + health probe are sufficient to distinguish healthy vs stale daemon state
4. all control-plane requests route through one ZMQ contract