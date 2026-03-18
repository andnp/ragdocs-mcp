# Contract: Huey Task Ownership & CLI Operations Surface

**Status:** Partially complete — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/plans/02-global-daemon-huey-and-soft-projects.md`, `docs/plans/03-tranche-implementation-roadmap.md`, `docs/plans/04-daemon-zmq-control-plane-contract.md`

## Executive Summary

This document defines two contracts that must land together: durable task ownership via Huey, and the CLI-first operator surface for inspecting and controlling the daemon. The first release keeps operations intentionally simple: start, stop, status, queue status, and index stats. No dashboard ships in v1.

## Verified Implementation Status (2026-03-17)

- **Implemented:** `SqliteHuey` wrapper in `src/coordination/queue.py`, document indexing/removal tasks in `src/indexing/tasks.py`, git refresh task scaffolding, and worker lifecycle in `src/worker/consumer.py`.
- **Implemented:** daemon lifecycle commands in `src/cli.py`, plus `queue status` and `index stats` support.
- **Implemented production wiring:** daemon startup now initializes `get_huey()`, registers tasks, creates `HueyWorker`, and enables task-based document watching.
- **Missing task families:** no `rebuild_git_index`, `reconcile_corpus`, `rebuild_search_artifacts`, or integrity-task implementations yet.
- **Still limited in CLI/admin surface:** queue inspection exists, but richer task detail and broader admin inspection remain absent.

## Mandatory Reference Implementation

Future agents should inspect `/home/andy/Projects/personal/mcp-memory` before changing the operator surface or task ownership semantics.

Minimum references:

- `src/mcp_memory/cli.py`
- `src/mcp_memory/management/service.py`
- `src/mcp_memory/management/models.py`

Use those files as the reference for payload density, admin ergonomics, and global-daemon assumptions. Adapt to Ragdocs' smaller scope.

## Goals

1. daemon is the only Huey consumer owner
2. background work survives client restarts
3. operators can inspect and control the daemon from the CLI
4. task failures remain visible and actionable

## Non-Goals

1. build a dashboard in v1
2. expose internal queue mechanics directly to MCP end users
3. allow clients to execute background tasks inline as a normal path

## Task Ownership Contract

### Ownership Rule

- thin clients may enqueue
- only the daemon may consume
- only the daemon may schedule recurring tasks

### Queue Storage

- backend: `SqliteHuey`
- queue database path: global, alongside daemon metadata/store paths
- queue path must not vary by project context

## Required Task Families

### Document Indexing

- `index_document`
- `delete_document`
- `reconcile_corpus`

### Git Search Maintenance

- `refresh_git_repository`
- `rebuild_git_index`

### Integrity / Maintenance

- `rebuild_search_artifacts`
- `run_integrity_check`
- `prune_runtime_state` if needed

## Task Payload Contract

Every task payload should contain explicit identity fields instead of relying on ambient context.

### Required Envelope Fields

| Field | Purpose |
|---|---|
| `task_id` | durable task identity |
| `task_name` | task type |
| `enqueued_at` | observability |
| `requested_by` | `daemon`, `mcp`, or `cli` |
| `project_context` | optional ranking context only |
| `payload` | task-specific arguments |

### Examples

#### Index Document

```json
{
  "task_id": "...",
  "task_name": "index_document",
  "requested_by": "daemon",
  "project_context": "repo-a",
  "payload": {
    "doc_path": "/abs/path/to/file.md",
    "doc_id": "optional-stable-id",
    "reason": "watcher_change"
  }
}
```

#### Git Refresh

```json
{
  "task_id": "...",
  "task_name": "refresh_git_repository",
  "requested_by": "daemon",
  "project_context": null,
  "payload": {
    "repo_path": "/abs/path/to/repo",
    "after_timestamp": 1234567890
  }
}
```

## Task Semantics

### Idempotency

All handlers must be idempotent.

- repeated indexing of the same file must overwrite safely
- repeated delete of the same document must succeed as a no-op
- repeated git refresh windows must not duplicate commits

### Retry Semantics

- transient failures retry with bounded attempts
- permanent failures stay visible for inspection
- cancellations should be explicit, not silent

### Coalescing

Where practical, the daemon should coalesce repeated requests for the same target while a prior task is still pending.

## CLI Contract

### Required Commands in V1

| Command | Behavior |
|---|---|
| `ragdocs daemon start` | ensure daemon is running |
| `ragdocs daemon stop` | graceful shutdown |
| `ragdocs daemon restart` | stop then start |
| `ragdocs daemon status` | print daemon metadata and health |
| `ragdocs queue status` | print queue depth and recent failures |
| `ragdocs index stats` | print document/chunk/git counts and store paths |

### Output Rules

1. human-readable by default
2. stable JSON output should be available for machine consumption
3. commands must succeed without requiring the user to know queue internals

## Minimum Admin Payloads

### Daemon Status

Should include:

- status
- PID
- started_at
- socket path
- queue DB path
- index DB path
- client count
- search health summary

### Queue Status

Should include:

- pending task count
- running task count
- failed task count
- most recent failures
- task counts by type if cheap to compute

### Index Stats

Should include:

- indexed document count
- indexed chunk count
- git commit count
- watched repository count
- configured project count
- active project context if any

## Ragdocs Modules Most Likely To Change

- `src/coordination/queue.py`
- `src/worker/consumer.py`
- `src/indexing/tasks.py`
- `src/indexing/watcher.py`
- `src/indexing/manager.py`
- `src/git/*`
- `src/cli.py`

## Acceptance Criteria

1. clients can enqueue without owning the consumer
2. daemon owns all recurring/background work
3. operator can start/stop/status the daemon from CLI only
4. operator can inspect queue and index health from CLI only
5. task payloads are explicit, durable, and idempotent by contract
