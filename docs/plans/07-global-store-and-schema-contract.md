# Contract: Global Store, Paths, and Schema Layout

**Status:** Draft
**Date:** 2026-03-16
**Related:** `docs/plans/04-daemon-zmq-control-plane-contract.md`, `docs/plans/05-huey-task-and-cli-contract.md`

## Executive Summary

This document defines the persistent storage contract for Ragdocs V2: which files exist, where they live, what they contain, and which runtime concepts are global versus metadata-only. It is the storage companion to the daemon and Huey contracts.

## Mandatory Reference Rule

Future agents should inspect `/home/andy/Projects/personal/mcp-memory` for daemon metadata and global-store patterns before editing storage layout or path semantics here.

## Global Path Contract

Ragdocs V2 should use one global runtime root per user environment.

Within that root, reserve explicit paths for:

| Artifact | Purpose |
|---|---|
| `index.db` | primary SQLite store for documents, chunks, graph, git state, system state |
| `queue.db` | Huey durable task queue |
| `daemon.json` | daemon metadata |
| `daemon.lock` | boot/startup lock |
| `daemon.sock` | ZMQ socket |
| `logs/` | optional runtime logs if file-backed logging is retained |

Project context must not alter these authoritative paths.

## Storage Principles

1. one global store
2. one global queue
3. project is metadata attached to content, not a directory boundary
4. daemon metadata is not the source of truth for indexed content, only for runtime ownership

## Schema Contract

### Keep

The V2 runtime should retain tables for:

- `documents`
- `chunks`
- `kv_store`
- `search_index`
- `graph_nodes`
- `graph_edges`
- `tasks` or Huey-required equivalents
- `system_state` if still needed after leader-election removal

### Remove

The V2 runtime should remove:

- `system1_journal`
- any memory-only relational structures added outside the current `DatabaseManager`

## Required Table Changes

### `documents`

Add or standardize fields for:

- `project_id TEXT NULL`
- `source_kind TEXT DEFAULT 'document'` if helpful for git/document distinction
- `last_index_reason TEXT NULL` for observability

### `chunks`

Add or standardize fields for:

- `project_id TEXT NULL`
- `source_file TEXT NULL` if not already derivable from metadata

### `tasks`

Either preserve or adapt the table so the operator/admin surface can answer:

- pending count
- running count
- failed count
- recent failures
- task type distribution

If Huey owns the queue schema entirely, Ragdocs must still define a read model or adapter for CLI status commands.

### `system_state`

Continue using `system_state` only for global runtime state that survives restarts, such as:

- schema version
- last successful git refresh timestamps
- last successful reconciliation timestamps
- daemon/runtime migration markers

Do **not** use it for per-project execution ownership.

## Metadata Rules

### `project_id`

- optional metadata field
- derived at ingest time
- used for ranking uplift and explicit filtering only

### `detected_project`

- client context only
- not part of daemon identity
- not part of storage partition identity

## Current Code Impact

### Files That Must Change

- `src/storage/db.py`
- `src/config.py`
- `src/context.py`
- `src/indexing/manager.py`
- `src/git/*` where commit index state is stored

### Config Simplification

`ProjectConfig` should stop implying separate index roots. V2 config should treat projects as metadata sources with optional path mapping only.

## Path Resolution Rules

1. active project influences ranking context, not the authoritative DB/queue/socket locations
2. daemon metadata, queue DB, and index DB are global
3. per-project documents may still be discovered from configured roots, but they land in the same store

## Migration Rules

1. if old configs specify per-project index paths, V2 ignores them after migration and emits explicit migration notes
2. if old data contains memory-era schema artifacts, migration removes or ignores them deterministically
3. schema upgrades must be idempotent

## Acceptance Criteria

1. one global runtime root holds the authoritative store, queue, metadata, lock, and socket
2. project context exists only as metadata columns and search context
3. memory-era storage artifacts are removed
4. the storage contract is stable enough for CLI/admin and daemon contracts to depend on it