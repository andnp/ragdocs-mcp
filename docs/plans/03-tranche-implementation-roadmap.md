# Plan: Tranche-by-Tranche Implementation Roadmap

**Status:** In Progress — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/specs/25-ragdocs-product-principles.md`, `docs/plans/00-ragdocs-v2-refactor-overview.md`, `docs/plans/01-memory-removal-and-surface-simplification.md`, `docs/plans/02-global-daemon-huey-and-soft-projects.md`

## Executive Summary

This roadmap turns the approved architecture direction into an execution sequence with small, reviewable tranches. The ordering is deliberate: first reduce scope, then establish the new control plane, then move work ownership behind durable tasks, then add operator visibility, then land soft-project ranking. The rule is simple: delete complexity before building new complexity on top of it.

## Verified Tranche Status (2026-03-17)

- **Tranche 0 — Guardrails & Inventory:** partially complete. The planning package exists, but it was not updated with implementation status before this pass.
- **Tranche 1 — Hard Delete Memory Product Surface:** partially complete. Runtime/code cleanup landed; docs cleanup is still outstanding.
- **Tranche 2 — Daemon Metadata, Boot Lock, and Thin-Client Skeleton:** partially complete. Metadata, lock, status helpers, and client forwarding exist, but the transport implementation differs from the ZMQ target and the global-store contract is not complete.
- **Tranche 3 — Huey Ownership Transfer:** partially complete. Production daemon wiring now exists for document tasks; git refresh migration and richer task/admin visibility are still in progress.
- **Tranche 4 — CLI-First Operations Surface:** partially complete. Daemon lifecycle commands, `queue status`, and `index stats` exist; deeper task inspection remains missing.
- **Tranche 5 — Soft Project Semantics:** not started.
- **Tranche 6 — Cleanup & Hardening:** not started.

## Tranche 0 — Guardrails & Inventory

### Objectives

1. freeze the approved product principles
2. inventory all memory imports, tools, docs, and tests
3. inventory all startup/control-plane entry points

### Deliverables

- import/dependency inventory for `src/memory/`
- public-tool inventory showing what disappears
- daemon/startup map showing current owners of lifecycle and background work

### Exit Criteria

- no unknown memory or lifecycle coupling remains in the implementation plan

## Tranche 1 — Hard Delete Memory Product Surface

### Objectives

1. remove memory MCP tools and public references
2. remove memory docs from the active product story
3. simplify config/context/lifecycle around the reduced scope

### Primary Files

- `src/memory/` (delete)
- memory tool registration files
- `src/context.py`
- `src/lifecycle.py`
- `src/config.py`
- memory docs/specs/tests

### Verification

- document MCP tool list no longer exposes memory tools
- document search and git history search still run
- import graph is clean after deleting `src/memory/`

## Tranche 2 — Daemon Metadata, Boot Lock, and Thin-Client Skeleton

### Objectives

1. define the global daemon metadata contract
2. add daemon boot lock and stale-daemon cleanup rules
3. create thin-client attach/start logic for CLI and MCP entry points

### Primary Files

- new daemon modules under `src/daemon/` or equivalent
- new ZMQ client/server transport under `src/ipc/` or `src/daemon/`
- `src/cli.py`
- `src/mcp/server.py`
- `src/lifecycle.py`

### Verification

- repeated client startup converges on one daemon
- stale metadata or dead sockets are cleaned safely
- clients can query daemon status without owning background work

## Tranche 3 — Huey Ownership Transfer

### Objectives

1. make the daemon the only Huey consumer owner
2. move watcher-triggered work to enqueue-only behavior
3. route indexing, reconciliation, and git refresh through durable tasks

### Primary Files

- `src/coordination/queue.py`
- `src/worker/consumer.py`
- `src/indexing/tasks.py`
- `src/indexing/watcher.py`
- `src/indexing/manager.py`
- `src/git/*`

### Verification

- task restart safety
- no duplicate indexing from multiple clients
- task failure visibility and retry behavior

## Tranche 4 — CLI-First Operations Surface

### Objectives

1. expose daemon status/start/stop/restart
2. expose queue status and recent failures
3. expose index and corpus statistics

### Primary Files

- `src/cli.py`
- new CLI command modules if needed
- daemon management/status request handlers

### Verification

- operator can inspect daemon health from CLI only
- operator can inspect queue depth and recent failures from CLI only
- operator can inspect document/git corpus stats from CLI only

## Tranche 5 — Soft Project Semantics

### Objectives

1. store project metadata on indexed content
2. remove project as a storage or worker boundary
3. apply fixed 1.2x active-project uplift in ranking

### Primary Files

- `src/config.py`
- `src/storage/db.py`
- `src/indexing/manager.py`
- `src/search/fusion.py`
- `src/search/orchestrator.py`
- `docs/guides/multi-project-setup.md`

### Verification

- same corpus remains globally searchable
- active project uplifts matching results by 1.2x
- explicit project filtering remains opt-in only

## Tranche 6 — Cleanup & Hardening

### Objectives

1. remove transitional fallback paths
2. remove obsolete multiprocess/snapshot docs and code
3. tighten tests, docs, and operational guidance around the new model

### Verification

- no snapshot/multiprocess architecture remains as the primary design
- docs reflect the daemon + Huey + soft-project model consistently
- focused end-to-end multi-client smoke coverage exists

## Cross-Tranche Rules

1. no tranche may reintroduce memory-scope behavior
2. no tranche may allow clients to own background consumers once daemon ownership lands
3. every control-plane change must include observability hooks or status visibility
4. every tranche ends with docs and tests updated in the same change set
5. future agents should inspect `/home/andy/Projects/personal/mcp-memory` before changing daemon, transport, admin, or metadata contracts

## Suggested First Implementation Commit Set

1. inventory + delete memory public tool surface
2. delete memory runtime/config/docs/tests
3. add daemon metadata and boot-lock scaffolding
4. add ZMQ transport skeleton with status request
5. move watcher actions to Huey enqueue path
6. add CLI daemon/status commands
7. add project metadata storage + 1.2x uplift

## Reference Implementation Checklist

Before starting any tranche in the control plane, read the matching `mcp-memory` references:

| Ragdocs Area | Minimum mcp-memory Reference |
|---|---|
| Global daemon boot | `src/mcp_memory/daemon.py`, `src/mcp_memory/daemon_lifecycle.py` |
| ZMQ transport | `src/mcp_memory/daemon_transport.py` |
| CLI operator surface | `src/mcp_memory/cli.py` |
| Metadata and health payloads | `src/mcp_memory/daemon_models.py`, `src/mcp_memory/management/models.py` |
| Global metadata semantics | `docs/specs/00-product-principles.md`, `docs/specs/01-architecture-principles.md` |

## Acceptance Criteria

1. each tranche is independently reviewable
2. the first implementation work starts with deletion, not new runtime features
3. the final architecture matches the approved principles without fallback ambiguity