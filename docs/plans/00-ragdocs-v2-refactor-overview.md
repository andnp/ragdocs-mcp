# Plan: Ragdocs V2 Refactor Overview

**Status:** In Progress — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/specs/25-ragdocs-product-principles.md`, `docs/specs/23-concurrency-huey.md`, `docs/specs/21-multiprocess-architecture.md`, `docs/specs/architecture-redesign.md`

## Executive Summary

Ragdocs has grown three competing architectural stories: hybrid document search, a memory subsystem, and a multiprocess coordination model. The user direction is now explicit: narrow the product to document indexing/search plus git history search, remove the memory subsystem entirely, replace the current coordination model with a global daemon and thin clients, use Huey for persistent work, and treat projects as ranking metadata rather than storage partitions. This plan defines the architecture delta, workstreams, milestones, and migration risks before implementation starts.

## Verified Implementation Status (2026-03-17)

- **Workstream A — scope reset:** partially complete. Runtime memory code and public memory MCP surface are gone, but memory-era docs still remain in `docs/memory.md`, `docs/specs/16-memory-management.md`, `docs/specs/22-memory-system-independence.md`, and `docs/specs/24-autonomous-memory.md`, and `README.md` still advertises memory features.
- **Workstream B — global daemon:** partially complete. Daemon metadata, boot lock, lifecycle helpers, and `daemon start|stop|status|restart` exist in `src/daemon/` and `src/cli.py`, but the transport is a Unix socket request server in `src/daemon/health.py`, not the ZMQ contract described here.
- **Workstream C — persistent task layer:** partially complete. Production daemon startup now initializes `SqliteHuey`, registers indexing tasks, starts `HueyWorker`, and enables task-based document watching. Git refresh task support is landing in the same tranche, but queue/task inspection and broader task families are still incomplete.
- **Workstream D — admin surface:** partially complete. Daemon lifecycle commands, `queue status`, and `index stats` are implemented; richer task/index inspection remains pending.
- **Workstream E — soft project semantics:** not started. `ApplicationContext` still resolves project-aware index paths, `src/storage/db.py` has no `project_id` columns, and ranking code has no $1.2\times$ project uplift.

## Goals

1. remove all memory-product behavior from ragdocs
2. adopt one global daemon with thin MCP/CLI clients
3. use Huey as the persistent task and scheduling substrate
4. add an operator-facing CLI/admin/status/dashboard surface
5. convert projects from hard boundaries into soft ranking uplift
6. preserve and improve document search and git history search during migration

## Locked Product Decisions

The following decisions are no longer open:

1. one global daemon and one global store per user environment
2. memory features are hard-deleted, not deprecated behind runtime stubs
3. the first admin surface is CLI-only
4. active-project uplift starts at **1.2x**

## Non-Goals

1. preserve backward compatibility for memory tools
2. preserve the snapshot-based multiprocess design
3. introduce remote infrastructure dependencies
4. redesign retrieval quality from scratch before the control plane is stabilized

## Why This Refactor Exists

### Current Pain

- multiple clients can still compete for background responsibility
- coordination complexity is high relative to product scope
- the memory subsystem inflates code surface, documentation surface, and runtime semantics
- hard project semantics over-segment the corpus and complicate startup logic
- there is no simple operator story for status, queue inspection, or daemon control

### Source Patterns Being Imported

From `mcp-memory`:

- global daemon authority
- ZMQ-based thin-client control path
- explicit daemon metadata and health inspection
- CLI/admin visibility patterns
- workspace-as-metadata semantics, adapted here as project-as-metadata semantics

### Reference Implementation Rule

Future agents working this refactor should consult `/home/andy/Projects/personal/mcp-memory` before making control-plane decisions.

Minimum reference set:

- `docs/specs/00-product-principles.md`
- `docs/plans/03-daemon-infrastructure-zmq-and-security.md`
- `src/mcp_memory/daemon.py`
- `src/mcp_memory/daemon_transport.py`
- `src/mcp_memory/cli.py`

Ragdocs is not expected to copy `mcp-memory` verbatim. The rule is to reuse the proven patterns and adapt them to the smaller product scope plus Huey-based task execution.

From existing ragdocs work:

- Huey draft in `docs/specs/23-concurrency-huey.md`
- existing hybrid search and git search pipeline
- existing application lifecycle and indexing manager abstractions

## Architecture Delta

| Area | Current | Target |
|---|---|---|
| Product scope | Documents + git + memory | Documents + git only |
| Runtime ownership | Client-local startup with complex coordination | One global daemon |
| IPC | Current lifecycle + worker coordination design | ZMQ thin client to daemon |
| Background work | Mixed direct calls and partial worker/task plumbing | Huey-backed persistent tasks |
| Project semantics | Harder segmentation and project activation | Global corpus with bounded project uplift |
| Operations | Limited CLI status/admin story | Explicit daemon/admin/status/dashboard surface |

## Major Workstreams

### Workstream A — Scope Reset

- delete `src/memory/`
- remove memory MCP tools and docs
- remove memory-specific config, lifecycle hooks, and task families

### Workstream B — Global Daemon

- add daemon-specific lifecycle modules
- turn MCP/CLI entry points into thin clients
- add daemon metadata, health checks, and socket lifecycle

### Workstream C — Persistent Task Layer

- finish Huey wiring
- move indexing, reconciliation, git refresh, and maintenance into durable tasks
- standardize retry, failure reporting, and observability

### Workstream D — Admin Surface

- add daemon commands: start, stop, status, restart
- add queue/status/index inspection commands
- add simple dashboard or HTTP admin surface backed by daemon state

### Workstream E — Soft Project Semantics

- stamp documents/chunks with project metadata
- apply a bounded search uplift for active-project matches
- remove project as an execution or storage boundary

## Proposed Milestones

### Milestone 0 — Documentation Lock-In

- approve product principles
- approve architecture plan
- approve removal and migration sequence

### Milestone 1 — Memory Removal

- remove memory code, config, tests, docs, and MCP surface
- keep document and git paths green

### Milestone 2 — Daemon Skeleton

- add global daemon identity, metadata, and thin-client startup path
- keep direct in-process execution as a temporary fallback only if needed for staged migration

### Milestone 3 — Huey Control Plane

- make indexing and git refresh task-driven
- daemon owns watcher and consumer lifecycle

### Milestone 4 — Admin Surface

- daemon status and queue inspection commands
- no dashboard in v1; HTTP/admin UI is deferred

### Milestone 5 — Soft Project Ranking

- unify storage
- add project metadata and ranking uplift
- remove hard project segmentation from the default mental model

## Recommended Document Set

This overview is the entry point. Detailed execution lives in:

1. `docs/plans/01-memory-removal-and-surface-simplification.md`
2. `docs/plans/02-global-daemon-huey-and-soft-projects.md`

## Acceptance Criteria

1. no user-facing memory tools remain in ragdocs
2. one daemon can serve multiple clients safely
3. indexing and git refresh are persistent and restart-safe
4. operators can inspect daemon and task state from the CLI
5. project context influences ranking without hard-partitioning the corpus
6. document search and git history search remain available throughout the migration

## Risk Register

| Risk | Severity | Why It Matters | Mitigation |
|---|---|---|---|
| Memory code shares indices and lifecycle assumptions | High | removal may break document paths indirectly | remove in a dedicated tranche with explicit context/lifecycle cleanup |
| Daemon migration can create mixed-runtime states | High | stale workers or old startup paths may coexist | define daemon metadata/provenance rules early and add migration checks |
| Huey integration can duplicate existing watchers or task triggers | High | duplicate indexing is expensive | daemon owns watcher; clients enqueue only |
| Project semantics may regress ranking quality | Medium | users expect local relevance | use bounded uplift and add evaluation coverage |
| Admin surface scope creep | Medium | can delay core refactor | keep v1 CLI-only: status/queue/index health only |

## Implementation Rule

No production implementation starts until the three planning docs and the governing product-principles spec are reviewed and accepted.