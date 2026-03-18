# Plan: Converge to One Global Daemon, One Global Corpus Runtime

**Status:** Proposed — prompted by live runtime validation on 2026-03-18
**Date:** 2026-03-18
**Related:** `docs/specs/25-ragdocs-product-principles.md`, `docs/plans/02-global-daemon-huey-and-soft-projects.md`, `docs/plans/03-tranche-implementation-roadmap.md`, `docs/plans/04-daemon-zmq-control-plane-contract.md`, `docs/plans/05-huey-task-and-cli-contract.md`, `docs/plans/07-global-store-and-schema-contract.md`, `docs/plans/08-search-and-ranking-contract.md`, `specs/10-multi-project-support.md`

## Executive Summary

Ragdocs already documents the intended product model clearly:

1. one global daemon per user environment
2. one shared local search store
3. thin clients that forward requests to the daemon
4. projects as metadata for ranking/filtering, not runtime/storage isolation

The implementation is **close**, but not fully converged. The daemon is global in process identity, socket path, and runtime state directory, yet it still often constructs a **single project-scoped `ApplicationContext`** at startup. In practice, that means the first client to start the daemon can determine which corpus receives background indexing, watcher coverage, reconciliation, and durable task ownership.

This plan closes that gap. The goal is to make the daemon truly global in **corpus ownership**, not just infrastructure identity.

## Why this plan now

A live operational review on 2026-03-18 confirmed the mismatch between the product principles and the runtime behavior:

- the daemon was healthy and global at `~/.local/state/mcp-markdown-ragdocs/daemon`
- the active runtime indexed only the currently attached repo corpus
- other configured corpuses were not receiving background indexing or watcher coverage
- user expectation was exactly what the product principles say: a single global daemon, with per-request project context affecting ranking only

This is therefore not a speculative redesign. It is a convergence task to align code with already accepted architecture.

## Lift estimate

**Estimate:** Moderate

### Why it is not small

The change crosses the control-plane spine:

- daemon bootstrap semantics
- `ApplicationContext` construction
- worker startup assumptions
- watcher/task ownership
- admin/index visibility
- migration and regression coverage

### Why it is not large

Most of the target architecture is already present:

- one global daemon runtime path exists
- thin-client forwarding exists
- global runtime metadata and status commands exist
- project metadata is already stamped during indexing/query paths
- search already supports bounded project uplift and explicit project filters

The main remaining work is **removing the last runtime-scoping assumption**.

## Current mismatch

### Product principle

Per `docs/specs/25-ragdocs-product-principles.md`:

- daemon identity must not vary by editor session
- daemon ownership must not vary by client type
- projects are metadata, not hard partitions
- search is global-first

### Runtime reality

Today the daemon startup path still flows through a project-selected `ApplicationContext`:

- thin clients may pass `--project` / detected project into daemon startup
- daemon startup may inherit that scope
- `ApplicationContext.create(project_override=...)` resolves one runtime root set
- the daemon-owned worker/watcher/task layer inherits that same runtime scope

So the daemon is global in **PID/socket/store location**, but often single-corpus in **live runtime ownership**.

## Architectural seam to fix

The critical seam is:

- `src/cli.py::_run_daemon_forever`
- `src/cli.py::_create_daemon_runtime`
- `src/daemon/management.py::start_daemon`
- `src/context.py::ApplicationContext.create`
- `src/worker/process.py`

Today these layers still allow client project context to influence daemon runtime construction.

### Rule that should replace it

> Project context should enter at **request ranking/filter boundaries**, not at **daemon/runtime construction**.

In other words:

$$
\text{daemon scope} = \text{global corpus}
$$

while

$$
\text{request project context} = \text{ranking/filter metadata only}
$$

## Target state

### Runtime model

- one daemon per user environment
- one daemon-owned global `ApplicationContext` covering all configured roots by default
- one global watcher/task/worker authority
- one shared local store for documents, git history, runtime metadata, and queue state

### Request model

- clients send optional `project_context`
- clients may send explicit `project_filter`
- default search remains global-first
- ranking uplift remains bounded and conservative

### Operator model

- `daemon status` reflects global daemon scope clearly
- `index stats` reports configured roots and aggregate corpus state
- admin surfaces do not imply a single active corpus unless an explicit filter requests one

## Phased implementation plan

### Phase 1 — Make daemon identity fully project-agnostic

**Goal:** prevent project selection from shaping daemon/process identity.

#### Deliverables

- daemon startup treats `project_override` as client/request context, not daemon scope
- daemon metadata explicitly describes a global runtime authority
- attach/start logic is documented as project-agnostic

#### Primary modules

- `src/daemon/management.py`
- `src/cli.py`
- `src/mcp/server.py`
- `src/lifecycle.py`

### Phase 2 — Separate global runtime scope from request context

**Goal:** build the daemon runtime over the configured global corpus.

#### Deliverables

- daemon runtime creation path always uses the global configured corpus roots
- request-time `project_context` and `project_filter` remain supported without changing runtime shape
- ambient `detected_project` stops leaking into daemon-owned corpus construction

#### Primary modules

- `src/context.py`
- `src/config.py`
- `src/search/orchestrator.py`
- `src/cli.py`

### Phase 3 — Globalize watcher, worker, and task ownership

**Goal:** make background work truly global across the configured corpus.

#### Deliverables

- daemon-owned watcher covers all configured roots
- worker subprocess launches in global-corpus mode
- Huey task handlers assume one authoritative global manager, not one project-selected manager
- git refresh/index maintenance follows the same scope rules

#### Primary modules

- `src/worker/process.py`
- `src/worker/consumer.py`
- `src/indexing/watcher.py`
- `src/indexing/tasks.py`
- `src/indexing/manager.py`
- `src/git/watcher.py`
- `src/git/commit_indexer.py`

### Phase 4 — Align storage and migration semantics

**Goal:** finish the global-store story and make legacy behavior explicit.

#### Deliverables

- authoritative store semantics documented and enforced
- legacy per-project isolated artifacts are treated as migration inputs, not active architecture
- rebuild and reconciliation semantics default to the global corpus unless narrowed explicitly
- doc identity and manifest handling remain stable across multi-root operation

#### Primary modules

- `src/storage/db.py`
- `src/indexing/manifest.py`
- `src/indexing/reconciler.py`
- `src/search/path_utils.py`
- `src/config.py`

### Phase 5 — Tighten admin/query semantics

**Goal:** ensure visibility surfaces tell the truth about the new runtime model.

#### Deliverables

- `daemon status` shows global scope and configured root counts
- `index stats` exposes aggregate counts and optional filtered views
- query/admin commands stop implying a single active corpus at runtime
- thin clients consistently treat project as request metadata only

#### Primary modules

- `src/cli.py`
- `src/mcp/server.py`
- `src/daemon/health.py`
- `src/daemon/metadata.py`
- `src/daemon/queue_status.py`

### Phase 6 — Regression coverage and doc cleanup

**Goal:** lock in the invariant and remove contradictory guidance.

#### Deliverables

- regression test for the “first client wins” failure mode
- multi-client tests with different project contexts against one daemon
- daemon-backed multi-root indexing/query coverage
- stale single-project-server guidance updated, superseded, or archived

#### Primary areas

- `tests/integration/`
- `tests/e2e/`
- `tests/regression/`
- `docs/guides/multi-project-setup.md`
- `docs/plans/02-global-daemon-huey-and-soft-projects.md`
- `docs/plans/03-tranche-implementation-roadmap.md`
- `docs/plans/04-daemon-zmq-control-plane-contract.md`
- `docs/plans/05-huey-task-and-cli-contract.md`
- `docs/plans/07-global-store-and-schema-contract.md`
- `docs/plans/08-search-and-ranking-contract.md`
- `specs/10-multi-project-support.md`

## Risks

### 1. Background indexing load increases

A truly global watcher/task runtime may watch and reconcile more roots than today.

**Mitigation:**
- keep existing backpressure/debounce controls
- surface root counts and queue pressure in admin views
- measure before widening any ranking or scheduling behavior

### 2. Legacy assumptions survive in thin seams

Single-root assumptions may still exist in reconciliation, task payloads, or admin summaries.

**Mitigation:**
- characterize current multi-root behavior
- add explicit regression tests before broadening runtime scope

### 3. Migration surprises for users with stale share-store history

Users may still have old per-project artifacts or stale docs that imply the older architecture.

**Mitigation:**
- prefer explicit migration notes over silent magic
- default to rebuild/reconciliation when correctness is uncertain

### 4. Query semantics drift

Project uplift/filtering could accidentally reintroduce hard partitioning at query time.

**Mitigation:**
- preserve global-first defaults
- keep explicit filters opt-in
- make ranking uplift bounded and observable

## Acceptance criteria

1. exactly one daemon can serve multiple clients from different project contexts without changing runtime corpus ownership
2. background indexing, watchers, and durable tasks span the configured global corpus
3. request `project_context` affects ranking only, not daemon scope
4. admin/index surfaces accurately report the global corpus model
5. old single-project daemon assumptions are removed or explicitly archived

## Suggested implementation order

If executed as code work, the lowest-risk sequence is:

1. characterization tests for current first-client-wins behavior
2. daemon bootstrap refactor to stop project-scoping runtime creation
3. global watcher/worker/task convergence
4. admin/index visibility updates
5. docs and migration cleanup

## Decision

Proceed with this as a **convergence refactor**, not a speculative redesign.

The architecture you want is already the accepted product direction. The work now is to make the runtime obey it consistently.