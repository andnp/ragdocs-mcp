# Implementation Plan: Tranche 1 Memory Removal

**Status:** Partially complete — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/plans/01-memory-removal-and-surface-simplification.md`, `docs/plans/03-tranche-implementation-roadmap.md`

## Executive Summary

This document turns Tranche 1 into a file-by-file implementation checklist. The goal is not just to delete `src/memory/`, but to remove every runtime, config, docs, and test dependency that assumes Ragdocs is a memory product.

## Verified Checklist Status (2026-03-17)

- **A. Delete Runtime Modules:** mostly complete. `src/memory/` runtime files are gone.
- **B. Delete MCP Memory Surface:** complete.
- **C. Remove Context/Lifecycle State:** complete.
- **D. Remove Config Surface:** complete.
- **E. Remove Schema Surface:** effectively complete for memory-only tables; current schema no longer exposes memory-specific tables.
- **F. Remove Worker/Queue Coupling To Memory:** complete for memory-specific logic, but this area still needs broader V2 task ownership work.
- **G. Remove Docs:** not complete.
- **H. Remove Tests:** complete for active runtime coverage, but docs/spec references still preserve the historical memory story.

## Mandatory Reference Rule

Future agents should still inspect `/home/andy/Projects/personal/mcp-memory` before changing adjacent daemon or admin code during this tranche. The purpose here is not to borrow memory features, but to avoid reintroducing them while preparing the new control plane.

## Tranche Goal

Hard-delete all memory features while keeping document search and git history search green.

## File-by-File Checklist

### A. Delete Runtime Modules

Delete the entire `src/memory/` package:

- `src/memory/__init__.py`
- `src/memory/consolidation.py`
- `src/memory/init.py`
- `src/memory/journal.py`
- `src/memory/link_parser.py`
- `src/memory/maintenance.py`
- `src/memory/manager.py`
- `src/memory/models.py`
- `src/memory/providers.py`
- `src/memory/search.py`
- `src/memory/storage.py`
- `src/memory/tools.py`

### B. Delete MCP Memory Surface

Edit:

- `src/mcp/server.py`
- `src/mcp/tools/memory_tools.py` or equivalent registration layer

Actions:

1. remove memory tool imports
2. remove memory tool registration from tool lists
3. remove memory handler imports and dispatch wiring

### C. Remove Context/Lifecycle State

Edit:

- `src/context.py`
- `src/lifecycle.py`

Actions:

1. remove memory manager/search fields from `ApplicationContext`
2. remove memory initialization paths from startup
3. remove memory maintenance hooks from shutdown and background bootstrap

### D. Remove Config Surface

Edit:

- `src/config.py`

Actions:

1. delete `MemoryRecencyConfig`
2. delete `MemoryConfig`
3. remove `memory` from `Config`
4. remove memory config parsing helpers
5. remove memory defaults from config load and validation

### E. Remove Schema Surface

Edit:

- `src/storage/db.py`

Actions:

1. remove `system1_journal`
2. remove memory-only task/state tables if they are no longer needed after Huey ownership is clarified
3. keep `tasks` and `system_state` only if still needed for Ragdocs V2 control plane

### F. Remove Worker/Queue Coupling To Memory

Edit:

- `src/worker/consumer.py`
- `src/indexing/tasks.py`
- `src/coordination/queue.py` if memory-specific comments or task families remain

Actions:

1. remove memory-related task definitions
2. remove memory bootstrap or recurring task registration
3. keep only document/git/task orchestration concepts

### G. Remove Docs

Delete or rewrite:

- `docs/memory.md`
- `docs/specs/16-memory-management.md`
- `docs/specs/22-memory-system-independence.md`
- `docs/specs/24-autonomous-memory.md`

Rewrite references in:

- `README.md`
- `docs/architecture.md`
- `docs/integration.md`
- `docs/migration-v1.5.md`

### H. Remove Tests

Delete or rewrite memory-only tests and fixtures.

Minimum audit areas:

- `tests/unit/`
- `tests/integration/`
- `tests/e2e/`
- `tests/conftest.py`

Targets:

1. delete memory CRUD/search tests
2. delete autonomous-memory/maintenance tests
3. remove fixtures that only exist for memory runtime setup

## Ordered Execution Sequence

1. remove MCP/public memory surface
2. remove context/lifecycle/config imports
3. delete `src/memory/`
4. clean worker/task/schema leftovers
5. remove docs and tests
6. run focused verification for document/git search paths

## Verification Checklist

### Static / Import Verification

- no imports of `src.memory` remain
- no memory tools appear in the MCP tool list
- config loads without memory sections

### Behavioral Verification

- document query tools still work
- git history search still works
- startup no longer initializes memory runtime state

### Documentation Verification

- README no longer describes memory features
- architecture docs no longer present memory as first-class runtime scope

## Refusal Rules

The tranche is not complete if any of the following remain:

- memory-specific config exposed to users
- memory-specific runtime state in `ApplicationContext`
- memory-specific task families in workers/queue plumbing
- memory feature references in public docs without explicit deprecation/removal language

## Acceptance Criteria

1. `src/memory/` is gone
2. Ragdocs exposes no memory-product surface
3. document search and git history search still function
4. no dead memory references remain in code or docs