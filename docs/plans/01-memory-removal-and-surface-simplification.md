# Plan: Memory Removal & Surface Simplification

**Status:** Partially complete — verified against code on 2026-03-17
**Date:** 2026-03-16
**Related:** `docs/specs/25-ragdocs-product-principles.md`, `docs/specs/16-memory-management.md`, `docs/specs/24-autonomous-memory.md`

## Executive Summary

Ragdocs should stop pretending to be two products. This plan removes the entire memory subsystem so the repository returns to a focused scope: document indexing/search and git history search. The work includes deleting memory modules, removing memory MCP tools, simplifying config and lifecycle wiring, pruning obsolete docs/tests, and cleaning any shared runtime assumptions that were introduced to support memory-specific behavior.

## Verified Implementation Status (2026-03-17)

- **Completed:** `src/memory/` runtime code is effectively removed; only `__pycache__/` remains.
- **Completed:** public memory MCP tooling is gone; the active MCP surface in `src/mcp/` is document/git focused.
- **Completed:** memory config/runtime state no longer appears in `src/config.py`, `src/context.py`, or `src/lifecycle.py`.
- **Still pending:** documentation cleanup. `docs/memory.md`, `docs/specs/16-memory-management.md`, `docs/specs/22-memory-system-independence.md`, and `docs/specs/24-autonomous-memory.md` still exist, and `README.md` still references the removed memory product.

This plan should not be marked complete until the documentation and migration-story cleanup lands.

## Goals

1. remove memory code, docs, tests, and public interfaces
2. simplify `ApplicationContext`, lifecycle, CLI, and MCP registration
3. preserve document search and git history search behavior
4. reduce schema and task-surface complexity

## Locked Decision

Memory features will be **hard-deleted**.

- no compatibility stubs
- no deprecated MCP aliases
- no one-release grace period in code
- migration guidance lives in docs only

## Non-Goals

1. keep a deprecated compatibility shim for memory tools
2. migrate memory data forward inside this repository
3. redesign document retrieval in the same tranche

## Current Scope To Remove

### Code

- `src/memory/`
- memory MCP tool registration and handlers
- memory-specific config models and defaults
- memory-specific background tasks and maintenance loops

### Documentation

- `docs/memory.md`
- `docs/specs/16-memory-management.md`
- `docs/specs/22-memory-system-independence.md`
- `docs/specs/24-autonomous-memory.md`
- memory references in `README.md`, `docs/architecture.md`, `docs/integration.md`, and migration guides

### Tests

- unit/integration coverage dedicated to memory CRUD, memory search, and autonomous maintenance
- fixtures that only exist to support memory runtime behavior

## Expected Architecture Changes

### `src/context.py`

- remove memory manager/search state
- remove memory-path setup from startup
- remove memory-specific readiness and shutdown logic

### `src/lifecycle.py`

- remove memory maintenance orchestration
- remove memory-related recurring task bootstrap

### `src/config.py`

- remove memory dataclasses and validation
- simplify config loading and environment expectations

### `src/storage/db.py`

- remove tables and state that only exist for the memory product
- preserve document, graph, task, and git-search state needed by the reduced product

## Public API Changes

The following tool surface should disappear:

- create/read/update/delete memory operations
- memory search and memory relationship inspection
- autonomous-memory or maintenance-facing public endpoints

The surviving public tool surface should center on:

- query documents
- search git history
- daemon/admin/status operations once the new control plane lands

Any attempt to preserve memory endpoints during the refactor should be treated as scope violation.

## Migration Sequence

### Phase 1 — Freeze and Inventory

1. mark memory docs/specs as deprecated by plan
2. inventory all imports from `src/memory/`
3. inventory all MCP/public references to memory tools

### Phase 2 — Delete the Public Surface

1. remove memory tool registration
2. remove memory CLI/help/documentation references
3. update tests that assert tool lists or API contracts

### Phase 3 — Delete Runtime Wiring

1. strip memory fields from application context
2. remove memory lifecycle hooks and recurring work
3. simplify config and schema

### Phase 4 — Delete Implementation and Tests

1. remove `src/memory/`
2. remove memory-only fixtures/tests
3. update any shared helpers that assumed memory runtime coexistence

### Phase 5 — Final Documentation Cleanup

1. rewrite architecture docs around the reduced scope
2. rewrite README and integration docs
3. add migration notes for users moving memory workloads elsewhere

## Decision Matrix

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Keep memory behind a disabled flag | lower immediate breakage | leaves dead complexity everywhere | Reject |
| Move memory into `legacy/` inside ragdocs | easier rollback | preserves conceptual sprawl and maintenance tax | Reject |
| Delete memory completely from ragdocs | cleanest scope, smallest long-term surface | hard break for memory users | Adopt |

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Hidden imports from memory modules remain | High | run import-audit and delete tranche-by-tranche |
| Shared tables or helpers are still needed elsewhere | Medium | classify each shared artifact before deleting schema |
| Docs drift during deletion | Medium | rewrite docs in same tranche as code removal |
| Tests lose too much coverage | Low | replace memory-oriented smoke coverage with document/git control-plane coverage |

## Acceptance Criteria

1. `src/memory/` no longer exists
2. no public MCP memory tools remain
3. config no longer advertises memory settings
4. docs no longer describe ragdocs as a memory product
5. document and git search flows remain green