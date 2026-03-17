# Product Principles: Ragdocs Refactor

**Status:** Accepted — guiding implementation as of 2026-03-17
**Date:** 2026-03-16
**Supersedes Direction In:** `docs/memory.md`, `docs/specs/16-memory-management.md`, `docs/specs/21-multiprocess-architecture.md`, `docs/specs/23-concurrency-huey.md`, `docs/guides/multi-project-setup.md`

## 1. Product Scope

`mcp-markdown-ragdocs` does two things:

1. index and search documentation
2. search git history

The repository does **not** own a persistent memory product, autonomous memory agents, or a memory CRUD surface.

## 2. One Global Runtime Authority

Ragdocs should run as one global daemon per user environment.

- daemon identity must not vary by editor session
- daemon ownership must not vary by client type
- thin clients connect to one shared runtime authority

## 3. One Shared Search Store

The runtime store is shared.

- document indices, git history indices, task state, telemetry, and runtime metadata live in one shared local authority
- clients are not storage owners
- per-client background workers are forbidden

## 4. Thin Clients, Stateful Daemon

MCP sessions, CLI commands, and optional HTTP/admin surfaces are thin clients.

- clients translate user requests into daemon requests
- the daemon owns file watching, indexing, scheduled tasks, and long-lived models
- client startup should prefer attachment to an existing daemon over spawning new process trees

## 5. Persistent Tasks, Not Ad-Hoc Background Work

Background work must be durable.

- indexing, reconciliation, git refresh, and maintenance tasks must survive client restarts
- duplicate work should be coalesced through one persistent task layer
- transient in-memory queues are not the system of record

## 6. Projects Are Metadata, Not Hard Partitions

Project context is descriptive metadata.

- documents may belong to one or more projects
- project context may boost ranking when the active client context matches
- project context must not imply storage isolation, daemon isolation, worker isolation, or query hard-filtering by default

## 7. Search Is Global-First

Search operates across the full indexed corpus.

- the full document and git corpus remains searchable
- active project context may apply a bounded score uplift
- project context must not silently remove results unless an explicit filter requests it

## 8. Admin Visibility Is Mandatory

Because the daemon becomes long-lived, operators need first-class inspection.

- daemon health, uptime, and ownership must be inspectable
- queue depth, task failures, and recent runs must be inspectable
- index health and corpus statistics must be inspectable

## 9. Migration Direction

When existing code conflicts with these principles:

1. product scope reduction wins over backward compatibility for memory features
2. global daemon semantics win over per-client or per-project worker semantics
3. durable task orchestration wins over custom transient coordination
4. soft project weighting wins over hard project segmentation

## 10. Explicit Non-Goals

This refactor does **not** aim to:

- preserve memory MCP tools under a deprecated namespace
- preserve the snapshot-based multiprocess design
- preserve per-project isolated indices as the primary architecture
- introduce remote infrastructure such as Redis or RabbitMQ