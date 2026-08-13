# Google Drive indexing durability

**Status:** implementation in progress

## Goal

Google Drive indexing must survive process restarts, duplicate work, provider
failures, permission changes, and more than one worker without serving stale
or unauthorized records. The Drive source remains logically isolated by
`source_kind="gdrive"`, while using the application's shared record index.

## Durable state

Each configured Drive scope is identified by `(source_kind, workspace_id,
scope_identity)`. SQLite is the source of truth for:

- inventory and changes cursors;
- evaluated sync status and last error;
- per-scope record membership;
- bounded backfill progress; and
- push-watch channel state.

The database uses WAL mode, a busy timeout, schema versioning, and short
`BEGIN IMMEDIATE` transactions. A cursor advances only after the matching
index mutation succeeds. Malformed or unsupported state is treated as an
explicit recovery condition; it must not silently authorize records.

## Record replacement

Drive updates are replacements, not appends. A replacement is keyed by the
canonical Drive source identity and contains the complete current chunk set.
The operation removes stale chunk keys, deduplicates source-map entries, and
removes tombstoned records. Replay of the same replacement is a no-op. When
the index store and SQLite state cannot share a transaction, a replacement
journal records the intended operation and is replayed during startup before
new Drive work is accepted.

## Scope membership

Membership is tracked independently for every configured scope. A record is
searchable only when it has at least one current membership in the caller's
authorized workspace. Losing access in one scope removes that scope's
membership; it removes the indexed record only when no other scope retains the
record. A transient provider error never removes membership. Definitive
`404`/`403` results are reconciled as permission loss only when the provider
operation establishes that interpretation.

## Lifecycle and leases

The lifecycle scheduler emits deduplicated work for startup inventory, change
polling, retry, backfill, watch renewal, and health evaluation. Startup work
is safe to enqueue more than once. Each scope has one durable lease owner;
long-running provider work heartbeats the lease and checks ownership before
publishing mutations. A lost lease aborts the operation without advancing its
cursor.

## Health

Health is derived from persisted per-scope sync outcomes rather than from the
fact that a health task ran. Aggregate precedence is deterministic:

`unavailable` > `acl-incomplete` > `stale` > `empty` > `healthy`.

The result includes scope-level errors and freshness, so operators can
distinguish an empty corpus from a source that has not synchronized.

## Verification contract

The acceptance suite covers:

1. restart and journal replay after interrupted replacement;
2. duplicate replacement and overlapping-scope membership;
3. permission loss, transient failures, and definitive tombstones;
4. bounded media reads and disabled-mode startup/teardown;
5. scheduler deduplication, lease heartbeat, and concurrent workers;
6. native scoped retrieval filters and health transitions; and
7. full Ruff, Pyright, Pyrefly, and repository test verification.

The operator procedure, rollback window, and credential ownership remain in
the [Google Drive operator runbook](../guides/gdrive-operator-runbook.md).
