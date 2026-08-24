# Work-item delivery contract

**Status:** approved target contract for Workpackage 5

**Scope:** durable producer intent, Huey delivery, worker execution, retry
recovery, and queue-state reporting. This document defines the contract that
the implementation must satisfy; it does not change the current schema or
runtime behavior by itself.

## Authority and terms

`work_items` is the sole durable authority for Huey-dispatched work. Huey is a
delivery transport, not a second state ledger. The authoritative business
payload is stored in SQLite with the work item.

The existing split is documented in the current implementation at
[`work_intents.py`](../../mcp_markdown_ragdocs/coordination/work_intents.py),
[`task_leases.py`](../../mcp_markdown_ragdocs/coordination/task_leases.py), and
[`worker/consumer.py`](../../mcp_markdown_ragdocs/worker/consumer.py). The
approved target removes queue-task authority from that split while preserving
the public task and submission contracts.

The following identities are distinct:

- `work_item_id`: stable identity for one logical delivery item;
- `canonical_key`: producer deduplication key within an operation;
- `claim_token`: opaque owner credential carried by the current attempt; and
- `fencing_token`: monotonically increasing CAS version that invalidates every
  older owner after preemption or reclaim.

## Work-item state machine

The state machine is:

```text
pending
  -> dispatching
  -> queued
  -> running
  -> succeeded

running -> pending       retryable failure with backoff
running -> failed        non-retryable failure
running -> failed        retry ceiling reached; dead_letter = true

dispatching, queued, running -> pending
  stale owner recovery with a new claim_token and fencing_token
```

State meanings and owners are fixed:

| State | Meaning | Transition owner |
| --- | --- | --- |
| `pending` | Accepted work with no active delivery owner | Producer or reconciler |
| `dispatching` | An enqueue attempt is in progress or interrupted | Dispatcher |
| `queued` | A thin Huey envelope is durable and awaiting execution | Dispatcher/reconciler |
| `running` | A worker owns the current fenced attempt | Worker |
| `succeeded` | Handler completed successfully | Worker through CAS |
| `failed` | Terminal non-retryable failure | Worker through CAS |
| `dead_letter=true` | Automatic retry is permanently disabled | Store through CAS |

`attempt_count` counts execution attempts. `max_retries` bounds automatic
reexecution. `dispatch_attempt_count` bounds enqueue retries separately.
`next_attempt_at` prevents hot-looping while either kind of retry is backed
off. A dead-lettered item is not automatically reopened; explicit operator or
producer action is required.

Writer-busy deferral returns an item to `pending` without treating contention
as a handler failure. Forced reopen cannot replace a fresh owner. It may
replace an expired owner only through the fenced reclaim transition.

## CAS and fencing requirements

Every mutable owner operation is conditional on the complete current lease:

```text
WHERE work_item_id = ?
  AND state = ?
  AND claim_token = ?
  AND fencing_token = ?
```

The store must apply this compare-and-swap predicate to:

- execution heartbeat;
- success;
- failure;
- release;
- dequeue claim; and
- stale reclaim.

Reclaim atomically verifies staleness, increments `fencing_token`, rotates
`claim_token`, and returns the new owner credentials. A worker that lost
ownership may finish its local function, but its heartbeat, success, failure,
release, and reclaim calls must all fail without changing durable state.

The worker must not infer ownership from a Huey task ID alone. The task ID is a
delivery correlation value; the claim and fencing tokens are the authority.

## Thin Huey envelope

New Huey messages contain only delivery metadata:

```json
{
  "work_item_id": "stable-id",
  "task_name": "index_document",
  "priority": 100
}
```

The task's business arguments, payload, retry policy, and current fencing
credentials remain authoritative in SQLite. The worker loads the work item,
performs the fenced dequeue transition, and invokes the registered handler
from the stored payload.

Existing task names, priorities, logical arguments, public submission result
shapes, and batch outcome semantics remain compatible. Legacy full-payload
Huey messages may be decoded during migration and recovery, but they are not a
second steady-state format.

## Retry, backoff, and poison pills

Handler failures are classified before the next transition:

| Condition | Durable result |
| --- | --- |
| Retryable handler failure below `max_retries` | `pending` with `next_attempt_at` |
| Non-retryable handler failure | `failed` |
| Execution retry ceiling reached | `failed`, `dead_letter=true` |
| Transient enqueue failure below dispatch ceiling | `pending` with enqueue backoff |
| Dispatch ceiling reached | `failed`, `dead_letter=true` |

Backoff is bounded exponential delay with jitter or an equivalent monotonic
policy. Reconciliation must honor `next_attempt_at`; it must never spin on a
poison pill or an unavailable queue.

Dead-letter state is observable through queue status and remains durable across
restart. Automatic reclaim must not clear it.

## Resource-lease separation

`resource_leases` is a separate durable ledger for exclusive resources such as
the index writer and Google Drive scopes. Resource ownership has its own
heartbeat, expiry, and owner token. It must not be represented as a
`work_items` state, and reclaiming a work item must not release or replace a
resource lease.

Work items may wait because a resource lease is busy, but the resulting
deferral is a work-item transition and does not mutate resource ownership.

## Delivery guarantees

The contract provides:

- durable producer intent before delivery;
- recovery after a producer crash before enqueue;
- recovery after an enqueue succeeds but state persistence is interrupted;
- recovery after dequeue but before the worker claim;
- recovery after a worker claim or side effect is interrupted;
- stale-owner fencing; and
- at-least-once delivery with idempotent handlers.

The contract does not provide exactly-once execution. SQLite cannot commit a
work-item transition atomically with Huey's enqueue or dequeue operation, and
a handler side effect cannot share a transaction with both systems. Duplicate
delivery is therefore expected and must remain safe.

## Migration and rollback

Migration is additive and transactional:

1. Stop all daemons, workers, and producers.
2. Acquire the runtime/database maintenance lock.
3. Run one `BEGIN IMMEDIATE` migration that creates `work_items`,
   `resource_leases`, and schema metadata.
4. Copy `work_intents` while preserving IDs, payloads, attempts, failures, and
   terminal outcomes.
5. Deserialize pending, scheduled, and leased Huey payloads to associate queue
   deliveries with work items.
6. Preserve unmatched deliveries as explicitly quarantined synthetic items;
   never guess their logical identity.
7. Validate row counts, required payloads, fencing values, and
   `PRAGMA integrity_check` before commit.
8. Retain `work_intents` and `task_leases` as legacy tables for rollback and
   audit; production code must not use them as live authorities after cutover.

There is no mixed-version live operation. An old binary must not run against a
database after the new schema version is committed. Rollback requires stopping
all processes and restoring the queue database backup; an in-place downgrade
is unsupported.

## Queue-status compatibility

The queue-status JSON contract retains these keys exactly:

```text
pending_count
scheduled_count
running_count
failed_count
historical_failure_count
worker_running
backpressure_limit
backpressure_utilization
task_counts
recent_failures
pending_tasks
scheduled_tasks
```

`running_count` and failure counts may be computed from `work_items`, but their
names and value meanings remain compatible. Dead-lettered failures are
included in existing failure counts. New diagnostic fields may be additive;
existing keys must not be renamed, removed, or repurposed.

## Ownership boundaries

- `WorkItemStore` owns schema, state transitions, CAS, fencing, retry fields,
  and authoritative payload reads/writes.
- The dispatcher owns producer dedupe, backpressure, thin-envelope creation,
  and enqueue orchestration.
- The worker owns dequeue claims, heartbeats, handler invocation, and outcome
  reporting through the store port.
- The reconciler owns stale scans, backoff eligibility, re-enqueue, and
  dead-letter enforcement.
- Existing task handlers own indexing side effects and idempotency; this
  contract does not extract or redesign task operations.
- The resource-lease adapter owns writer and Drive scope exclusivity.
- Queue-status code owns only a read-only projection of durable state.

## Required verification

The implementation must test:

- preemption of a worker followed by rejected stale heartbeat, success,
  failure, release, and reclaim calls;
- duplicate delivery and idempotent handler behavior;
- the dequeue crash gap before work-item claim;
- enqueue failure backoff and poison-pill dead lettering;
- isolation between resource leases and work-item ownership;
- transactional migration rollback and migration integrity; and
- exact queue-status JSON key parity.

Workpackage 3 task-operation extraction is outside this contract. Its handler
boundaries, operation implementations, and payload semantics are inputs to
the delivery system, not part of this documentation or migration.
