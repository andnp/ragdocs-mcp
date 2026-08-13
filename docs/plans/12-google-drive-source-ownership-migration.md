# Google Drive source ownership migration

**Status:** ragdocs is the current Google Drive owner; shadow comparison and
operator controls are published — 2026-08-12
**Design authority:** `../../devkit/specs/17-google-drive-ragdocs-ownership.md`
**Repositories:** `mcp-markdown-ragdocs`, `devkit`, `andnp-searchkernel`

## Goal

Make `mcp-markdown-ragdocs` the canonical, always-running owner of Google
Drive synchronization, extraction, indexing, and searchable-source behavior.
Make Devkit a consumer of that source through the existing searchkernel
federation contract.

## Current migration status

`mcp-markdown-ragdocs` is the owner of the Google Drive source and its
durable sync/index lifecycle. R48 added the deterministic, source-ID-keyed
shadow comparison artifact and write/index barrier. R49 published the
operator runbook covering credential separation, quiescence, retention,
health, and rollback. The remaining migration work is operational validation
and the controlled Devkit search cutover described below; historical numbered
specifications remain unchanged.

This plan is a migration of ownership, not a new Drive feature. Existing
Devkit behavior is the compatibility baseline, especially its durable cursor,
retry, tombstone, backfill, permission, watch, health, and bounded-extraction
semantics.

## Scope

### Move to ragdocs

- Google Drive source client and provider session boundary.
- OAuth configuration required by the indexing service.
- Scope inventory and change-token synchronization.
- Bounded download/export and MIME-aware extraction.
- Stable Drive record mapping and scope membership.
- Durable cursors, leases, retries, backfill, tombstones, and permission
  reconciliation.
- Push-channel renewal with polling fallback.
- Drive index composition, health, freshness, and source capabilities.
- Drive results through the existing `/v1/search` federation endpoint.
- Drive records in ragdocs' existing global record/index runtime, isolated
  logically by `source_kind="gdrive"`.

### Retain in Devkit

- Google Docs/Sheets note pull/push workflows.
- Explicit Drive operations exposed through Devkit's application surface.
- Devkit configuration for selecting and authenticating the remote ragdocs
  search source.
- Unified search, source selection, rank fusion, and degraded-source policy.

### Do not change in searchkernel

- `Record`, `ContentSource`, `SearchSource`, `SearchRequest`, and
  `SearchResponse` semantics unless a provider-neutral gap is proven.
- Generic chunking, embedding, index mutation, retrieval, and federation.

## Current implementation to migrate

The Devkit implementation is distributed across these areas:

- `src/devkit/integrations/gdrive/` — Drive client, extraction, source mapping,
  and provider error handling.
- `src/devkit/automations/gdrive_indexer.py` — inventory/change polling and
  durable checkpoint advancement.
- `src/devkit/automations/gdrive_backfill.py` — recovery of eligible failed
  content.
- `src/devkit/automations/gdrive_reconciler.py` — scope and permission
  reconciliation.
- `src/devkit/automations/gdrive_watch.py` — push-channel lifecycle.
- `src/devkit/cache/cursors.py` and `src/devkit/cache/gdrive_health.py` —
  durable state and telemetry.
- `src/devkit/application/bootstrap.py` and `src/devkit/application/search.py`
  — local Drive index construction and search registration.
- `src/devkit/application/config.py` — Drive indexing and embedding settings.
- `tests/test_gdrive_*.py` — current deterministic behavior and acceptance
  baseline.

The exact destination modules in ragdocs should follow its current
composition/lifecycle conventions rather than copying Devkit's package
layout.

## Refactor decision

```text
Refactor decision: larger
Concrete friction: Drive provider lifecycle and index ownership are currently
embedded in Devkit, while ragdocs already owns the always-running document
indexing runtime and the versioned remote search contract.
Proposed seam: ragdocs owns a Drive ContentSource plus a durable source
runtime; Devkit consumes it only through HttpSearchSource.
Expected payoff: one canonical Drive index, one sync lifecycle, no duplicated
embeddings, and a smaller Devkit search/bootstrap surface.
Scope: Drive indexing ownership only; Google Workspace workflows stay in
Devkit and searchkernel remains provider-neutral.
Stopping condition: ragdocs serves Drive through `/v1/search`, Devkit passes
remote-source acceptance, and all Devkit-local Drive indexing paths are
removed.
```

This is intentionally a larger refactor because the owner, durable state, and
runtime process change together. A partial move would create two authorities
and make correctness harder to establish.

## Workstreams

### W0 — freeze contracts and migration inventory

**Owner:** cross-repository
**Depends on:** design approval

Tasks:

1. Treat `specs/17-google-drive-ragdocs-ownership.md` as the new authority.
2. Enumerate the existing Devkit Drive test cases and map each to a ragdocs
   contract or integration test.
3. Record the current source identity, workspace identity, fingerprint,
   status, tombstone, and checkpoint formats as compatibility fixtures.
4. Record the resolved credential decision: ragdocs owns a separate configured
   credential file, initially seeded by an explicit copy of Devkit's current
   authorized-user token. Decide the exact credential path/permissions and
   old-index retention before implementation or production cutover. The
   caller decision is settled: Devkit uses a static single-user `devkit`
   identity with `search:read` and configured workspace claims. The source
   routing decision is settled: Devkit selects the ragdocs owner and sends
   `source_kinds=["gdrive"]`. The logical Drive corpus decision is settled:
   use the existing global record/index runtime with separate Drive
   sync-state namespaces.
5. Add a migration status section to the relevant current-state documents
   once implementation begins; do not rewrite historical specifications yet.

**Exit criteria:** the credential bootstrap rule is documented; the exact
credential path/permissions, persistence ownership, and rollback retention
details are resolved before W1 starts.

### W1 — implement the ragdocs Drive source adapter

**Owner:** `mcp-markdown-ragdocs`
**Depends on:** W0

Likely areas:

- `mcp_markdown_ragdocs/adapters/sources/gdrive.py` for the public
  `ContentSource`-level adapter.
- A Drive-specific integration package for client, extraction, retry, and
  typed provider models.
- `mcp_markdown_ragdocs/config.py` for source configuration.
- `mcp_markdown_ragdocs/app/composition.py` for source construction.

Tasks:

1. Define typed Drive file, change, scope, and provider-error models.
2. Implement the Drive client behind a narrow async protocol.
3. Add ragdocs-owned credential-path configuration and validate that the
   credential file is present, readable only by the service user, and outside
   the source tree. The initial operator copy from Devkit is documented but
   not automated as a runtime dependency.
4. Port bounded export/download and structure-preserving extraction.
5. Map files to stable `Record` values using the design document's identity
   and metadata contract.
6. Keep folders, shortcuts, unsupported files, and extraction failures as
   status-bearing metadata records without publishing partial content.
7. Make source construction explicit and testable without starting the daemon
   or mutating global process state.

Tests:

- Record identity and metadata mapping.
- Export/download MIME selection.
- extraction bounds and no-partial-content behavior.
- transient versus definitive provider-error classification.
- overlapping scope visibility converging on one record.

**Exit criteria:** the source can materialize deterministic records through
public searchkernel ports without any searchkernel change.

### W2 — port durable sync and recovery lifecycle

**Owner:** `mcp-markdown-ragdocs`
**Depends on:** W1

Tasks:

1. Add a versioned durable checkpoint store for inventory and change phases.
2. Persist the inventory start token before listing the first page.
3. Commit accepted index mutations before advancing the corresponding cursor.
4. Add per-scope/workspace leases so overlapping daemon tasks cannot duplicate
   work.
5. Port retry and backfill work items with durable status and bounded replay.
6. Port definitive-removal and confirmed-permission-loss tombstones.
7. Port scope membership reconciliation independently of record identity.
8. Port watch renewal and polling fallback using ragdocs lifecycle/task
   primitives.
9. Persist coverage, sync, quota, and source-health metrics in ragdocs-owned
   state.

Tests:

- interrupted inventory page resumes from the durable checkpoint;
- index mutation committed before cursor advancement;
- duplicate scope visibility does not duplicate embeddings;
- retryable 429/5xx/auth failures are recoverable;
- token invalidation triggers bounded full resynchronization;
- permission loss creates only the appropriate tombstones;
- extractor/chunker version changes reprocess affected records;
- lease expiry permits recovery without concurrent duplicate ownership.

**Exit criteria:** a restartable ragdocs sync has the same or stronger
correctness guarantees as the current Devkit sync.

### W3 — register the Drive index and expose the source

**Owner:** `mcp-markdown-ragdocs`
**Depends on:** W2

Tasks:

1. Register the Drive logical source in the canonical ragdocs composition
   root.
2. Route records through the existing global `RecordIndexManager` and searchkernel
   ingestion boundary.
3. Keep the Drive corpus independently filterable by `source_kind="gdrive"`.
4. Expose Drive in `/v1/search/capabilities` and `/v1/search`.
5. Include source identity, URI, lifecycle status, freshness/index epoch, and
   bounded citation-safe text in search responses.
6. Apply configured caller/workspace visibility before returning hits.
7. Expose enough health/freshness state for Devkit to distinguish empty,
   stale, unavailable, and healthy results.

Tests:

- composition with Drive enabled and disabled;
- source-filtered search and joint local search;
- capability discovery advertises Drive correctly;
- health and freshness state transitions;
- unauthorized or out-of-scope records are excluded;
- endpoint contract tests against the existing `HttpSearchSource` client.

**Exit criteria:** a running ragdocs instance can answer a Drive-only v1
search with stable provenance and no Devkit dependency.

### W4 — run shadow synchronization and compare results

**Owner:** cross-repository
**Depends on:** W3

Tasks:

1. Run ragdocs sync against the configured Drive scopes while Devkit's index
   remains available but is no longer the canonical result source.
2. Compare record counts, stable IDs, status/tombstone outcomes, freshness,
   and representative search results.
3. Investigate every mismatch using source IDs and fingerprints, not titles or
   path-like names.
4. Run restart, token invalidation, transient failure, permission-loss, and
   extractor-upgrade scenarios in a disposable corpus.
5. Perform one live authenticated Drive acceptance run.

**Exit criteria:** differences are either zero or documented as intentional
metadata/ranking differences; no unexplained identity, ACL, deletion, or
resume mismatch remains.

### W5 — cut Devkit over to remote ragdocs search

**Owner:** `devkit`
**Depends on:** W3, W4

Tasks:

1. Configure Devkit's existing ragdocs `HttpSearchSource` with the Drive
   source identity and production endpoint.
2. Preserve Devkit's `--source gdrive` and joint-search behavior at the user
   surface.
3. Pass caller/workspace scope and request correlation data through the v1
   federation request.
4. Verify unavailable/stale ragdocs behavior becomes an explicit partial or
   degraded federation response.
5. Stop scheduling Devkit's Drive indexer during the shadow/cutover window.
6. Keep the Devkit Drive client available for non-indexing operations.

Tests:

- remote Drive-only search;
- joint Devkit search with a remote ragdocs source;
- source-kind routing and provenance preservation;
- timeout/unavailable/stale-source degradation;
- non-indexing Docs/Sheets and Drive operation regressions.

**Exit criteria:** Devkit search has one canonical Drive source: ragdocs.

### W6 — remove the obsolete Devkit index owner

**Owner:** `devkit`
**Depends on:** W5 plus rollback approval

Tasks:

1. Remove Drive indexer, backfill, permission reconciler, and watch lifecycle
   registrations from the Devkit automation catalog and daemon composition.
2. Remove local Drive search composition and Drive embedding-provider wiring.
3. Remove Drive indexing-only configuration and local state paths.
4. Retain the Drive client, models, and operation handlers required by
   user-facing non-indexing workflows.
5. Remove or migrate Devkit Drive sync tables only after the retention window
   and state backup are verified.
6. Update specs 11, 14, README indexing instructions, and operational docs to
   describe ragdocs as the owner.

Tests:

- automation catalog no longer schedules Drive indexing;
- bootstrap never creates a local Drive index;
- Devkit Drive operations remain available;
- unified search uses only the remote Drive source;
- static search confirms no production Devkit Drive index path remains.

**Exit criteria:** Devkit cannot accidentally become a second Drive index
owner, and the retained Google Workspace workflows remain green.

## Contract and test matrix

| Behavior | ragdocs test boundary | Devkit test boundary |
| --- | --- | --- |
| Record identity | source adapter unit tests | remote hit provenance |
| Extraction bounds | extractor unit/integration tests | n/a |
| Cursor ordering | sync integration tests | n/a |
| Retry/tombstone policy | sync recovery tests | degraded-source mapping |
| Scope deduplication | sync/index integration tests | duplicate-hit guard |
| ACL/workspace filtering | source endpoint acceptance | request context propagation |
| Index freshness | source health/endpoint tests | stale-source diagnostics |
| Search contract | `/v1/search` contract tests | `HttpSearchSource` integration |
| Joint search | ragdocs local source search | Devkit federation tests |
| Runtime restart | daemon/worker acceptance | remote availability handling |
| Google Workspace workflows | n/a | Docs/Sheets and Drive operation tests |

## Data and deployment rules

- Ragdocs owns the Drive index database, embedding cache, cursor store, retry
  work, lease state, and health state after W6.
- Drive records share ragdocs' global record/index runtime with Markdown and
  Git records, but Drive sync checkpoints, retries, leases, permissions, and
  health use separate source-specific namespaces.
- Devkit does not mount or read those databases.
- OAuth tokens are owned by the process that performs Drive indexing. Ragdocs
  reads its own credential file, which may initially be seeded by manually
  copying Devkit's current authorized-user token. The applications never
  share a credential path or refresh the same file. Re-provisioning after
  revocation or rotation is an explicit operator action.
- Credential files live outside source trees and artifacts, use owner-only
  permissions, and are never committed to either repository.
- The v1 endpoint must be authenticated and must carry caller/workspace scope.
- The old Devkit index is retained until shadow comparison, live acceptance,
  cutover, and rollback verification are complete.

## Rollback and failure handling

Before W6, set Devkit's Drive source back to its local index and re-enable its
indexer if ragdocs cannot meet the acceptance gates. Do not run both sources in
the same user-visible federation result set; shadow comparisons must use
isolated source selection or diagnostics.

If ragdocs is unavailable after W6, Devkit reports a degraded Drive source and
continues serving other sources. Recovery is a ragdocs service restoration or
an explicit release rollback, not a new hidden local index path in Devkit.

## Verification commands

The exact commands should use each repository's lockfile and CI entry points.
At minimum, before cutover:

```text
mcp-markdown-ragdocs: focused Drive/source tests
mcp-markdown-ragdocs: federation endpoint contract tests
mcp-markdown-ragdocs: full lint, type, import-boundary, and test checks
devkit: focused remote federation and retained Google operation tests
devkit: full Ruff, Pyright, Pyrefly, and test checks
live: authenticated Drive sync, restart, search, permission, and deletion run
```

For Python projects, use the repository's existing environment and lockfile.
Do not replace the required repository checks with spot checks.

## Deferred work

- Multi-tenant Drive credential management beyond the current single-user
  deployment.
- Shared embedding/model rollout across federated sources.
- A provider-neutral source administration API beyond existing health and
  capability endpoints.
- Moving other Devkit pull sources such as Gmail or Obsidian.
- Physically separating Drive into another ragdocs index or corpus runtime.
