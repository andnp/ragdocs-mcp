# Google Drive owner operator runbook

This runbook covers the migration to `mcp-markdown-ragdocs` as the canonical
Google Drive indexing and search owner. Devkit remains responsible for
user-facing Google Workspace operations, but it must not run a second Drive
index during shadowing or after cutover.

## Credential bootstrap

Create a dedicated ragdocs authorized-user credential outside both source
trees and artifacts. The service user must own the file and its permissions
must prevent group and other access. Ragdocs owns and refreshes this file;
Devkit must not mount it or share its refresh state.

The initial bootstrap is a one-time manual operation:

1. Stop or quiesce Drive indexing in Devkit.
2. Copy Devkit's current authorized-user token to the separately configured
   ragdocs credential path.
3. Set owner-only file permissions and verify the ragdocs service user can
   read the copy.
4. Start ragdocs and perform an authenticated inventory/sync check.
5. Record the credential path and owner in the deployment secret inventory;
   never commit the token or place it in a repository, backup artifact, or log.

The token copy is not a runtime dependency. If the token is revoked or
rotated, provision the ragdocs credential explicitly and restart the owning
service after validating it.

## Runtime ownership

After cutover, ragdocs owns Drive credentials, scopes, inventory and change
cursors, retries, leases, backfill state, tombstones, health state, records,
embeddings, and the searchable Drive index. Devkit selects the ragdocs source
through the authenticated search endpoint and may continue its non-indexing
Docs/Sheets and Drive workflows.

Keep the old Devkit index read-only during migration. Do not include both
indexes in the same user-visible result set.

## Shadow and quiescence procedure

1. Back up the Devkit Drive index and its cursor/recovery state.
2. Confirm the dedicated ragdocs credential, configured scopes, workspace
   identity, and service health.
3. Quiesce Devkit Drive indexing: stop its scheduler and watcher, drain or
   cancel pending Drive indexing work, and confirm no writer is active.
4. Run ragdocs inventory and change synchronization against the same scopes.
5. Compare the deterministic artifact at
   `gdrive-shadow-comparison.json`, keyed by stable Drive `source_id`.
   Ranking, chunk identity, and index-epoch differences are normalized as
   allowed drift; missing sources and explicitly disallowed differences are
   mismatches.
6. Investigate every mismatch by `source_id`, fingerprint, status, and
   freshness. Titles and paths are not identity keys.
7. Repeat after restart and after representative update, deletion,
   permission-loss, transient-failure, and extractor-version scenarios.
8. Cut Devkit search over only after the live acceptance checks pass. Keep
   Devkit's old index unavailable to user-visible federation during the
   comparison window.

Quiescence is complete only when the Devkit Drive writer is stopped, its
pending queue is empty or explicitly recorded, and the ragdocs sync owner is
the only process advancing Drive state.

## Backup and retention

Before shadowing, retain a restorable backup of the old Devkit index,
credentials metadata (never token contents in plaintext), cursors, retry
work, and health/checkpoint state. Store it outside both source trees with
restricted access. Retain the old index and migration backup for at least 30
days after cutover and after rollback verification. Do not delete the backup
until the retention owner confirms the window has elapsed and the ragdocs
index has a verified restore path.

## Health states

The Drive source reports one of these states:

- `healthy`: available, ACL-complete, populated, and within the freshness
  threshold;
- `empty`: available but no indexed or remote records are present;
- `stale`: the last successful sync is older than the configured threshold;
- `acl-incomplete`: one or more configured scopes have incomplete permission
  coverage;
- `unavailable`: credentials, provider access, or the ragdocs source is not
  available.

Treat `empty`, `stale`, `acl-incomplete`, and `unavailable` as migration gates
until explained. A degraded source must be visible to callers; do not silently
fall back to a second local Drive index.

## Rollback

Before the old Devkit index is retired, rollback is:

1. Stop ragdocs Drive indexing and prevent it from advancing cursors.
2. Restore the last verified Devkit index and recovery state if needed.
3. Re-enable exactly one Devkit Drive indexer and watcher.
4. Route user-visible Drive search back to Devkit and verify health,
   freshness, permissions, updates, and deletions.
5. Record the reason, timestamps, restored backup, and any missed changes.

After the retention window, rollback requires an explicit release decision and
a verified ragdocs restore or re-bootstrap. Never run both owners in the same
result set; dual writers can produce duplicate embeddings, cursor races, and
conflicting deletion state.
