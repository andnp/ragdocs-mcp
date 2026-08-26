"""Google Drive replacement policy for canonical record indexing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from searchkernel.api import Record, RecordIdentity, RecordIngestor

from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementEntry,
    GDriveReplacementJournal,
    canonical_gdrive_source_id,
    canonical_gdrive_source_key,
    group_gdrive_records,
    is_gdrive_tombstone,
)
from mcp_markdown_ragdocs.gdrive.domain import GDriveScopeIdentity
from mcp_markdown_ragdocs.gdrive.port import GDriveStatePort
from mcp_markdown_ragdocs.indexing.record_ports import (
    RecordStorage,
    SourceMapStore,
    SqliteSourceMapStore,
)

_SourceMapDelta = dict[str, "list[str] | None"]


@dataclass(frozen=True, slots=True)
class _GDriveReplacementPlan:
    source_key: str
    old_keys: tuple[str, ...]
    new_keys: tuple[str, ...]
    identities: tuple[GDriveScopeIdentity, ...]
    entry: GDriveReplacementEntry
    records_to_index: tuple[Record, ...]
    membership_records: tuple[Record, ...]
    tombstones: tuple[Record, ...]
    removed_scopes: tuple[str, ...]


class GDriveReplacementPolicy:
    """Apply Drive replacements through provider-neutral record capabilities."""

    def __init__(
        self,
        ingestor: RecordIngestor,
        storage: RecordStorage,
        source_records: dict[str, list[str]],
        source_map_store: SourceMapStore,
        journal: GDriveReplacementJournal,
        state_repository: GDriveStatePort | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._storage = storage
        self._source_records = source_records
        self._source_map_store = source_map_store
        self._journal = journal
        self._state_repository = state_repository

    async def replace(self, records: Sequence[Record]) -> None:
        grouped_by_source = group_gdrive_records(records)
        existing_keys = self._record_keys_by_source()
        plans = tuple(
            self._prepare_replacement(
                source_key, grouped, existing_keys.get(source_key, ())
            )
            for source_key, grouped in grouped_by_source.items()
        )
        delta = await self._index_plans(plans)
        self._apply_source_map_delta(delta)

    def recover(self) -> bool:
        recovered = False
        reverse_index: dict[str, set[str]] | None = None
        delta: _SourceMapDelta = {}
        for entry in self._journal.load():
            if entry.phase == "prepared":
                partial_keys = tuple(
                    key
                    for key in entry.new_keys
                    if key not in entry.old_keys
                    and self._storage.hydrate_record(key) is not None
                )
                if partial_keys:
                    self._storage.delete(partial_keys)
                self._journal.complete(entry.source_key)
                recovered = True
                continue
            if reverse_index is None:
                reverse_index = self._build_reverse_index()
            new_keys = tuple(
                key
                for key in entry.new_keys
                if self._storage.hydrate_record(key) is not None
            )
            self._delete_stale_keys(
                entry.source_key, entry.old_keys, new_keys, reverse_index, delta
            )
            if new_keys:
                self._set_source_keys(entry.source_key, list(new_keys), reverse_index, delta)
            else:
                self._pop_source_keys(entry.source_key, reverse_index, delta)
            self._journal.complete(entry.source_key)
            recovered = True
        if recovered:
            self._apply_source_map_delta(delta)
        return recovered

    def _prepare_replacement(
        self,
        source_key: str,
        records: Sequence[Record],
        old_keys: tuple[str, ...],
    ) -> _GDriveReplacementPlan:
        representative = records[0]
        tombstones = tuple(record for record in records if is_gdrive_tombstone(record))
        if not tombstones:
            indexed_records = self._with_memberships(records)
            new_keys = tuple(
                dict.fromkeys(record.storage_key for record in indexed_records)
            )
            identities = self._scope_identities(representative, indexed_records)
            entry = self._journal.prepare(source_key, old_keys, new_keys, identities)
            return _GDriveReplacementPlan(
                source_key,
                old_keys,
                new_keys,
                identities,
                entry,
                indexed_records,
                indexed_records,
                (),
                (),
            )

        existing = tuple(
            record
            for key in old_keys
            if (record := self._storage.hydrate_record(key)) is not None
            and not is_gdrive_tombstone(record)
        )
        known_scopes = set(self._scopes_from_records(existing))
        removed_scopes = set(self._scopes_from_records(tombstones))
        if self._state_repository is not None:
            workspace_id = next(
                (
                    record.workspace_id
                    for record in (*records, *existing)
                    if record.workspace_id
                ),
                None,
            )
            source_id = canonical_gdrive_source_id(records[0])
            if workspace_id:
                durable_scopes = self._state_repository.memberships_for_source(
                    SOURCE_KIND,
                    workspace_id,
                    source_id,
                )
                known_scopes.update(durable_scopes)
        if not removed_scopes:
            removed_scopes = set(known_scopes)
        remaining_scopes = tuple(sorted(known_scopes - removed_scopes))
        replacement_records = (
            tuple(self._record_with_scopes(record, remaining_scopes) for record in existing)
            if remaining_scopes
            else ()
        )
        new_keys = tuple(
            dict.fromkeys(record.storage_key for record in replacement_records)
        )
        identities = self._scope_identities(records[0], (*records, *existing))
        entry = self._journal.prepare(source_key, old_keys, new_keys, identities)
        return _GDriveReplacementPlan(
            source_key,
            old_keys,
            new_keys,
            identities,
            entry,
            replacement_records,
            (),
            tombstones,
            tuple(sorted(removed_scopes)),
        )

    async def _index_plans(
        self,
        plans: Sequence[_GDriveReplacementPlan],
    ) -> _SourceMapDelta:
        records_to_index = tuple(
            record for plan in plans for record in plan.records_to_index
        )
        if records_to_index:
            receipt = await self._ingestor.index_records(records_to_index)
            if receipt.failed:
                raise RuntimeError(
                    "; ".join(item.error or "unknown error" for item in receipt.failures)
                )

        for plan in plans:
            self._journal.mark_indexed(plan.entry, plan.identities)
        reverse_index = self._build_reverse_index()
        delta: _SourceMapDelta = {}
        for plan in plans:
            self._delete_stale_keys(
                plan.source_key, plan.old_keys, plan.new_keys, reverse_index, delta
            )
            if plan.new_keys:
                self._set_source_keys(plan.source_key, list(plan.new_keys), reverse_index, delta)
            else:
                self._pop_source_keys(plan.source_key, reverse_index, delta)
            if plan.membership_records:
                self._record_memberships(plan.membership_records)
            if plan.tombstones:
                self._remove_memberships(plan.tombstones, plan.removed_scopes)
            self._journal.complete(plan.source_key, plan.identities)
        return delta

    _RECORD_KEYS_BATCH_SIZE = 500

    def _record_keys_by_source(self) -> Mapping[str, tuple[str, ...]]:
        """Build the full source_key -> existing storage keys mapping.

        `_source_records` already associates every tracked Drive doc_id with
        its own keys (see `_set_source_keys`/`_index_plans`), so for any key
        still tracked there the source key is known for free. Only keys that
        the index reports as Drive records (`iter_identities`, which needs no
        hydration) but that are absent from that tracking -- genuine drift,
        e.g. after an interrupted run -- require hydrating a record to learn
        its canonical source key.
        """
        drive_keys = {
            identity.storage_key
            for identity in self._storage.iter_identities(source_kind=SOURCE_KIND)
        }
        known_source_of: dict[str, str] = {}
        for doc_id, keys in self._source_records.items():
            for key in keys:
                if key in drive_keys:
                    known_source_of[key] = doc_id

        grouped: dict[str, list[str]] = {}
        for key in drive_keys:
            source_key = known_source_of.get(key)
            if source_key is not None:
                grouped.setdefault(source_key, []).append(key)

        drift_keys = sorted(drive_keys.difference(known_source_of))
        for start in range(0, len(drift_keys), self._RECORD_KEYS_BATCH_SIZE):
            batch = drift_keys[start : start + self._RECORD_KEYS_BATCH_SIZE]
            hydrated = self._storage.hydrate_records(
                tuple(RecordIdentity.from_storage_key(key) for key in batch)
            )
            for key in batch:
                record = hydrated.get(key)
                if record is None or record.source_kind != SOURCE_KIND:
                    continue
                grouped.setdefault(canonical_gdrive_source_key(record), []).append(key)
        return {source_key: tuple(sorted(keys)) for source_key, keys in grouped.items()}

    def _build_reverse_index(self) -> dict[str, set[str]]:
        """Map each tracked storage key to the doc_id(s) currently holding it.

        Built once per `replace()`/`recover()` call so `_delete_stale_keys`
        can find the handful of doc_ids a batch might affect without walking
        the entire `_source_records` map (which holds every Drive source and
        every note doc_id) for every source key in the batch.
        """
        reverse_index: dict[str, set[str]] = {}
        for doc_id, keys in self._source_records.items():
            for key in keys:
                reverse_index.setdefault(key, set()).add(doc_id)
        return reverse_index

    def _unindex_source_keys(self, doc_id: str, reverse_index: dict[str, set[str]]) -> None:
        for key in self._source_records.get(doc_id, ()):
            doc_ids = reverse_index.get(key)
            if doc_ids is None:
                continue
            doc_ids.discard(doc_id)
            if not doc_ids:
                del reverse_index[key]

    def _set_source_keys(
        self,
        doc_id: str,
        keys: list[str],
        reverse_index: dict[str, set[str]],
        delta: _SourceMapDelta,
    ) -> None:
        self._unindex_source_keys(doc_id, reverse_index)
        self._source_records[doc_id] = keys
        for key in keys:
            reverse_index.setdefault(key, set()).add(doc_id)
        delta[doc_id] = keys

    def _pop_source_keys(
        self,
        doc_id: str,
        reverse_index: dict[str, set[str]],
        delta: _SourceMapDelta,
    ) -> None:
        if doc_id not in self._source_records:
            return
        self._unindex_source_keys(doc_id, reverse_index)
        self._source_records.pop(doc_id, None)
        delta[doc_id] = None

    def _delete_stale_keys(
        self,
        source_key: str,
        old_keys: Sequence[str],
        new_keys: Sequence[str],
        reverse_index: dict[str, set[str]],
        delta: _SourceMapDelta,
    ) -> None:
        """Drop keys `source_key` no longer owns, from storage and bookkeeping.

        Only doc_ids whose keys intersect the stale or newly-claimed keys can
        possibly change, so `reverse_index` (kept consistent with
        `_source_records` across the whole batch by the caller) narrows the
        doc_ids visited to that set plus `source_key` itself, instead of
        every doc_id ever tracked.
        """
        stale_keys = set(old_keys) - set(new_keys)
        if stale_keys:
            self._storage.delete(sorted(stale_keys))
        new_key_set = set(new_keys)
        affected_doc_ids = {source_key}
        for key in stale_keys:
            affected_doc_ids.update(reverse_index.get(key, ()))
        for key in new_key_set:
            affected_doc_ids.update(reverse_index.get(key, ()))
        for doc_id in affected_doc_ids:
            keys = self._source_records.get(doc_id)
            if keys is None:
                continue
            retained = [
                key
                for key in dict.fromkeys(keys)
                if key not in stale_keys
                and not (key in new_key_set and doc_id != source_key)
                and not (doc_id == source_key and key not in new_key_set)
            ]
            if retained:
                self._set_source_keys(doc_id, retained, reverse_index, delta)
            else:
                self._pop_source_keys(doc_id, reverse_index, delta)

    def _with_memberships(self, records: Sequence[Record]) -> tuple[Record, ...]:
        scopes = set(self._scopes_from_records(records))
        source_id = canonical_gdrive_source_id(records[0])
        workspace_id = records[0].workspace_id
        if self._state_repository is not None and workspace_id:
            scopes.update(
                self._state_repository.memberships_for_source(
                    SOURCE_KIND,
                    workspace_id,
                    source_id,
                )
            )
        return tuple(self._record_with_scopes(record, tuple(sorted(scopes))) for record in records)

    def _record_with_scopes(self, record: Record, scopes: Sequence[str]) -> Record:
        if not scopes:
            return record
        metadata = {**record.metadata, "scope_memberships": list(scopes)}
        return replace(record, metadata=metadata)

    def _scopes_from_records(self, records: Sequence[Record]) -> tuple[str, ...]:
        scopes: set[str] = set()
        for record in records:
            raw_scopes = record.metadata.get("scope_memberships")
            if isinstance(raw_scopes, (list, tuple, set)):
                scopes.update(
                    scope
                    for scope in raw_scopes
                    if isinstance(scope, str) and scope
                )
        return tuple(sorted(scopes))

    def _scope_identities(
        self,
        representative: Record,
        records: Sequence[Record],
    ) -> tuple[GDriveScopeIdentity, ...]:
        if not representative.workspace_id:
            return ()
        return tuple(
            GDriveScopeIdentity(
                SOURCE_KIND,
                representative.workspace_id,
                scope,
            )
            for scope in self._scopes_from_records(records)
        )

    def _record_memberships(self, records: Sequence[Record]) -> None:
        repository = self._state_repository
        if repository is None:
            return
        for record in records:
            if not record.workspace_id:
                continue
            source_id = canonical_gdrive_source_id(record)
            for scope in self._scopes_from_records((record,)):
                repository.add_membership(
                    GDriveScopeIdentity(SOURCE_KIND, record.workspace_id, scope),
                    source_id,
                )

    def _remove_memberships(
        self,
        records: Sequence[Record],
        scopes: Sequence[str],
    ) -> None:
        repository = self._state_repository
        if repository is None:
            return
        for record in records:
            if not record.workspace_id:
                continue
            source_id = canonical_gdrive_source_id(record)
            for scope in scopes:
                repository.remove_membership(
                    GDriveScopeIdentity(SOURCE_KIND, record.workspace_id, scope),
                    source_id,
                )

    def _apply_source_map_delta(self, delta: _SourceMapDelta) -> None:
        """Persist only the doc_ids this call actually touched.

        `_source_map_store` is declared as the narrower `SourceMapStore`
        Protocol (no `apply_delta`), but the object bound to it in production
        (`SqliteSourceMapStore`, via `build_gdrive_integration`) does expose
        it. Dispatch on that concrete type the same way
        `RecordIndexManager._save_source_map_delta` already does
        (record_manager.py:409), and fall back to a full rewrite only for a
        store that genuinely lacks the incremental API.
        """
        if not delta:
            return
        if isinstance(self._source_map_store, SqliteSourceMapStore):
            upserts = {doc_id: keys for doc_id, keys in delta.items() if keys is not None}
            removals = [doc_id for doc_id, keys in delta.items() if keys is None]
            self._source_map_store.apply_delta(upserts, removals)
            return
        self._source_map_store.save(self._source_records)


__all__ = ["GDriveReplacementPolicy"]
