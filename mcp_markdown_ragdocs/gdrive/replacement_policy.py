"""Google Drive replacement policy for canonical record indexing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from searchkernel.api import Record, RecordIngestor

from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.replacement import (
    GDriveReplacementEntry,
    GDriveReplacementJournal,
    canonical_gdrive_source_id,
    canonical_gdrive_source_key,
    group_gdrive_records,
    is_gdrive_tombstone,
)
from mcp_markdown_ragdocs.gdrive.state import (
    GDriveScopeIdentity,
    GDriveStateRepository,
)
from mcp_markdown_ragdocs.indexing.record_ports import RecordStorage, SourceMapStore


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
        state_repository: GDriveStateRepository | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._storage = storage
        self._source_records = source_records
        self._source_map_store = source_map_store
        self._journal = journal
        self._state_repository = state_repository

    async def replace(self, records: Sequence[Record]) -> None:
        plans = tuple(
            self._prepare_replacement(source_key, grouped)
            for source_key, grouped in group_gdrive_records(records).items()
        )
        await self._index_plans(plans)
        self._save_source_map()

    def recover(self) -> bool:
        recovered = False
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
            new_keys = tuple(
                key
                for key in entry.new_keys
                if self._storage.hydrate_record(key) is not None
            )
            self._delete_stale_keys(entry.source_key, entry.old_keys, new_keys)
            if new_keys:
                self._source_records[entry.source_key] = list(new_keys)
            else:
                self._source_records.pop(entry.source_key, None)
            self._journal.complete(entry.source_key)
            recovered = True
        if recovered:
            self._save_source_map()
        return recovered

    def _prepare_replacement(
        self,
        source_key: str,
        records: Sequence[Record],
    ) -> _GDriveReplacementPlan:
        representative = records[0]
        old_keys = self._record_keys(source_key)
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
    ) -> None:
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
        for plan in plans:
            self._delete_stale_keys(plan.source_key, plan.old_keys, plan.new_keys)
            if plan.new_keys:
                self._source_records[plan.source_key] = list(plan.new_keys)
            else:
                self._source_records.pop(plan.source_key, None)
            if plan.membership_records:
                self._record_memberships(plan.membership_records)
            if plan.tombstones:
                self._remove_memberships(plan.tombstones, plan.removed_scopes)
            self._journal.complete(plan.source_key, plan.identities)

    def _record_keys(self, source_key: str) -> tuple[str, ...]:
        candidates: set[str] = set()
        for keys in self._source_records.values():
            candidates.update(keys)
        candidates.update(
            record.storage_key
            for record in self._storage.iter_records()
            if record.source_kind == SOURCE_KIND
        )
        matching: list[str] = []
        for key in sorted(candidates):
            record = self._storage.hydrate_record(key)
            if record is None or record.source_kind != SOURCE_KIND:
                continue
            if canonical_gdrive_source_key(record) == source_key:
                matching.append(key)
        return tuple(matching)

    def _delete_stale_keys(
        self,
        source_key: str,
        old_keys: Sequence[str],
        new_keys: Sequence[str],
    ) -> None:
        stale_keys = sorted(set(old_keys) - set(new_keys))
        if stale_keys:
            self._storage.delete(stale_keys)
        new_key_set = set(new_keys)
        for doc_id, keys in tuple(self._source_records.items()):
            retained = [
                key
                for key in dict.fromkeys(keys)
                if key not in stale_keys
                and not (key in new_key_set and doc_id != source_key)
                and not (doc_id == source_key and key not in new_key_set)
            ]
            if retained:
                self._source_records[doc_id] = retained
            else:
                self._source_records.pop(doc_id, None)

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

    def _save_source_map(self) -> None:
        self._source_map_store.save(self._source_records)


__all__ = ["GDriveReplacementPolicy"]
