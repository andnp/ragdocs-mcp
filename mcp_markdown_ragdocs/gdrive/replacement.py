"""Idempotent replacement state for Google Drive records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from searchkernel.api import Record, RecordIdentity, RecordStatus

from mcp_markdown_ragdocs.gdrive.json_record_store import JsonEnvelopeStore
from mcp_markdown_ragdocs.gdrive.records import SOURCE_KIND
from mcp_markdown_ragdocs.gdrive.domain import (
    GDriveScopeIdentity,
    GDriveSyncStatus,
)
from mcp_markdown_ragdocs.gdrive.port import GDriveStatePort

REPLACEMENT_SCHEMA_VERSION = 1
REPLACEMENT_JOURNAL_FILENAME = "gdrive-replacements.json"
_Phase = Literal["prepared", "indexed"]


@dataclass(frozen=True, slots=True)
class GDriveReplacementEntry:
    """Durable intent for one canonical Drive source replacement."""

    source_key: str
    old_keys: tuple[str, ...]
    new_keys: tuple[str, ...]
    phase: _Phase = "prepared"


def canonical_gdrive_source_id(record: Record) -> str:
    """Return the stable Drive file identity, including chunked records."""

    if record.source_kind != SOURCE_KIND:
        raise ValueError("record is not a Google Drive record")
    for name in ("gdrive_source_id", "source_id", "doc_id"):
        value = record.metadata.get(name)
        if isinstance(value, str) and value:
            return value
    if not record.source_id:
        raise ValueError("Google Drive record source_id must be non-empty")
    return record.source_id


def canonical_gdrive_source_key(record: Record) -> str:
    """Return the workspace-safe source-map key for a Drive file."""

    return RecordIdentity(
        record.workspace_id,
        SOURCE_KIND,
        canonical_gdrive_source_id(record),
    ).storage_key


def is_gdrive_tombstone(record: Record) -> bool:
    """Identify provider-confirmed Drive removal records."""

    return (
        record.source_kind == SOURCE_KIND
        and (
            record.metadata.get("deleted") is True
            or record.metadata.get("extraction_status") == "tombstone"
            or record.status is RecordStatus.ARCHIVED
        )
    )


def group_gdrive_records(records: Iterable[Record]) -> dict[str, tuple[Record, ...]]:
    """Group Drive records by canonical source and deduplicate storage keys."""

    grouped: dict[str, dict[str, Record]] = {}
    for record in records:
        source_key = canonical_gdrive_source_key(record)
        candidates = grouped.setdefault(source_key, {})
        previous = candidates.get(record.storage_key)
        if previous is None or _record_order(record) < _record_order(previous):
            candidates[record.storage_key] = record
    return {
        source_key: tuple(sorted(records_by_key.values(), key=lambda item: item.storage_key))
        for source_key, records_by_key in grouped.items()
    }


class GDriveReplacementJournal:
    """Persist replacement phases and optionally mirror scope health in Drive state."""

    def __init__(
        self,
        path: Path,
        state_repository: GDriveStatePort | None = None,
    ) -> None:
        self.path = Path(path)
        self.state_repository = state_repository
        self._envelope = JsonEnvelopeStore(self.path, REPLACEMENT_SCHEMA_VERSION, "entries")
        # Authoritative in-process state, lazily hydrated from disk on first
        # use. Avoids re-reading and re-parsing the whole journal file on
        # every write; only the initial load (or a fresh process) pays that
        # cost. `None` means "not loaded yet", distinct from an empty journal.
        self._entries: dict[str, GDriveReplacementEntry] | None = None

    def prepare(
        self,
        source_key: str,
        old_keys: Sequence[str],
        new_keys: Sequence[str],
        state_identities: Sequence[GDriveScopeIdentity] = (),
    ) -> GDriveReplacementEntry:
        """Record intent before indexing can make a partial durable write."""

        entry = GDriveReplacementEntry(
            source_key,
            tuple(dict.fromkeys(old_keys)),
            tuple(dict.fromkeys(new_keys)),
        )
        self._write(entry)
        self._save_status(state_identities, "replacement-pending")
        return entry

    def mark_indexed(
        self,
        entry: GDriveReplacementEntry,
        state_identities: Sequence[GDriveScopeIdentity] = (),
    ) -> GDriveReplacementEntry:
        """Record that all replacement records reached the index writer."""

        indexed = GDriveReplacementEntry(
            entry.source_key,
            entry.old_keys,
            entry.new_keys,
            "indexed",
        )
        self._write(indexed)
        self._save_status(state_identities, "replacement-indexed")
        return indexed

    def complete(
        self,
        source_key: str,
        state_identities: Sequence[GDriveScopeIdentity] = (),
    ) -> None:
        """Remove the intent only after index and source-map cleanup finish."""

        entries = self._entries_by_key()
        entries.pop(source_key, None)
        self._flush()
        self._save_status(state_identities, "healthy")

    def load(self) -> tuple[GDriveReplacementEntry, ...]:
        """Load valid journal entries in stable source-key order.

        Always re-reads the file rather than the in-process cache: callers
        use this to observe state a *different* journal instance recovered
        or wrote (e.g. a fresh instance built after a simulated crash), and
        it is not on the per-entry write hot path, so refreshing here is
        cheap relative to the savings in `_write`/`_flush`.
        """

        self._entries = {entry.source_key: entry for entry in self._read_from_disk()}
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.source_key))

    def _entries_by_key(self) -> dict[str, GDriveReplacementEntry]:
        """Return the in-process entry cache, hydrating it from disk once."""

        if self._entries is None:
            self._entries = {entry.source_key: entry for entry in self._read_from_disk()}
        return self._entries

    def _read_from_disk(self) -> tuple[GDriveReplacementEntry, ...]:
        raw_entries = self._envelope.read(list)
        if raw_entries is None:
            return ()
        entries: list[GDriveReplacementEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            try:
                source_key = raw["source_key"]
                old_keys = raw["old_keys"]
                new_keys = raw["new_keys"]
                phase = raw["phase"]
                if (
                    not isinstance(source_key, str)
                    or not isinstance(old_keys, list)
                    or not isinstance(new_keys, list)
                    or phase not in {"prepared", "indexed"}
                    or not all(isinstance(key, str) for key in (*old_keys, *new_keys))
                ):
                    continue
                entries.append(
                    GDriveReplacementEntry(
                        source_key,
                        tuple(dict.fromkeys(old_keys)),
                        tuple(dict.fromkeys(new_keys)),
                        phase,
                    )
                )
            except (KeyError, TypeError):
                continue
        return tuple(entries)

    def _write(self, entry: GDriveReplacementEntry) -> None:
        entries = self._entries_by_key()
        entries[entry.source_key] = entry
        self._flush()

    def _flush(self) -> None:
        """Atomically persist the cached entries; still one fsync per call.

        This keeps the durability contract prepare()/mark_indexed()/complete()
        rely on (each phase transition is on disk before the next step
        proceeds) but serializes straight from the in-memory cache instead of
        re-reading and re-parsing the whole file first.
        """

        entries = self._entries_by_key()
        self._envelope.write(
            [
                {
                    "source_key": entry.source_key,
                    "old_keys": list(entry.old_keys),
                    "new_keys": list(entry.new_keys),
                    "phase": entry.phase,
                }
                for entry in sorted(entries.values(), key=lambda item: item.source_key)
            ]
        )

    def _save_status(
        self,
        identities: Sequence[GDriveScopeIdentity],
        status: str,
    ) -> None:
        if self.state_repository is None:
            return
        for identity in identities:
            self.state_repository.save_sync_status(GDriveSyncStatus(identity, status))


def _record_order(record: Record) -> tuple[str, str, str]:
    return (record.storage_key, record.updated_at.isoformat(), record.body)


__all__ = [
    "GDriveReplacementEntry",
    "GDriveReplacementJournal",
    "REPLACEMENT_JOURNAL_FILENAME",
    "REPLACEMENT_SCHEMA_VERSION",
    "canonical_gdrive_source_id",
    "canonical_gdrive_source_key",
    "group_gdrive_records",
    "is_gdrive_tombstone",
]
