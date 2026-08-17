"""Application-owned capabilities for canonical record indexing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from searchkernel.api import LocalRecordKernel, Record, RecordIdentity


class RecordStorage(Protocol):
    """Read and mutate canonical records without exposing a kernel backend."""

    @property
    def db_manager(self) -> object: ...

    def hydrate_record(self, identity: RecordIdentity | str) -> Record | None: ...

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]: ...

    def iter_records(self) -> Iterable[Record]: ...

    def delete(self, storage_keys: Sequence[str]) -> None: ...


class SourceMapStore(Protocol):
    """Persist source-to-record membership with application-owned formatting."""

    def load(self) -> dict[str, list[str]]: ...

    def save(self, records: Mapping[str, Sequence[str]]) -> None: ...


class LocalRecordStorage:
    """Adapt the public local kernel stores to the record manager port."""

    def __init__(self, kernel: LocalRecordKernel) -> None:
        self._kernel = kernel

    @property
    def db_manager(self) -> object:
        return self._kernel.backend.db_manager

    def hydrate_record(self, identity: RecordIdentity | str) -> Record | None:
        if isinstance(identity, RecordIdentity):
            canonical = identity
        else:
            canonical = RecordIdentity.from_storage_key(identity)
        return self.hydrate_records((canonical,)).get(canonical.storage_key)

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]:
        return self._kernel.backend.hydrate_records(identities)

    def iter_records(self) -> Iterable[Record]:
        """Enumerate records through the installed local backend adapter."""
        rows = self._kernel.backend._record_rows()
        identities = [
            RecordIdentity.from_storage_key(str(row["storage_key"])) for row in rows
        ]
        hydrated = self.hydrate_records(identities)
        return tuple(
            record
            for identity in identities
            if (record := hydrated.get(identity.storage_key)) is not None
        )

    def delete(self, storage_keys: Sequence[str]) -> None:
        self._kernel.vector_store.delete(list(storage_keys))


class JsonSourceMapStore:
    """Store source-map membership using the legacy JSON representation."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, list[str]]:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(doc_id): [str(key) for key in keys if isinstance(key, str)]
            for doc_id, keys in value.items()
            if isinstance(keys, list)
        }

    def save(self, records: Mapping[str, Sequence[str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {doc_id: list(keys) for doc_id, keys in records.items()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._path)


__all__ = ["JsonSourceMapStore", "LocalRecordStorage", "RecordStorage", "SourceMapStore"]
