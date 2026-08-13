"""Deterministic shadow comparison artifacts for Google Drive search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from searchkernel.api import Record, atomic_write_json

SHADOW_SCHEMA_VERSION = 1
SHADOW_ARTIFACT_FILENAME = "gdrive-shadow-comparison.json"


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    """The stable fields needed to compare one Drive search hit."""

    source_id: str
    rank: int | None = None
    chunk_id: str | None = None
    index_epoch: int | str | None = None
    source_kind: str | None = None
    workspace_id: str | None = None
    status: str | None = None
    acl_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("shadow source_id is required")
        if self.rank is not None and self.rank < 1:
            raise ValueError("shadow rank must be positive or null")


@dataclass(frozen=True, slots=True)
class ShadowComparisonPolicy:
    """Differences that are expected during a non-authoritative shadow run."""

    allow_ranking: bool = True
    allow_chunk: bool = True
    allow_index_epoch: bool = True


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """A stable, source-keyed comparison result."""

    query: str
    entries: dict[str, dict[str, object]]

    @property
    def mismatches(self) -> int:
        return sum(1 for entry in self.entries.values() if entry["unexpected"])

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "query": self.query,
            "mismatches": self.mismatches,
            "entries": self.entries,
        }


class ShadowArtifactStore:
    """Atomically persist the latest deterministic comparison artifact."""

    def __init__(self, index_root: Path) -> None:
        self.path = Path(index_root) / SHADOW_ARTIFACT_FILENAME

    def save(self, comparison: ShadowComparison) -> None:
        atomic_write_json(self.path, comparison.to_payload())


class ShadowRecordWriter(Protocol):
    def index_records(self, records: Sequence[Record]) -> bool: ...

    def persist(self) -> None: ...


def persist_index_then_shadow_artifact(
    writer: ShadowRecordWriter,
    records: Sequence[Record],
    comparison: ShadowComparison,
    artifact_store: ShadowArtifactStore,
) -> None:
    """Publish the artifact only after the corresponding index is durable."""
    if not writer.index_records(records):
        raise RuntimeError("Google Drive shadow index write failed")
    writer.persist()
    artifact_store.save(comparison)


def compare_shadow_results(
    query: str,
    owner: Sequence[ShadowObservation],
    shadow: Sequence[ShadowObservation],
    *,
    policy: ShadowComparisonPolicy = ShadowComparisonPolicy(),
) -> ShadowComparison:
    """Compare results while normalizing explicitly allowed implementation drift."""
    owner_by_id = _by_source_id(owner)
    shadow_by_id = _by_source_id(shadow)
    entries: dict[str, dict[str, object]] = {}
    for source_id in sorted(owner_by_id.keys() | shadow_by_id.keys()):
        left = owner_by_id.get(source_id)
        right = shadow_by_id.get(source_id)
        differences: set[str] = set()
        unexpected: set[str] = set()
        if left is None or right is None:
            unexpected.add("missing_source")
        else:
            if left.rank != right.rank:
                differences.add("ranking")
            if left.chunk_id != right.chunk_id:
                differences.add("chunk")
            if left.index_epoch != right.index_epoch:
                differences.add("index_epoch")
            if (left.source_kind, left.workspace_id) != (
                right.source_kind,
                right.workspace_id,
            ):
                differences.add("identity")
            if left.status != right.status:
                differences.add("status")
            if left.acl_fingerprint != right.acl_fingerprint:
                differences.add("acl")
            allowed = {
                "ranking": policy.allow_ranking,
                "chunk": policy.allow_chunk,
                "index_epoch": policy.allow_index_epoch,
                "identity": False,
                "status": False,
                "acl": False,
            }
            unexpected.update(name for name in differences if not allowed[name])
        entries[source_id] = {
            "differences": sorted(differences),
            "allowed": sorted(differences - unexpected),
            "unexpected": sorted(unexpected),
        }
    return ShadowComparison(query=query, entries=entries)


def _by_source_id(
    observations: Sequence[ShadowObservation],
) -> dict[str, ShadowObservation]:
    result: dict[str, ShadowObservation] = {}
    for observation in observations:
        if observation.source_id in result:
            raise ValueError(f"duplicate shadow source_id: {observation.source_id}")
        result[observation.source_id] = observation
    return result


__all__ = [
    "SHADOW_ARTIFACT_FILENAME",
    "SHADOW_SCHEMA_VERSION",
    "ShadowArtifactStore",
    "ShadowComparison",
    "ShadowComparisonPolicy",
    "ShadowObservation",
    "compare_shadow_results",
    "persist_index_then_shadow_artifact",
]
