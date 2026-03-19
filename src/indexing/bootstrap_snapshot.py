from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.indexing.bootstrap_checkpoint import BootstrapCheckpoint, BootstrapFileStamp
from src.indexing.manifest import IndexManifest

PublicIndexStatus = Literal["indexing", "partial", "ready"]


@dataclass(frozen=True)
class PublicIndexStateSnapshot:
    status: PublicIndexStatus
    indexed_count: int
    total_count: int


@dataclass(frozen=True)
class BootstrapReadinessSnapshot:
    total_targets: int
    durably_completed_targets: int
    loaded_indexed_count: int
    queryable: bool
    public_state: PublicIndexStateSnapshot


def compute_bootstrap_completed_paths(
    checkpoint: BootstrapCheckpoint | None,
    saved_manifest: IndexManifest | None,
    target_stamps: dict[str, BootstrapFileStamp],
) -> set[str]:
    if checkpoint is None or saved_manifest is None:
        return set()

    indexed_files = saved_manifest.indexed_files or {}
    completed_paths: set[str] = set()
    for relative_path, stamp in checkpoint.completed.items():
        if relative_path not in target_stamps:
            continue
        if not stamp.matches(target_stamps[relative_path]):
            continue

        doc_id = str(Path(relative_path).with_suffix(""))
        if indexed_files.get(doc_id) != relative_path:
            continue

        completed_paths.add(relative_path)

    return completed_paths


def derive_loaded_index_state_snapshot(
    total_targets: int,
    loaded_indexed_count: int,
) -> PublicIndexStateSnapshot:
    status: PublicIndexStatus = "ready"
    if total_targets > 0 and loaded_indexed_count < total_targets:
        status = "partial"

    return PublicIndexStateSnapshot(
        status=status,
        indexed_count=loaded_indexed_count,
        total_count=total_targets,
    )


def derive_bootstrap_readiness_snapshot(
    checkpoint: BootstrapCheckpoint | None,
    saved_manifest: IndexManifest | None,
    target_stamps: dict[str, BootstrapFileStamp],
    *,
    loaded_indexed_count: int,
    queryable: bool,
    rebuild_pending: bool,
) -> BootstrapReadinessSnapshot | None:
    completed_paths = compute_bootstrap_completed_paths(
        checkpoint,
        saved_manifest,
        target_stamps,
    )
    durably_completed_targets = len(completed_paths)
    total_targets = len(target_stamps)
    indexed_count = max(loaded_indexed_count, durably_completed_targets)

    if total_targets > 0:
        indexed_count = min(indexed_count, total_targets)

    if indexed_count == 0 and total_targets > 0:
        return None

    if total_targets == 0:
        status: PublicIndexStatus = "ready"
    elif indexed_count < total_targets:
        status = "partial"
    elif rebuild_pending:
        status = "indexing"
    else:
        status = "ready"

    return BootstrapReadinessSnapshot(
        total_targets=total_targets,
        durably_completed_targets=durably_completed_targets,
        loaded_indexed_count=loaded_indexed_count,
        queryable=queryable,
        public_state=PublicIndexStateSnapshot(
            status=status,
            indexed_count=indexed_count,
            total_count=total_targets,
        ),
    )