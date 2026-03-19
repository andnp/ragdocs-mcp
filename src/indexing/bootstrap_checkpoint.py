from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from src.indexing.manifest import IndexManifest
from src.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION = "1.0.0"
BOOTSTRAP_CHECKPOINT_FILE_NAME = "bootstrap.checkpoint.json"


@dataclass(frozen=True)
class BootstrapFileStamp:
    relative_path: str
    mtime_ns: int
    size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BootstrapFileStamp:
        relative_path = data["relative_path"]
        mtime_ns = data["mtime_ns"]
        size = data["size"]
        if not isinstance(relative_path, str):
            raise KeyError("relative_path")
        if not isinstance(mtime_ns, int):
            raise KeyError("mtime_ns")
        if not isinstance(size, int):
            raise KeyError("size")
        return cls(
            relative_path=relative_path,
            mtime_ns=mtime_ns,
            size=size,
        )

    def matches(self, other: BootstrapFileStamp) -> bool:
        return self.mtime_ns == other.mtime_ns and self.size == other.size


@dataclass(frozen=True)
class BootstrapCheckpoint:
    schema_version: str
    generation: str
    complete: bool
    targets: dict[str, BootstrapFileStamp]
    completed: dict[str, BootstrapFileStamp]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "complete": self.complete,
            "targets": [
                self.targets[key].to_dict()
                for key in sorted(self.targets)
            ],
            "completed": [
                self.completed[key].to_dict()
                for key in sorted(self.completed)
            ],
        }


def checkpoint_path(index_path: Path) -> Path:
    return index_path / BOOTSTRAP_CHECKPOINT_FILE_NAME


def _common_docs_root(documents_roots: list[Path]) -> Path | None:
    if not documents_roots:
        return None
    if len(documents_roots) == 1:
        return documents_roots[0].resolve()
    common = os.path.commonpath([str(root.resolve()) for root in documents_roots])
    return Path(common).resolve()


def _relative_path_for_roots(
    file_path: str | Path,
    documents_roots: list[Path],
) -> str | None:
    resolved_path = Path(file_path).resolve()
    common_root = _common_docs_root(documents_roots)
    if common_root is None:
        return None

    for root in documents_roots:
        try:
            resolved_path.relative_to(root.resolve())
            return str(resolved_path.relative_to(common_root))
        except ValueError:
            continue

    return None


def build_file_stamps(
    file_paths: list[str],
    documents_roots: list[Path],
) -> dict[str, BootstrapFileStamp]:
    stamps: dict[str, BootstrapFileStamp] = {}
    for file_path in file_paths:
        relative_path = _relative_path_for_roots(file_path, documents_roots)
        if relative_path is None:
            logger.warning(
                "Bootstrap checkpoint skipping file outside configured roots: %s",
                file_path,
            )
            continue

        try:
            stat_result = Path(file_path).stat()
        except OSError:
            logger.warning(
                "Bootstrap checkpoint could not stat file: %s",
                file_path,
                exc_info=True,
            )
            continue

        stamps[relative_path] = BootstrapFileStamp(
            relative_path=relative_path,
            mtime_ns=stat_result.st_mtime_ns,
            size=stat_result.st_size,
        )

    return stamps


def compute_bootstrap_generation(
    manifest: IndexManifest,
    target_stamps: dict[str, BootstrapFileStamp],
) -> str:
    payload = {
        "manifest": {
            "spec_version": manifest.spec_version,
            "embedding_model": manifest.embedding_model,
            "chunking_config": manifest.chunking_config,
        },
        "target_paths": sorted(target_stamps),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def save_bootstrap_checkpoint(index_path: Path, checkpoint: BootstrapCheckpoint) -> None:
    atomic_write_json(checkpoint_path(index_path), checkpoint.to_dict())


def load_bootstrap_checkpoint(index_path: Path) -> BootstrapCheckpoint | None:
    path = checkpoint_path(index_path)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema_version = data["schema_version"]
        generation = data["generation"]
        complete = data.get("complete", False)
        targets = _load_stamp_entries(data.get("targets", []))
        completed = _load_stamp_entries(data.get("completed", []))
        if not isinstance(schema_version, str):
            raise KeyError("schema_version")
        if not isinstance(generation, str):
            raise KeyError("generation")
        if not isinstance(complete, bool):
            raise KeyError("complete")
        return BootstrapCheckpoint(
            schema_version=schema_version,
            generation=generation,
            complete=complete,
            targets=targets,
            completed=completed,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        logger.warning(
            "Bootstrap checkpoint is missing or corrupted at %s; ignoring it",
            path,
            exc_info=True,
        )
        return None


def _load_stamp_entries(entries: object) -> dict[str, BootstrapFileStamp]:
    if not isinstance(entries, list):
        raise KeyError("entries")

    stamps: dict[str, BootstrapFileStamp] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise KeyError("entry")
        stamp = BootstrapFileStamp.from_dict(entry)
        stamps[stamp.relative_path] = stamp
    return stamps


def prepare_bootstrap_checkpoint(
    index_path: Path,
    generation: str,
    target_stamps: dict[str, BootstrapFileStamp],
) -> BootstrapCheckpoint:
    checkpoint = load_bootstrap_checkpoint(index_path)
    if checkpoint is None or checkpoint.generation != generation:
        checkpoint = BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation=generation,
            complete=not target_stamps,
            targets=dict(target_stamps),
            completed={},
        )
        save_bootstrap_checkpoint(index_path, checkpoint)
        return checkpoint

    valid_completed = {
        relative_path: stamp
        for relative_path, stamp in checkpoint.completed.items()
        if relative_path in target_stamps and stamp.matches(target_stamps[relative_path])
    }
    updated_checkpoint = BootstrapCheckpoint(
        schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        generation=generation,
        complete=len(valid_completed) == len(target_stamps),
        targets=dict(target_stamps),
        completed=valid_completed,
    )
    if updated_checkpoint != checkpoint:
        save_bootstrap_checkpoint(index_path, updated_checkpoint)
    return updated_checkpoint


def mark_bootstrap_file_completed(
    index_path: Path,
    documents_roots: list[Path],
    file_path: str,
) -> bool:
    checkpoint = load_bootstrap_checkpoint(index_path)
    if checkpoint is None:
        return False

    current_stamps = build_file_stamps([file_path], documents_roots)
    if not current_stamps:
        return False

    stamp = next(iter(current_stamps.values()))
    if stamp.relative_path not in checkpoint.targets:
        return False

    updated_targets = dict(checkpoint.targets)
    updated_targets[stamp.relative_path] = stamp
    updated_completed = dict(checkpoint.completed)
    updated_completed[stamp.relative_path] = stamp
    updated_checkpoint = BootstrapCheckpoint(
        schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        generation=checkpoint.generation,
        complete=len(updated_completed) == len(updated_targets),
        targets=updated_targets,
        completed=updated_completed,
    )
    if updated_checkpoint == checkpoint:
        return False

    save_bootstrap_checkpoint(index_path, updated_checkpoint)
    return True


def has_incomplete_bootstrap_checkpoint(index_path: Path) -> bool:
    checkpoint = load_bootstrap_checkpoint(index_path)
    if checkpoint is None:
        return False
    return not checkpoint.complete and bool(checkpoint.targets)