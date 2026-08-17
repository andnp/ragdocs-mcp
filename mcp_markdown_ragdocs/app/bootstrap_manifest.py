"""Manifest construction and startup rebuild decisions."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from searchkernel.api import (
    CURRENT_MANIFEST_SPEC_VERSION,
    IndexManifest,
    load_manifest,
    should_rebuild,
)

from mcp_markdown_ragdocs.config import Config

logger = logging.getLogger(__name__)


class ManifestHost(Protocol):
    config: Config
    index_manager: Any
    index_path: Path
    fallback_index_path: Path | None
    current_manifest: IndexManifest | None
    _is_virgin_startup: bool


@dataclass
class ManifestCoordinator:
    """Coordinate manifest construction and persisted-index decisions."""

    host: ManifestHost

    def build_manifest(self) -> IndexManifest:
        host = self.host
        saved_manifest = load_manifest(host.index_path)
        return IndexManifest(
            spec_version=CURRENT_MANIFEST_SPEC_VERSION,
            embedding_model=host.config.llm.embedding_model,
            chunking_config={
                "strategy": host.config.chunking.strategy,
                "min_chunk_chars": host.config.chunking.min_chunk_chars,
                "max_chunk_chars": host.config.chunking.max_chunk_chars,
                "overlap_chars": host.config.chunking.overlap_chars,
            },
            active_model=(
                saved_manifest.active_model if saved_manifest is not None else None
            ),
            migration=saved_manifest.migration if saved_manifest is not None else None,
        )

    def check_and_rebuild_if_needed(self) -> bool:
        host = self.host
        host.index_path.mkdir(parents=True, exist_ok=True)
        self.hydrate_index_path_from_fallback()
        host.current_manifest = self.build_manifest()
        saved_manifest = load_manifest(host.index_path)
        host._is_virgin_startup = saved_manifest is None
        return should_rebuild(host.current_manifest, saved_manifest)

    def has_persisted_index_state(self, index_path: Path) -> bool:
        return any(
            candidate.exists()
            for candidate in [
                index_path / "index.manifest.json",
                index_path / "vector" / "docstore.json",
                index_path / "vector" / "faiss_index.bin",
                index_path / "graph" / "graph.json",
            ]
        )

    def hydrate_index_path_from_fallback(self) -> None:
        host = self.host
        if hasattr(host.index_manager, "kernel"):
            # Canonical records use a single SQLite database.  Copying the
            # legacy vector/graph bundle into an override path can overwrite
            # the freshly composed database with an incompatible schema.
            return
        if host.fallback_index_path is None:
            return
        if host.index_path == host.fallback_index_path:
            return
        if self.has_persisted_index_state(host.index_path):
            return
        if not self.has_persisted_index_state(host.fallback_index_path):
            return

        logger.info(
            "Hydrating runtime index at %s from existing persisted index at %s",
            host.index_path,
            host.fallback_index_path,
        )

        for directory_name in ["vector", "keyword", "graph"]:
            source_dir = host.fallback_index_path / directory_name
            if source_dir.exists():
                shutil.copytree(
                    source_dir,
                    host.index_path / directory_name,
                    dirs_exist_ok=True,
                )

        for file_name in [
            "index.manifest.json",
            "index.db",
            "index.db-shm",
            "index.db-wal",
            "chunk_hashes.json",
        ]:
            source_file = host.fallback_index_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, host.index_path / file_name)


__all__ = ["ManifestCoordinator", "ManifestHost"]
