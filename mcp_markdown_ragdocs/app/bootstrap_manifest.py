"""Manifest construction and startup rebuild decisions."""

from __future__ import annotations

import logging
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
        host.current_manifest = self.build_manifest()
        saved_manifest = load_manifest(host.index_path)
        host._is_virgin_startup = saved_manifest is None
        return should_rebuild(host.current_manifest, saved_manifest)


__all__ = ["ManifestCoordinator", "ManifestHost"]
