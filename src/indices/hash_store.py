"""Persistent storage for chunk content hashes."""

import json
import logging
from pathlib import Path

from src.models import Chunk

logger = logging.getLogger(__name__)


class ChunkHashStore:
    """Store and retrieve chunk content hashes for delta detection."""

    def __init__(self, storage_path: Path):
        self._storage_path = storage_path
        self._hashes: dict[str, str] = {}  # chunk_id -> content_hash
        self._reverse_hashes: dict[str, list[str]] = {}  # content_hash -> [chunk_ids]
        self._load()

    def _load(self) -> None:
        """Load hashes from persistent storage with corruption recovery."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r") as f:
                    self._hashes = json.load(f)
                logger.info(
                    f"Loaded {len(self._hashes)} chunk hashes from {self._storage_path}"
                )
                # Build reverse lookup
                self._build_reverse_lookup()
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Failed to load hash store: {e}. Starting fresh.", exc_info=True
                )
                self._hashes = {}
                self._reverse_hashes = {}

    def _build_reverse_lookup(self) -> None:
        """Build reverse lookup from content_hash to chunk_ids."""
        self._reverse_hashes.clear()
        for chunk_id, content_hash in self._hashes.items():
            if content_hash not in self._reverse_hashes:
                self._reverse_hashes[content_hash] = []
            self._reverse_hashes[content_hash].append(chunk_id)

    def persist(self) -> None:
        """Persist hashes to storage."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._storage_path, "w") as f:
                json.dump(self._hashes, f)
            logger.debug(f"Persisted {len(self._hashes)} chunk hashes")
        except OSError as e:
            logger.error(f"Failed to persist hash store: {e}", exc_info=True)

    def get_hash(self, chunk_id: str) -> str | None:
        """Get stored hash for chunk_id."""
        return self._hashes.get(chunk_id)

    def set_hash(self, chunk_id: str, content_hash: str) -> None:
        """Store hash for chunk_id."""
        # Remove old reverse mapping if updating
        old_hash = self._hashes.get(chunk_id)
        if old_hash and old_hash in self._reverse_hashes:
            try:
                self._reverse_hashes[old_hash].remove(chunk_id)
                if not self._reverse_hashes[old_hash]:
                    del self._reverse_hashes[old_hash]
            except ValueError:
                pass

        # Add new mapping
        self._hashes[chunk_id] = content_hash
        if content_hash not in self._reverse_hashes:
            self._reverse_hashes[content_hash] = []
        if chunk_id not in self._reverse_hashes[content_hash]:
            self._reverse_hashes[content_hash].append(chunk_id)

    def remove_document(self, doc_id: str) -> None:
        """Remove all hashes for a document.

        Handles both chunk ID formats:
        - {doc_id}_chunk_{index} (HeaderBasedChunker format)
        - {doc_id}#{separator}{index} (custom formats)
        """
        to_remove = [
            cid
            for cid in self._hashes
            if cid.startswith(f"{doc_id}_chunk_") or cid.startswith(f"{doc_id}#")
        ]
        for cid in to_remove:
            # Remove from reverse lookup
            content_hash = self._hashes.get(cid)
            if content_hash and content_hash in self._reverse_hashes:
                try:
                    self._reverse_hashes[content_hash].remove(cid)
                    if not self._reverse_hashes[content_hash]:
                        del self._reverse_hashes[content_hash]
                except ValueError:
                    pass
            # Remove from forward lookup
            del self._hashes[cid]

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove hash for a specific chunk."""
        content_hash = self._hashes.get(chunk_id)
        if content_hash and content_hash in self._reverse_hashes:
            try:
                self._reverse_hashes[content_hash].remove(chunk_id)
                if not self._reverse_hashes[content_hash]:
                    del self._reverse_hashes[content_hash]
            except ValueError:
                pass
        self._hashes.pop(chunk_id, None)

    def has_changed(self, chunk: Chunk) -> bool:
        """Check if chunk content has changed since last index.

        Returns True if:
        - Chunk is new (no stored hash)
        - Stored hash differs from current hash
        """
        stored_hash = self.get_hash(chunk.chunk_id)
        if stored_hash is None:
            return True
        return stored_hash != chunk.content_hash

    def get_chunk_id_by_hash(self, content_hash: str) -> str | None:
        """Get first chunk_id with matching content hash.

        Returns None if hash not found.
        Used for move detection to find old chunk with same content.
        """
        chunk_ids = self._reverse_hashes.get(content_hash, [])
        return chunk_ids[0] if chunk_ids else None

    def get_chunks_by_document(self, doc_id: str) -> list[tuple[str, str]]:
        """Get all (chunk_id, content_hash) pairs for a document.

        Returns empty list if document not found.
        Used for move detection to compare old vs new document chunks.
        """
        chunks = []
        for chunk_id, content_hash in self._hashes.items():
            if chunk_id.startswith(f"{doc_id}_chunk_") or chunk_id.startswith(f"{doc_id}#"):
                chunks.append((chunk_id, content_hash))
        return chunks
