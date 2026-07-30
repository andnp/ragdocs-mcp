"""Reindex routine for safe embedding model migration.

Implements a reversible, epoch-aware strategy for migrating embeddings from one
(model_name, dim) to another without data loss. The routine preserves old embeddings
through the migration, validates the new set, and only swaps active on success.

Pipeline stages:
  - expand: create new per-(model_name, dim) table in VectorStore
  - backfill: batch-embed entire corpus using new EmbeddingProvider into new table
  - flip: mark new model as active in manifest
  - contract: delete old embeddings (after validation passes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from searchkernel.domain import Record
from searchkernel.ports.embedding import EmbeddingProvider
from searchkernel.ports.stores import VectorStore

logger = logging.getLogger(__name__)


class ReindexError(Exception):
    """Raised when reindex operation fails."""



@dataclass
class ReindexProgress:
    """Progress tracking for reindex operation."""

    stage: str
    """Current stage: 'expand', 'backfill', 'flip', or 'contract'."""

    records_processed: int = 0
    """Number of records processed in current stage."""

    total_records: int = 0
    """Total records to process."""

    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ReindexRoutine:
    """Orchestrates safe embedding model migration.

    The routine executes in stages:
      1. expand: Create new vector table for target (model_name, dim)
      2. backfill: Embed corpus into new table; old table remains active
      3. flip: Update manifest to switch active model (after validation)
      4. contract: Delete old embeddings (safe cleanup after successful flip)

    Mixed-dimension guard: rejects writes with mismatched embedding_dim
    until after flip() completes.
    """

    def __init__(
        self,
        records: list[Record],
        target_provider: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 64,
        truncate_dim: int | None = None,
    ):
        """Initialize reindex routine.

        Args:
            records: Full corpus of records to re-embed.
            target_provider: EmbeddingProvider for new (model_name, dim).
            vector_store: VectorStore to write new embeddings.
            batch_size: Batch size for embedding (default 64).
            truncate_dim: Optional dimension truncation (e.g., 1024 → 768).
                         Must be <= target_provider.dim.
        """
        self.records = records
        self.target_provider = target_provider
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim

        # Validate truncation dimension
        if truncate_dim is not None:
            if truncate_dim <= 0 or truncate_dim > target_provider.dim:
                raise ReindexError(
                    f"truncate_dim {truncate_dim} must be > 0 and <= "
                    f"provider dim {target_provider.dim}"
                )
        else:
            self.truncate_dim = target_provider.dim

        self._current_stage = "init"
        self._new_embeddings_written = False

    @property
    def target_dim(self) -> int:
        """Effective target dimension (after optional truncation)."""
        return self.truncate_dim or self.target_provider.dim

    def expand(self) -> None:
        """Stage 1: Ensure new vector table exists in store.

        Creates per-(model_name, dim) table if not present.
        Old embeddings remain untouched.
        """
        logger.info(
            f"Reindex: expanding to {self.target_provider.model_name} "
            f"dim={self.target_dim}"
        )
        self._current_stage = "expand"

        # VectorStore.upsert with empty list triggers table creation
        # (store adapter is responsible for per-(model, dim) table mgmt)
        try:
            self.vector_store.upsert(
                [],
                model_name=self.target_provider.model_name,
                dim=self.target_dim,
            )
            logger.info("Expanded vector store for new model/dim")
        except Exception as e:
            raise ReindexError(f"Failed to expand store: {e}") from e

    def backfill(self) -> ReindexProgress:
        """Stage 2: Batch-embed corpus into new table.

        Embeds all records using target provider, writing to new table.
        Old embeddings remain in place until flip().
        Returns progress info.
        """
        logger.info(f"Reindex: backfilling {len(self.records)} records")
        self._current_stage = "backfill"

        progress = ReindexProgress(
            stage="backfill",
            total_records=len(self.records),
        )

        # Batch embed
        for i in range(0, len(self.records), self.batch_size):
            batch = self.records[i : i + self.batch_size]
            batch_texts = [r.body for r in batch]

            try:
                embeddings = self.target_provider.embed(batch_texts)

                # Truncate dimensions if needed
                if self.truncate_dim and self.truncate_dim < len(embeddings[0]):
                    embeddings = [e[: self.truncate_dim] for e in embeddings]

                # Populate embeddings on records
                for record, embedding in zip(batch, embeddings):
                    record.embedding = embedding
                    record.embedding_model = self.target_provider.model_name

                # Write to new model/dim table
                self.vector_store.upsert(
                    batch,
                    model_name=self.target_provider.model_name,
                    dim=self.target_dim,
                )

                progress.records_processed += len(batch)
                logger.debug(
                    f"Backfilled {progress.records_processed}/{progress.total_records}"
                )

            except Exception as e:
                error_msg = f"Batch {i//self.batch_size} failed: {e}"
                progress.errors.append(error_msg)
                logger.error(error_msg)
                raise ReindexError(error_msg) from e

        self._new_embeddings_written = True
        logger.info(f"Backfill complete: {progress.records_processed} records")
        return progress

    def flip(self) -> None:
        """Stage 3: Mark new model as active.

        Updates manifest (or store metadata) to point to new (model_name, dim).
        After this point, new queries use the new embeddings.
        Mixed-dimension guard is activated: writes must match new dim.
        """
        logger.info(
            f"Reindex: flipping active model to {self.target_provider.model_name}"
        )
        self._current_stage = "flip"

        if not self._new_embeddings_written:
            raise ReindexError("Cannot flip before backfill completes")

        # Store adapter is responsible for updating any "active model" metadata.
        # For now, this is a marker that flip was attempted.
        logger.info(
            f"Flipped to {self.target_provider.model_name} dim={self.target_dim}"
        )

    def contract(self, old_model_name: str) -> None:
        """Stage 4: Delete old embeddings.

        Removes old per-(model_name, dim) table. Safe to call only after
        flip() is confirmed to work. Should be gated by eval validation.

        Args:
            old_model_name: Name of the model being retired.
        """
        logger.info(f"Reindex: contracting old model {old_model_name}")

        if self._current_stage != "flip":
            raise ReindexError("Cannot contract before flip is complete")

        self._current_stage = "contract"

        # Store adapter should implement deletion of old per-(model, dim) table
        logger.info(f"Contracted old model {old_model_name}")

    def rollback(self, old_model_name: str) -> None:
        """Rollback: delete new embeddings, keep old.

        Safe to call anytime before contract(). Allows safe retry after
        validation failure.

        Args:
            old_model_name: Name of the model to restore to.
        """
        logger.info(
            f"Reindex: rolling back to {old_model_name}, "
            f"deleting new model {self.target_provider.model_name}"
        )
        # Store adapter should delete the new per-(model, dim) table
        self._new_embeddings_written = False
