"""
Memory system initialization helpers.

These helpers ensure consistent memory system initialization.
Memory indices are stored at `memory_path/indices/`.

See ADR-022 for architectural rationale.
"""

import logging
from pathlib import Path

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.manager import MemoryIndexManager
from src.memory.search import MemorySearchOrchestrator
from src.storage.db import DatabaseManager

logger = logging.getLogger(__name__)


def create_memory_indices(
    embedding_model_name: str,
    embedding_workers: int,
    db_manager: DatabaseManager | None = None,
) -> tuple[VectorIndex, KeywordIndex, GraphStore]:
    """Create fresh memory indices."""
    vector = VectorIndex(
        embedding_model_name=embedding_model_name,
        embedding_workers=embedding_workers,
    )
    keyword = KeywordIndex(db_manager)
    graph = GraphStore(db_manager)
    return vector, keyword, graph


def load_or_rebuild_memory_indices(
    manager: MemoryIndexManager,
) -> int:
    """Load memory indices or rebuild from files if needed.

    This is the canonical way to initialize memory indices. It handles:
    1. Loading persisted indices from `memory_path/indices/`
    2. Running reconciliation to detect new files
    3. Rebuilding from scratch if indices are corrupted/missing

    Args:
        manager: The MemoryIndexManager to initialize

    Returns:
        Number of memories indexed (0 if loaded from existing indices
        with no reconciliation needed)
    """
    try:
        manager.load()
        reindexed = manager.reconcile()
        if reindexed > 0:
            manager.persist()
        logger.info(
            "Memory system loaded: %d indexed, %d reconciled",
            manager.get_memory_count(),
            reindexed,
        )
        return reindexed
    except Exception as e:
        logger.warning("Failed to load memory indices, rebuilding: %s", e)
        count = manager.reindex_all()
        manager.persist()
        logger.info("Memory system rebuilt: %d memories indexed", count)
        return count


def create_memory_system(
    config: Config,
    memory_path: Path,
    embedding_model_name: str,
) -> tuple[MemoryIndexManager, MemorySearchOrchestrator]:
    """Create complete memory system with manager and search orchestrator.

    This is the canonical way to create the memory system components.
    After calling this, use `load_or_rebuild_memory_indices()` to
    populate the indices.

    Args:
        config: Application configuration
        memory_path: Path to memory storage directory
        embedding_model_name: Embedding model to use

    Returns:
        Tuple of (MemoryIndexManager, MemorySearchOrchestrator)
    """
    vector, keyword, graph = create_memory_indices(
        embedding_model_name=embedding_model_name,
        embedding_workers=config.indexing.embedding_workers,
        db_manager=DatabaseManager(memory_path / "indices" / "memory.db"),
    )

    manager = MemoryIndexManager(
        config=config,
        memory_path=memory_path,
        vector=vector,
        keyword=keyword,
        graph=graph,
    )

    search = MemorySearchOrchestrator(
        vector=vector,
        keyword=keyword,
        graph=graph,
        config=config,
        manager=manager,
        documents_path=memory_path,
    )

    return manager, search
