import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import Config
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.memory.manager import MemoryIndexManager
from src.memory.models import (
    LinkedMemoryResult,
    MemoryFrontmatter,
    MemorySearchResult,
)
from src.search.base_orchestrator import BaseSearchOrchestrator
from src.search.score_pipeline import ScorePipelineConfig
from src.utils.similarity import cosine_similarity_lists

logger = logging.getLogger(__name__)


def apply_recency_boost(
    score: float,
    created_at: datetime | None,
    boost_window_days: int,
    max_boost_amount: float,
    boost_decay_rate: float,
) -> float:
    """
    Apply exponential additive recency boost to memory score.

    Recent memories receive an exponentially decaying bonus ADDED to their base score.
    Old memories (beyond boost window) retain their base score without penalty.

    Args:
        score: Base calibrated score from search
        created_at: Memory creation timestamp
        boost_window_days: Days within which to apply boost
        max_boost_amount: Maximum bonus (at age=0 days)
        boost_decay_rate: Exponential decay rate (e.g., 0.95)

    Returns:
        Score with additive recency boost (base + exponential bonus)
    """
    if created_at is None:
        return score

    # Ensure timezone-aware datetime
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    # Calculate age in days
    age_days = (datetime.now(timezone.utc) - created_at).days

    # Apply exponential additive boost for recent memories
    if age_days <= boost_window_days:
        # Exponential decay of boost amount (not score!)
        boost_factor = boost_decay_rate ** age_days
        bonus = boost_factor * max_boost_amount
        boosted_score = min(1.0, score + bonus)  # Cap at 1.0
        return boosted_score
    else:
        # Old memories: no boost, but NO PENALTY either!
        return score


def _normalize_time_filters(
    after_timestamp: int | None,
    before_timestamp: int | None,
    relative_days: int | None,
) -> tuple[int | None, int | None]:
    """
    Validate and normalize time filter parameters.

    Args:
        after_timestamp: Unix timestamp for lower bound (inclusive)
        before_timestamp: Unix timestamp for upper bound (exclusive)
        relative_days: Number of days back from now (overrides absolute timestamps)

    Returns:
        Tuple of (after_timestamp, before_timestamp) with relative_days applied

    Raises:
        ValueError: If timestamps are invalid or relative_days is negative
    """
    # Validate timestamp range
    if after_timestamp is not None and before_timestamp is not None:
        if after_timestamp >= before_timestamp:
            raise ValueError("after_timestamp must be less than before_timestamp")

    # Handle relative_days (overrides absolute timestamps)
    if relative_days is not None:
        if relative_days < 0:
            raise ValueError("relative_days must be non-negative")
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=relative_days)
        return int(cutoff.timestamp()), None

    return after_timestamp, before_timestamp


def _get_filtering_timestamp(chunk_data: dict) -> datetime | None:
    """
    Extract timestamp for time filtering, with fallback to file mtime.

    Args:
        chunk_data: Chunk metadata dictionary

    Returns:
        Datetime for filtering, or None if unavailable
    """
    metadata = chunk_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    # Try memory_created_at from frontmatter first
    created_at_str = metadata.get("memory_created_at")
    if created_at_str:
        try:
            return datetime.fromisoformat(created_at_str)
        except ValueError:
            pass

    # Fallback to file mtime
    file_path = chunk_data.get("file_path")
    if file_path and isinstance(file_path, str):
        path = Path(file_path)
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)

    return None


def _passes_time_filter(
    filtering_timestamp: datetime | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
) -> bool:
    """
    Check if a timestamp passes the time filter criteria.

    Args:
        filtering_timestamp: The datetime to check
        after_timestamp: Lower bound (inclusive), None to skip
        before_timestamp: Upper bound (exclusive), None to skip

    Returns:
        True if timestamp passes filter (or if no timestamp available)
    """
    if filtering_timestamp is None:
        return True

    # Normalize timezone
    if filtering_timestamp.tzinfo is None:
        filtering_timestamp = filtering_timestamp.replace(tzinfo=timezone.utc)

    timestamp = int(filtering_timestamp.timestamp())

    if after_timestamp is not None and timestamp < after_timestamp:
        return False

    if before_timestamp is not None and timestamp > before_timestamp:
        return False

    return True


class MemorySearchOrchestrator(BaseSearchOrchestrator[MemorySearchResult]):
    def __init__(
        self,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
        config: Config,
        manager: MemoryIndexManager,
        documents_path: Path | None = None,
    ):
        super().__init__(vector, keyword, graph, config, documents_path)
        self._manager = manager
        self._pending_reindex: set[str] = set()
        self._reindex_tasks: set[asyncio.Task] = set()

    def _build_score_pipeline_config(
        self, weights: dict[str, float]
    ) -> ScorePipelineConfig:
        return ScorePipelineConfig(
            rrf_k=self._config.search.rrf_k_constant,
            strategy_weights=weights,
            use_dynamic_weights=False,
            calibration_threshold=self._config.search.score_calibration_threshold,
            calibration_steepness=self._config.search.score_calibration_steepness,
        )

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        filter_type: str | None = None,
        load_full_memory: bool = False,
        after_timestamp: int | None = None,
        before_timestamp: int | None = None,
        relative_days: int | None = None,
    ) -> list[MemorySearchResult]:
        if not query or not query.strip():
            return []

        after_timestamp, before_timestamp = _normalize_time_filters(
            after_timestamp, before_timestamp, relative_days
        )

        top_k = max(20, limit * 4)

        ctx = await self._execute_parallel_search(query, top_k)

        self._apply_tag_expansion(ctx, top_k)

        strategy_results = self._build_strategy_results(ctx)

        weights = self._get_base_weights()

        fused = self._apply_score_pipeline(strategy_results, weights)

        memory_results: list[MemorySearchResult] = []

        missing_chunk_ids: list[str] = []
        for chunk_id, score in fused:
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                missing_chunk_ids.append(chunk_id)
                continue

            metadata = chunk_data.get("metadata", {})

            # Type guard: ensure metadata is a dict
            if not isinstance(metadata, dict):
                continue

            if filter_type and metadata.get("memory_type") != filter_type:
                continue

            # Time filtering (with fallback to file mtime)
            if after_timestamp is not None or before_timestamp is not None:
                filtering_timestamp = _get_filtering_timestamp(chunk_data)
                if not _passes_time_filter(filtering_timestamp, after_timestamp, before_timestamp):
                    continue

            # Extract created_at for recency boost
            created_at = None
            created_at_str = metadata.get("memory_created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except ValueError:
                    pass

            memory_type = metadata.get("memory_type", "journal")
            recency_config = self._config.memory.get_recency_config(memory_type)

            boosted_score = apply_recency_boost(
                score,
                created_at,
                recency_config.boost_window_days,
                recency_config.max_boost_amount,
                recency_config.boost_decay_rate,
            )

            # Apply threshold filtering (post-boost)
            if boosted_score < self._config.memory.score_threshold:
                continue

            frontmatter = MemoryFrontmatter(
                type=memory_type,
                status=metadata.get("memory_status", "active"),
                tags=metadata.get("memory_tags", []),
                created_at=created_at,
            )

            content = str(chunk_data.get("content", ""))
            if load_full_memory:
                file_path = chunk_data.get("file_path")
                if file_path and isinstance(file_path, str):
                    try:
                        full_content = Path(file_path).read_text(encoding="utf-8")
                        content = full_content
                    except Exception as e:
                        logger.warning(f"Failed to load full memory from {file_path}: {e}")

            memory_results.append(MemorySearchResult(
                memory_id=str(chunk_data.get("doc_id", chunk_id)),
                score=boosted_score,
                content=content,
                frontmatter=frontmatter,
                file_path=str(chunk_data.get("file_path", "")),
                header_path=str(chunk_data.get("header_path", "")),
            ))

            if len(memory_results) >= limit:
                break

        memory_results.sort(key=lambda r: r.score, reverse=True)
        if missing_chunk_ids:
            self._queue_reindex_for_chunks(missing_chunk_ids, "docstore lookup failed")
        return memory_results[:limit]

    async def search_linked_memories(
        self,
        query: str,
        target_document: str,
        limit: int = 5,
    ) -> list[LinkedMemoryResult]:
        ghost_id = f"ghost:{target_document}"

        edges = self._graph.get_edges_to(ghost_id)
        if not edges:
            return []

        memory_ids = [edge["source"] for edge in edges]
        edge_map = {edge["source"]: edge for edge in edges}

        memory_chunks: list[tuple[str, str, float]] = []

        missing_chunk_ids: list[str] = []
        for memory_id in memory_ids:
            chunk_ids = self._vector.get_chunk_ids_for_document(memory_id)

            for chunk_id in chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_id)
                if not chunk_data:
                    missing_chunk_ids.append(chunk_id)
                    continue

                content = str(chunk_data.get("content", ""))
                if query.lower() in content.lower():
                    memory_chunks.append((memory_id, chunk_id, 1.0))
                else:
                    memory_chunks.append((memory_id, chunk_id, 0.5))

        if query.strip():
            query_embedding = self._vector.get_text_embedding(query)
            if query_embedding:
                scored_chunks: list[tuple[str, str, float]] = []
                for memory_id, chunk_id, base_score in memory_chunks:
                    chunk_embedding = self._vector.get_embedding_for_chunk(chunk_id)
                    if chunk_embedding:
                        similarity = cosine_similarity_lists(query_embedding, chunk_embedding)
                        scored_chunks.append((memory_id, chunk_id, similarity))
                    else:
                        scored_chunks.append((memory_id, chunk_id, base_score))

                memory_chunks = scored_chunks

        memory_chunks.sort(key=lambda x: x[2], reverse=True)

        seen_memories: set[str] = set()
        results: list[LinkedMemoryResult] = []

        for memory_id, chunk_id, score in memory_chunks:
            if memory_id in seen_memories:
                continue

            seen_memories.add(memory_id)

            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                missing_chunk_ids.append(chunk_id)
                continue

            edge = edge_map.get(memory_id, {})

            results.append(LinkedMemoryResult(
                memory_id=memory_id,
                score=score,
                content=str(chunk_data.get("content", "")),
                anchor_context=edge.get("edge_context", ""),
                edge_type=edge.get("edge_type", "related_to"),
                file_path=str(chunk_data.get("file_path", "")),
            ))

            if len(results) >= limit:
                break

        if missing_chunk_ids:
            self._queue_reindex_for_chunks(missing_chunk_ids, "docstore lookup failed")
        return results

    async def search_by_tag_cluster(
        self, tag: str, depth: int = 2, limit: int = 10
    ) -> list[MemorySearchResult]:
        from src.memory.link_parser import normalize_tag

        normalized_tag = normalize_tag(tag)
        tag_id = f"tag:{normalized_tag}"

        if not self._graph.has_node(tag_id):
            return []

        depth = min(depth, 3)

        memory_ids = set()
        current_level = {tag_id}

        for _ in range(depth):
            next_level = set()
            for node_id in current_level:
                if node_id.startswith("tag:"):
                    predecessors = [p for p in self._graph.get_neighbors(node_id, 1) if p.startswith("memory:")]
                    memory_ids.update(predecessors)

                    tag_neighbors = [p for p in self._graph.get_neighbors(node_id, 1) if p.startswith("tag:")]
                    next_level.update(tag_neighbors)

            current_level = next_level

        if not memory_ids:
            return []

        results: list[MemorySearchResult] = []

        missing_chunk_ids: list[str] = []
        for memory_id in list(memory_ids)[:limit]:
            chunk_ids = self._vector.get_chunk_ids_for_document(memory_id)
            if not chunk_ids:
                continue

            chunk_id = chunk_ids[0]
            chunk_data = self._vector.get_chunk_by_id(chunk_id)
            if not chunk_data:
                missing_chunk_ids.append(chunk_id)
                continue

            metadata = chunk_data.get("metadata", {})
            # Type guard: ensure metadata is a dict
            if not isinstance(metadata, dict):
                continue

            created_at_str = metadata.get("memory_created_at")
            created_at = None
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except ValueError:
                    pass

            frontmatter = MemoryFrontmatter(
                type=metadata.get("memory_type", "journal"),
                status=metadata.get("memory_status", "active"),
                tags=metadata.get("memory_tags", []),
                created_at=created_at,
            )

            results.append(MemorySearchResult(
                memory_id=memory_id,
                score=1.0,
                content=str(chunk_data.get("content", "")),
                frontmatter=frontmatter,
                file_path=str(chunk_data.get("file_path", "")),
                header_path=str(chunk_data.get("header_path", "")),
            ))

            if len(results) >= limit:
                break

        if missing_chunk_ids:
            self._queue_reindex_for_chunks(missing_chunk_ids, "docstore lookup failed")
        return results

    def _extract_memory_id_from_chunk_id(self, chunk_id: str):
        if "_chunk_" in chunk_id:
            return chunk_id.split("_chunk_", 1)[0]
        return chunk_id

    def _queue_reindex_for_chunks(self, chunk_ids: list[str], reason: str):
        memory_ids = {
            self._extract_memory_id_from_chunk_id(chunk_id)
            for chunk_id in chunk_ids
            if chunk_id
        }

        if not memory_ids:
            return

        pending: list[str] = []
        for memory_id in memory_ids:
            if memory_id and memory_id not in self._pending_reindex:
                self._pending_reindex.add(memory_id)
                pending.append(memory_id)

        if not pending:
            return

        logger.warning(
            "Detected %d missing memory chunks; scheduling reindex for %d memories (%s)",
            len(chunk_ids),
            len(pending),
            reason,
        )
        try:
            task = asyncio.create_task(self._run_reindex(pending, reason))
        except RuntimeError:
            self._reindex_memories_sync(pending, reason)
            return

        self._reindex_tasks.add(task)
        task.add_done_callback(lambda finished: self._reindex_tasks.discard(finished))

    async def _run_reindex(self, memory_ids: list[str], reason: str):
        try:
            await asyncio.to_thread(self._reindex_memories_sync, memory_ids, reason)
        finally:
            for memory_id in memory_ids:
                self._pending_reindex.discard(memory_id)

    def _reindex_memories_sync(self, memory_ids: list[str], reason: str):
        reindexed = 0
        for memory_id in memory_ids:
            if self._manager.reindex_memory(memory_id, reason=reason):
                reindexed += 1

        if reindexed > 0:
            self._manager.persist()
            logger.info("Reindexed %d memories after missing chunk recovery", reindexed)

    async def drain_reindex(self, timeout: float | None = None):
        tasks = [task for task in self._reindex_tasks if not task.done()]
        if not tasks:
            return 0

        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return len(tasks)

        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        return len(done)

    def get_related_tags(self, tag: str) -> list[tuple[str, int]]:
        from src.memory.link_parser import normalize_tag

        normalized_tag = normalize_tag(tag)
        tag_id = f"tag:{normalized_tag}"

        if not self._graph.has_node(tag_id):
            return []

        memory_ids = [p for p in self._graph.get_neighbors(tag_id, 1) if p.startswith("memory:")]

        tag_counts: dict[str, int] = {}

        for memory_id in memory_ids:
            tag_neighbors = [p for p in self._graph.get_neighbors(memory_id, 1) if p.startswith("tag:")]
            for tag_node_id in tag_neighbors:
                if tag_node_id == tag_id:
                    continue

                tag_name = tag_node_id[4:]
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1

        return sorted(tag_counts.items(), key=lambda x: -x[1])

    def get_tag_frequency_map(self) -> dict[str, int]:
        tag_counts: dict[str, int] = {}

        for node_id in self._graph._graph.nodes():
            if node_id.startswith("tag:"):
                tag_name = node_id[4:]
                memory_ids = [p for p in self._graph.get_neighbors(node_id, 1) if p.startswith("memory:")]
                tag_counts[tag_name] = len(memory_ids)

        return tag_counts

    def get_memory_versions(self, memory_id: str) -> dict:
        if not self._graph.has_node(memory_id):
            return {"error": f"Memory not found: {memory_id}"}

        chain: list[dict] = []
        current_id = memory_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)

            chunk_ids = self._vector.get_chunk_ids_for_document(current_id)
            if chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                if chunk_data:
                    chain.append({
                        "memory_id": current_id,
                        "file_path": str(chunk_data.get("file_path", "")),
                    })

            edges = self._graph._graph.out_edges(current_id, data=True)
            supersedes_edges = [e for e in edges if e[2].get("edge_type") == "SUPERSEDES"]

            if supersedes_edges:
                current_id = supersedes_edges[0][1]
            else:
                break

        return {"version_chain": chain, "count": len(chain)}

    def get_memory_dependencies(self, memory_id: str) -> list[dict]:
        if not self._graph.has_node(memory_id):
            return []

        dependencies: list[dict] = []
        edges = self._graph._graph.out_edges(memory_id, data=True)
        depends_edges = [e for e in edges if e[2].get("edge_type") == "DEPENDS_ON"]

        for _, target_id, edge_data in depends_edges:
            chunk_ids = self._vector.get_chunk_ids_for_document(target_id)
            if chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                if chunk_data:
                    dependencies.append({
                        "memory_id": target_id,
                        "file_path": str(chunk_data.get("file_path", "")),
                        "context": edge_data.get("edge_context", ""),
                    })

        return dependencies

    def detect_contradictions(self, memory_id: str) -> list[dict]:
        if not self._graph.has_node(memory_id):
            return []

        contradictions: list[dict] = []
        edges = self._graph._graph.out_edges(memory_id, data=True)
        contradict_edges = [e for e in edges if e[2].get("edge_type") == "CONTRADICTS"]

        for _, target_id, edge_data in contradict_edges:
            chunk_ids = self._vector.get_chunk_ids_for_document(target_id)
            if chunk_ids:
                chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                if chunk_data:
                    contradictions.append({
                        "memory_id": target_id,
                        "file_path": str(chunk_data.get("file_path", "")),
                        "context": edge_data.get("edge_context", ""),
                    })

        return contradictions

    def get_memory_relationships(
        self, memory_id: str, relationship_type: str | None = None
    ) -> dict:
        """
        Get memory relationships by edge type.

        Args:
            memory_id: Memory ID to query relationships for
            relationship_type: Type of relationship - "supersedes", "depends_on", "contradicts", or None for all

        Returns:
            Dictionary with relationship types as keys and lists of related memories as values.
            For "supersedes", returns a "version_chain" dict instead of a list.
            Returns {"error": str} if memory not found or invalid relationship type.
        """
        if not self._graph.has_node(memory_id):
            return {"error": f"Memory not found: {memory_id}"}

        # Map of edge types to query
        edge_type_map = {
            "supersedes": "SUPERSEDES",
            "depends_on": "DEPENDS_ON",
            "contradicts": "CONTRADICTS",
        }

        # Determine which edge types to query
        if relationship_type:
            if relationship_type not in edge_type_map:
                return {"error": f"Invalid relationship type: {relationship_type}. Must be one of: {', '.join(edge_type_map.keys())}"}
            edge_types_to_query = {relationship_type: edge_type_map[relationship_type]}
        else:
            edge_types_to_query = edge_type_map

        result: dict[str, list[dict] | dict] = {}

        # Handle SUPERSEDES specially (builds a chain)
        if "supersedes" in edge_types_to_query:
            chain: list[dict] = []
            current_id = memory_id
            visited = set()

            while current_id and current_id not in visited:
                visited.add(current_id)

                chunk_ids = self._vector.get_chunk_ids_for_document(current_id)
                if chunk_ids:
                    chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                    if chunk_data:
                        chain.append({
                            "memory_id": current_id,
                            "file_path": str(chunk_data.get("file_path", "")),
                        })

                edges = self._graph._graph.out_edges(current_id, data=True)
                supersedes_edges = [e for e in edges if e[2].get("edge_type") == "SUPERSEDES"]

                if supersedes_edges:
                    current_id = supersedes_edges[0][1]
                else:
                    break

            result["supersedes"] = {"version_chain": chain, "count": len(chain)}

        # Handle other edge types (simple list)
        for rel_type, edge_type in edge_types_to_query.items():
            if rel_type == "supersedes":
                continue  # Already handled above

            relationships: list[dict] = []
            edges = self._graph._graph.out_edges(memory_id, data=True)
            matching_edges = [e for e in edges if e[2].get("edge_type") == edge_type]

            for _, target_id, edge_data in matching_edges:
                chunk_ids = self._vector.get_chunk_ids_for_document(target_id)
                if chunk_ids:
                    chunk_data = self._vector.get_chunk_by_id(chunk_ids[0])
                    if chunk_data:
                        relationships.append({
                            "memory_id": target_id,
                            "file_path": str(chunk_data.get("file_path", "")),
                            "context": edge_data.get("edge_context", ""),
                        })

            result[rel_type] = relationships

        return result
